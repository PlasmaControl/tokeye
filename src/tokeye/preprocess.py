"""Turn user data into the spectrogram the model sees.

:func:`prepare` is the one preprocessing entry point shared by the Python
API, the CLI, batch runs and the app:

- a 1D signal becomes a log-magnitude STFT (``kind="stft"``);
- a 1D signal plus ``reference=`` becomes a cross-power STFT
  (``kind="cross"``);
- a 2D array is taken as a ready spectrogram (``kind="spectrogram"``),
  ``log1p``-scaled first when ``config.log`` is on.

The STFT frames are centred as in the model's training (see
:func:`tokeye.transforms.compute_stft`), so ``N`` samples give
``1 + N // hop`` columns for an even ``n_fft``. Every kind has the same
axes: row ``r`` is FFT bin ``b = r + 1`` with ``clip_dc`` (else ``b = r``),
at ``b * fs / n_fft`` Hz, and column ``j`` is at ``j * hop / fs`` seconds
from the first sample. Without ``fs`` they are the bin and frame indices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .config import SpectrogramConfig
from .transforms import compute_stft, log_scale

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

KINDS = ("stft", "cross", "spectrogram")


@dataclass(frozen=True, eq=False)
class Spectrogram:
    """A model-ready spectrogram and its axes.

    Attributes
    ----------
    values
        ``(H, W)`` float32 image, rows = frequency (row 0 lowest).
    freqs
        ``(H,)`` row centres: FFT bin ``b`` (``r + 1`` with ``clip_dc``,
        else ``r``) at ``b * fs / n_fft`` Hz when ``fs`` is known, else the
        bin index ``b``.
    times
        ``(W,)`` column centres: ``j * hop / fs`` seconds from the first
        sample when ``fs`` is known, else the frame index ``j``.
    kind
        ``"stft"``, ``"cross"`` or ``"spectrogram"``.
    config
        The :class:`~tokeye.config.SpectrogramConfig` used.
    fs
        Sampling rate in Hz, or ``None`` when unknown.
    n_samples
        Length of the 1D signal (and of its reference) the values came
        from, or ``None`` for 2D input. The signal lasts ``n_samples / fs``
        seconds; the columns span only ``n_samples // hop`` hops.
    """

    values: np.ndarray
    freqs: np.ndarray
    times: np.ndarray
    kind: str
    config: SpectrogramConfig
    fs: float | None
    n_samples: int | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def __repr__(self) -> str:
        return (
            f"Spectrogram(kind={self.kind!r}, shape={np.shape(self.values)}, "
            f"{_axes_text(self.fs, self.freqs, self.times)})"
        )


def _axis_span(axis: Any, unit: str | None, index: str) -> str:
    """``"62.5 to 4000 Hz"``, or ``"bins 1 to 64"`` (``index``) without a unit."""
    values = np.asarray(axis).ravel()
    if values.size == 0:
        return f"{index} (none)" if unit is None else "(none)"
    span = f"{float(values[0]):g} to {float(values[-1]):g}"
    return f"{index} {span}" if unit is None else f"{span} {unit}"


def _axes_text(fs: float | None, freqs: Any, times: Any) -> str:
    """The ``fs=..., freqs=..., times=...`` part of a repr, with units."""
    if fs is None:
        return (
            f"fs=None, freqs={_axis_span(freqs, None, 'bins')}, "
            f"times={_axis_span(times, None, 'frames')}"
        )
    return (
        f"fs={float(fs):g} Hz, freqs={_axis_span(freqs, 'Hz', '')}, "
        f"times={_axis_span(times, 's', '')}"
    )


def prepare(
    data: Any,
    config: SpectrogramConfig | Mapping[str, Any] | None = None,
    *,
    fs: float | None = None,
    reference: Any = None,
) -> Spectrogram:
    """Build the :class:`Spectrogram` the model will see.

    Parameters
    ----------
    data
        A 1D signal or a 2D ``(freq, time)`` spectrogram (array-like, real).
        A 1D signal becomes centred STFT frames (see
        :func:`tokeye.transforms.compute_stft`): ``1 + N // hop`` columns
        for ``N`` samples and an even ``n_fft``.
    config
        Preprocessing settings; ``None`` uses the defaults.
    fs
        Sampling rate in Hz. Only labels the axes; it never changes values:
        row ``r`` is FFT bin ``b`` (``r + 1`` with ``clip_dc``) at
        ``b * fs / n_fft`` Hz, and column ``j`` is at ``j * hop / fs``
        seconds. For 2D input the axes assume the config's ``n_fft``,
        ``hop`` and ``clip_dc``, and centred frames.
    reference
        A second 1D signal, same length as ``data``, for a cross-power
        spectrum.

    Raises
    ------
    TypeError
        ``fs`` is not a number (a bool, string or bytes value included).
    ValueError
        On complex, non-numeric, empty or non-finite input, a bad shape
        (including a 2D input with a single row or column, which is a 1D
        signal stored as a vector), a signal shorter than
        ``n_fft // 2 + 1`` samples, a bad ``fs``, or a mismatched
        ``reference``.
    """
    cfg = SpectrogramConfig.coerce(config)
    fs = _check_fs(fs)
    values64, kind, n_samples = _values64(data, cfg, reference)
    if kind == "spectrogram" and 1 in values64.shape:
        raise ValueError(
            f"data has shape {values64.shape}; for a 1D signal pass np.ravel(data) "
            "(a spectrogram needs at least 2 rows and 2 columns)"
        )
    # Always C-ordered (a plain astype keeps an F-ordered input's layout).
    values = np.ascontiguousarray(values64, dtype=np.float32)
    freqs, times = _axes(values.shape, cfg, fs)
    return Spectrogram(values, freqs, times, kind, cfg, fs, n_samples)


def _values64(
    data: Any, cfg: SpectrogramConfig, reference: Any = None
) -> tuple[np.ndarray, str, int | None]:
    """:func:`prepare`'s validation and values, in float64.

    Returns ``(values, kind, n_samples)``. It makes no copy of its own, so
    a 2D float64 input with ``cfg.log`` off comes back as the caller's own
    array; a caller that hands the values out must copy them.
    """
    arr = _as_real_array(data, "data")
    n_samples: int | None = None

    if reference is not None:
        ref = _as_real_array(reference, "reference")
        if arr.ndim != 1 or ref.ndim != 1:
            raise ValueError(
                "reference= needs two 1D signals; got shapes "
                f"{arr.shape} and {ref.shape}"
            )
        if arr.shape != ref.shape:
            raise ValueError(
                "signal and reference must have the same length; got "
                f"{arr.shape[0]} and {ref.shape[0]}"
            )
        values = compute_stft(np.stack([arr, ref]), **cfg.stft_kwargs())
        kind, n_samples = "cross", arr.shape[0]
    elif arr.ndim == 1:
        values = compute_stft(arr, **cfg.stft_kwargs())
        kind, n_samples = "stft", arr.shape[0]
    elif arr.ndim == 2:
        values = log_scale(arr) if cfg.log else arr
        kind = "spectrogram"
    else:
        raise ValueError(
            "expected a 1D signal or a 2D spectrogram, got "
            f"ndim={arr.ndim} (shape {arr.shape})"
        )
    return values, kind, n_samples


def _as_real_array(data: Any, name: str) -> np.ndarray:
    arr = np.asarray(data)
    if np.iscomplexobj(arr):
        raise ValueError(
            f"{name} is complex; pass a real signal (np.abs(x) for a "
            "magnitude spectrogram, or reference= for cross-power)"
        )
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got dtype {arr.dtype}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    arr = arr.astype(np.float64, copy=False)
    n_bad = arr.size - int(np.count_nonzero(np.isfinite(arr)))
    if n_bad:
        raise ValueError(
            f"{name} contains {n_bad} non-finite value(s) (NaN or inf); "
            "remove or interpolate them first"
        )
    return arr


def _check_fs(fs: Any) -> float | None:
    if fs is None:
        return None
    # bool (b), str (U) and bytes (S) convert with float(), but are not rates.
    if np.asarray(fs).dtype.kind in "bSU":
        raise TypeError(f"fs must be a number, got {fs!r}")
    try:
        value = float(fs)
    except (TypeError, ValueError):
        raise TypeError(f"fs must be a number, got {fs!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"fs must be a finite positive number, got {fs!r}")
    return value


def _axes(
    shape: tuple[int, int], cfg: SpectrogramConfig, fs: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Row and column centres for a spectrogram of ``shape``, for every kind.

    Row ``r`` is FFT bin ``b = r + 1`` with ``clip_dc`` (else ``r``), at
    ``b * fs / n_fft`` Hz; column ``j`` is at ``j * hop / fs`` seconds.
    Without ``fs`` they are ``b`` and ``j``.
    """
    n_rows, n_cols = shape
    offset = 1 if cfg.clip_dc else 0
    bins = np.arange(n_rows, dtype=np.float64) + offset
    frames = np.arange(n_cols, dtype=np.float64)
    if fs is None:
        return bins, frames
    return bins * fs / cfg.n_fft, frames * cfg.hop / fs
