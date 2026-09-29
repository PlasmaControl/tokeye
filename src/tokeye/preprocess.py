"""Turn user data into the spectrogram the model sees.

:func:`prepare` is the one preprocessing entry point shared by the Python
API, the CLI, batch runs and the app:

- a 1D signal becomes a log-magnitude STFT (``kind="stft"``);
- a 1D signal plus ``reference=`` becomes a cross-power STFT
  (``kind="cross"``);
- a 2D array is taken as a ready spectrogram (``kind="spectrogram"``),
  ``log1p``-scaled first when ``config.log`` is on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal as sps

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
        ``(H,)`` row centres: Hz when ``fs`` is known, else bin index.
    times
        ``(W,)`` column centres: seconds when ``fs`` is known, else frame
        index.
    kind
        ``"stft"``, ``"cross"`` or ``"spectrogram"``.
    config
        The :class:`~tokeye.config.SpectrogramConfig` used.
    fs
        Sampling rate in Hz, or ``None`` when unknown.
    """

    values: np.ndarray
    freqs: np.ndarray
    times: np.ndarray
    kind: str
    config: SpectrogramConfig
    fs: float | None

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape


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
    config
        Preprocessing settings; ``None`` uses the defaults.
    fs
        Sampling rate in Hz. Only labels the axes; it never changes values.
        For 2D input the axes assume the config's ``n_fft``/``hop``/``clip_dc``.
    reference
        A second 1D signal, same length as ``data``, for a cross-power
        spectrum.

    Raises
    ------
    ValueError
        On complex, non-numeric, empty or non-finite input, a bad shape,
        a bad ``fs``, or a mismatched ``reference``.
    """
    cfg = SpectrogramConfig.coerce(config)
    fs = _check_fs(fs)
    values64, kind, n_samples = _values64(data, cfg, reference)
    # Always C-ordered (a plain astype keeps an F-ordered input's layout).
    values = np.ascontiguousarray(values64, dtype=np.float32)
    freqs, times = _axes(values.shape, kind, cfg, fs, n_samples)
    return Spectrogram(values, freqs, times, kind, cfg, fs)


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
    try:
        value = float(fs)
    except (TypeError, ValueError):
        raise TypeError(f"fs must be a number, got {fs!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"fs must be a finite positive number, got {fs!r}")
    return value


def _axes(
    shape: tuple[int, int],
    kind: str,
    cfg: SpectrogramConfig,
    fs: float | None,
    n_samples: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Row and column centres for a spectrogram of ``shape``."""
    n_rows, n_cols = shape
    offset = 1 if cfg.clip_dc else 0
    rows = np.arange(n_rows, dtype=np.float64)
    cols = np.arange(n_cols, dtype=np.float64)

    if fs is None:
        freqs = rows if kind == "spectrogram" else rows + offset
        return freqs, cols

    freqs = (rows + offset) * fs / cfg.n_fft
    if kind == "spectrogram" or n_samples is None:
        return freqs, cols * cfg.hop / fs
    win = sps.get_window(cfg.window, cfg.n_fft)
    times = sps.ShortTimeFFT(win, cfg.hop, fs).t(n_samples)
    return freqs, np.asarray(times, dtype=np.float64)
