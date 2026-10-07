"""``Segmentation``: a TokEye mask together with the spectrogram it came from."""

from __future__ import annotations

import dataclasses
import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import export
from ._version import __version__
from .config import (
    _BOOL_FIELDS,
    _FLOAT_FIELDS,
    _INT_FIELDS,
    DEFAULT_CHANNELS,
    DEFAULT_CONFIG,
    SpectrogramConfig,
    _user_stacklevel,
    resolve_channels,
)
from .preprocess import KINDS, Spectrogram, _axes, _axes_text

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_FLAT_CONFIG_KEYS = ("n_fft", "hop", "clip_dc", "clip_low", "clip_high")


@dataclass(frozen=True, eq=False)
class Segmentation:
    """The result of :meth:`tokeye.TokEye.segment`.

    Attributes
    ----------
    mask
        ``(C, H, W)`` float32 scores in ``[0, 1]``; for ``big_tf_unet``,
        channel 0 is coherent and channel 1 transient activity.
    spectrogram
        The :class:`~tokeye.preprocess.Spectrogram` the model saw.
    channels
        One name per mask channel.
    model
        Registry name or file name of the model that produced the mask.
    weights
        Which weights produced it (see :func:`tokeye.hub.weights_info`):
        ``repo``, ``filename``, ``revision`` and ``sha256`` for a registry
        model, ``name`` and ``sha256`` for a local checkpoint; ``None`` when
        unknown. A mapping is stored as a plain ``dict``.
    """

    mask: np.ndarray
    spectrogram: Spectrogram
    channels: tuple[str, ...] = DEFAULT_CHANNELS
    model: str = "big_tf_unet"
    weights: Mapping[str, str | None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spectrogram, Spectrogram):
            raise TypeError(
                "spectrogram must be a tokeye Spectrogram, got "
                f"{type(self.spectrogram).__name__}"
            )
        if self.weights is not None:
            if not isinstance(self.weights, Mapping):
                raise TypeError(
                    "weights must be a mapping or None, got "
                    f"{type(self.weights).__name__}"
                )
            object.__setattr__(self, "weights", dict(self.weights))
        mask = np.asarray(self.mask, dtype=np.float32)
        if mask.ndim != 3 or mask.shape[1:] != self.spectrogram.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match spectrogram shape "
                f"{self.spectrogram.shape}; expected (C, H, W)"
            )
        channels = tuple(self.channels)
        if len(channels) != mask.shape[0]:
            raise ValueError(
                f"{len(channels)} channel name(s) for a {mask.shape[0]}-channel mask"
            )
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "channels", channels)

    def __repr__(self) -> str:
        return (
            f"Segmentation(model={self.model!r}, mask={self.mask.shape}, "
            f"channels={self.channels!r}, "
            f"{_axes_text(self.fs, self.freqs, self.times)})"
        )

    def __getitem__(self, name: str) -> np.ndarray:
        """The ``(H, W)`` scores of channel ``name``."""
        try:
            return self.mask[self.channels.index(name)]
        except ValueError:
            raise KeyError(
                f"no channel {name!r}; channels: {list(self.channels)}"
            ) from None

    def _channel_attribute(self, name: str) -> np.ndarray:
        # AttributeError, so hasattr()/getattr(..., default) work when the
        # channel is absent; seg[name] keeps raising KeyError.
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(exc.args[0]) from None

    @property
    def coherent(self) -> np.ndarray:
        """``(H, W)`` coherent-activity scores (``AttributeError`` if absent)."""
        return self._channel_attribute("coherent")

    @property
    def transient(self) -> np.ndarray:
        """``(H, W)`` transient-activity scores (``AttributeError`` if absent)."""
        return self._channel_attribute("transient")

    @property
    def freqs(self) -> np.ndarray:
        """Row centres: Hz when ``fs`` is known, else bin index."""
        return self.spectrogram.freqs

    @property
    def times(self) -> np.ndarray:
        """Column centres: seconds when ``fs`` is known, else frame index."""
        return self.spectrogram.times

    @property
    def fs(self) -> float | None:
        """Sampling rate in Hz, or ``None``."""
        return self.spectrogram.fs

    def threshold(self, t: float = 0.5) -> np.ndarray:
        """Boolean ``(C, H, W)`` mask, ``True`` where the score is ``>= t``."""
        return self.mask >= t

    def save(self, path: str | Path) -> Path:
        """Save as a ``tokeye-analysis/v1`` ``.npz`` (the app's schema).

        Returns the path written (``.npz`` is appended when missing). With
        ``fs`` known, the result's own axes are stored (in ms and kHz).
        Without it the axes are bin and frame indices and are not stored;
        other axes then warn (``UserWarning``), since they reload as indices.
        """
        spec = self.spectrogram
        axes = None
        if spec.fs is not None:
            axes = (
                np.asarray(spec.times, dtype=np.float64) * 1e3,
                np.asarray(spec.freqs, dtype=np.float64) / 1e3,
            )
        params = {
            "config": spec.config.to_dict(),
            "model": self.model,
            "weights": self.weights,
            "tokeye_version": __version__,
            "kind": spec.kind,
            "fs": spec.fs,
            "n_samples": None if spec.n_samples is None else int(spec.n_samples),
            "channels": list(self.channels),
        }
        bundle = export.analysis_bundle(
            spectrogram=spec.values,
            mask=self.mask,
            axes=axes,
            params=params,
            source="segment",
        )
        written = export.save_npz(path, bundle)
        if spec.fs is None:
            freqs, times = _axes(spec.shape, spec.config, None)
            if not (
                np.array_equal(spec.freqs, freqs) and np.array_equal(spec.times, times)
            ):
                warnings.warn(
                    f"{written}: axes without fs are not saved; they reload as "
                    "bin/frame indices",
                    UserWarning,
                    stacklevel=_user_stacklevel(),
                )
        return written

    @classmethod
    def load(cls, path: str | Path) -> Segmentation:
        """Load a ``tokeye-analysis/v1`` bundle that contains a mask.

        Bundles written by the app (flat ``n_fft``/``hop``/... params, no
        axes) load too; missing settings fall back to the defaults, silently.
        A recorded value that is invalid (an unknown or out-of-range setting,
        a bad ``fs``, ``kind``, ``n_samples``, channel list, weights record or
        stored axis) is not used: it falls back to its default with a
        ``UserWarning`` naming the file. Bundles without a weights record
        load with ``weights=None``, and without ``n_samples`` with
        ``n_samples=None``.

        Bundles do not record their STFT framing, so their stored axes
        (``time_ms``/``freq_khz``) are the only record of it: with a valid
        ``fs``, each valid stored axis is used as stored, and only a missing
        or invalid one is rebuilt from ``fs`` and the settings.

        Raises
        ------
        ValueError
            The file is not a ``tokeye-analysis/v1`` bundle (a bare ``.npy``
            mask, say: ``tokeye run --format npz`` and :meth:`save` write
            bundles), or has no mask.
        """
        loaded = np.load(Path(path), allow_pickle=False)
        if isinstance(loaded, np.ndarray):
            raise ValueError(
                f"{path} is a bare array (such as the mask `tokeye run` writes by "
                f"default), not a {export.SCHEMA_ANALYSIS} bundle; "
                "`tokeye run --format npz` or Segmentation.save() writes a bundle "
                "that loads"
            )
        with loaded as data:
            files = set(data.files)
            schema = str(data["schema"]) if "schema" in files else None
            if schema != export.SCHEMA_ANALYSIS:
                raise ValueError(
                    f"{path} is not a {export.SCHEMA_ANALYSIS} bundle "
                    f"(schema={schema!r})"
                )
            if "mask" not in files:
                raise ValueError(f"{path} has no mask (saved before inference?)")
            values = np.asarray(data["spectrogram"], dtype=np.float32)
            mask = np.asarray(data["mask"], dtype=np.float32)
            params_json = str(data["params_json"]) if "params_json" in files else "{}"
            time_ms = data["time_ms"] if "time_ms" in files else None
            freq_khz = data["freq_khz"] if "freq_khz" in files else None

        if mask.ndim == 2:
            mask = mask[np.newaxis]
        params = _params_from_json(path, params_json)
        cfg = _config_from_params(params, path)

        kind = params.get("kind")
        if kind is None:
            kind = "spectrogram"
        elif kind not in KINDS:
            _warn_ignored(path, "kind", kind, f"not one of {KINDS}", "'spectrogram'")
            kind = "spectrogram"

        fs = params.get("fs")
        if fs is not None and not _positive_finite(fs):
            _warn_ignored(
                path,
                "fs",
                fs,
                "not a positive finite number",
                "no fs: the axes are rebuilt as bin/frame indices",
            )
            fs = None
        fs = None if fs is None else float(fs)

        freqs, times = _axes(values.shape, cfg, fs)
        if fs is not None:
            stored_times = _stored_axis(path, "time_ms", time_ms, values.shape[1])
            if stored_times is not None:
                times = stored_times / 1e3
            stored_freqs = _stored_axis(path, "freq_khz", freq_khz, values.shape[0])
            if stored_freqs is not None:
                freqs = stored_freqs * 1e3
        n_samples = _n_samples_from(path, params.get("n_samples"))
        spectrogram = Spectrogram(values, freqs, times, kind, cfg, fs, n_samples)

        default_channels = resolve_channels(DEFAULT_CHANNELS, mask.shape[0])
        channels = params.get("channels")
        if channels is None:
            channels = default_channels
        elif not (
            isinstance(channels, list)
            and len(channels) == mask.shape[0]
            and all(isinstance(name, str) for name in channels)
        ):
            _warn_ignored(
                path,
                "channels",
                channels,
                f"not a list of {mask.shape[0]} names",
                f"{list(default_channels)}",
            )
            channels = default_channels
        model = params.get("model") or "unknown"
        weights = params.get("weights")
        if weights is not None and not _is_weights_record(weights):
            _warn_ignored(
                path,
                "weights",
                weights,
                "not a mapping of strings to strings or null",
                "None",
            )
            weights = None
        return cls(mask, spectrogram, tuple(channels), str(model), weights)

    def plot(
        self, ax: Axes | None = None, *, threshold: float = 0.5, alpha: float = 0.4
    ) -> Axes:
        """Draw the spectrogram with the thresholded mask overlaid.

        Uses ``origin="lower"`` (low frequencies at the bottom) and labelled
        axes. Creates a new pyplot figure when ``ax`` is ``None``.
        """
        from ._plotting import draw_segmentation

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        return draw_segmentation(ax, self, threshold=threshold, alpha=alpha)


def _warn_ignored(
    path: str | Path, name: str, value: Any, reason: str, fallback: str
) -> None:
    """Warn that a recorded value of the bundle at ``path`` is not used."""
    shown = (
        f"array of shape {value.shape}"
        if isinstance(value, np.ndarray)
        else repr(value)
    )
    warnings.warn(
        f"{path}: ignoring {name}={shown} ({reason}); using {fallback}",
        UserWarning,
        stacklevel=_user_stacklevel(),
    )


def _is_weights_record(value: Any) -> bool:
    """Whether ``value`` is a dict of strings to strings or ``None``."""
    return isinstance(value, dict) and all(
        isinstance(key, str) and (item is None or isinstance(item, str))
        for key, item in value.items()
    )


def _warn(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=_user_stacklevel())


def _params_from_json(path: str | Path, text: str) -> dict[str, Any]:
    """The bundle's params: a JSON object, else ``{}`` with a warning."""
    try:
        params = json.loads(text)
    except ValueError as exc:
        _warn(f"{path}: ignoring params_json (not valid JSON: {exc}); using {{}}")
        return {}
    if not isinstance(params, dict):
        _warn_ignored(path, "params_json", params, "not a JSON object", "{}")
        return {}
    return params


def _positive_finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:  # an int too large for a float
        return False


def _n_samples_from(path: str | Path, recorded: Any) -> int | None:
    """A recorded signal length: a positive int (``4096.0`` reads as 4096)."""
    if recorded is None:
        return None
    value = recorded
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    _warn_ignored(path, "n_samples", recorded, "not a positive integer", "None")
    return None


def _stored_axis(
    path: str | Path, name: str, stored: np.ndarray | None, length: int
) -> np.ndarray | None:
    """A valid stored axis as float64; ``None`` (with a warning if invalid)."""
    if stored is None:
        return None
    reason = None
    if stored.shape != (length,):
        reason = f"expected shape ({length},)"
    elif stored.dtype == np.bool_ or not np.issubdtype(stored.dtype, np.number):
        reason = f"not numeric, dtype {stored.dtype}"
    elif np.iscomplexobj(stored) or not np.isfinite(stored).all():
        reason = "not all real and finite"
    if reason is not None:
        _warn_ignored(
            path, name, stored, reason, "axes rebuilt from fs and the settings"
        )
        return None
    return np.asarray(stored, dtype=np.float64)


def _from_file(name: str, value: Any) -> Any:
    """A recorded setting as the type the config takes, where that is lossless.

    JSON writers may emit ``512.0`` for an integer, and a file's ``0``/``1``
    flags become bools without the deprecation warning meant for code.
    """
    if name in _INT_FIELDS and isinstance(value, float) and value.is_integer():
        return int(value)
    if (
        name in _BOOL_FIELDS
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value in (0, 1)
    ):
        return bool(value)
    return value


def _config_from_params(params: dict[str, Any], path: str | Path) -> SpectrogramConfig:
    """The config a bundle recorded, without its invalid fields.

    The source is a ``config`` mapping, else the flat keys the app writes.
    Unknown and invalid fields warn and fall back to their defaults; the
    clip percentiles are checked (and fall back) as a pair.
    """
    raw = params.get("config")
    if raw is not None and not isinstance(raw, dict):
        _warn(f"{path}: params 'config' is not a mapping; using the flat settings")
        raw = None
    if raw is None:
        source = {key: params[key] for key in _FLAT_CONFIG_KEYS if key in params}
    else:
        valid = {f.name for f in dataclasses.fields(SpectrogramConfig)}
        unknown = sorted((key for key in raw if key not in valid), key=repr)
        if unknown:
            _warn(f"{path}: ignoring unknown config field(s) {unknown}")
        source = {key: value for key, value in raw.items() if key in valid}

    kept: dict[str, Any] = {}
    for name, recorded in source.items():
        if name in _FLOAT_FIELDS:
            continue
        value = _from_file(name, recorded)
        try:
            SpectrogramConfig.from_dict({name: value})
        except (TypeError, ValueError) as exc:
            default = getattr(DEFAULT_CONFIG, name)
            _warn(
                f"{path}: ignoring {name}={recorded!r} ({exc}); "
                f"using the default {default!r}"
            )
            continue
        kept[name] = value

    clips = {name: source[name] for name in _FLOAT_FIELDS if name in source}
    if clips:
        try:
            SpectrogramConfig.from_dict(clips)
        except (TypeError, ValueError) as exc:
            recorded = ", ".join(f"{name}={value!r}" for name, value in clips.items())
            defaults = ", ".join(
                f"{name}={getattr(DEFAULT_CONFIG, name)!r}" for name in _FLOAT_FIELDS
            )
            _warn(f"{path}: ignoring {recorded} ({exc}); using the defaults {defaults}")
        else:
            kept.update(clips)
    return SpectrogramConfig.from_dict(kept)
