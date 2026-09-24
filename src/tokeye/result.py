"""``Segmentation``: a TokEye mask together with the spectrogram it came from."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import export
from ._version import __version__
from .config import DEFAULT_CHANNELS, SpectrogramConfig, resolve_channels
from .preprocess import KINDS, Spectrogram, _axes

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
        Registry name or path of the model that produced the mask.
    """

    mask: np.ndarray
    spectrogram: Spectrogram
    channels: tuple[str, ...] = DEFAULT_CHANNELS
    model: str = "big_tf_unet"

    def __post_init__(self) -> None:
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

    def __getitem__(self, name: str) -> np.ndarray:
        """The ``(H, W)`` scores of channel ``name``."""
        try:
            return self.mask[self.channels.index(name)]
        except ValueError:
            raise KeyError(
                f"no channel {name!r}; channels: {list(self.channels)}"
            ) from None

    @property
    def coherent(self) -> np.ndarray:
        """``(H, W)`` coherent-activity scores."""
        return self["coherent"]

    @property
    def transient(self) -> np.ndarray:
        """``(H, W)`` transient-activity scores."""
        return self["transient"]

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

        Returns the path written (``.npz`` is appended when missing).
        """
        spec = self.spectrogram
        cfg = spec.config
        stft_meta = None
        if spec.fs is not None:
            stft_meta = {
                "fs": spec.fs,
                "t0_ms": float(spec.times[0]) * 1e3,
                "n_fft": cfg.n_fft,
                "hop": cfg.hop,
                "clip_dc": cfg.clip_dc,
            }
        params = {
            "config": cfg.to_dict(),
            "model": self.model,
            "tokeye_version": __version__,
            "kind": spec.kind,
            "fs": spec.fs,
            "channels": list(self.channels),
        }
        bundle = export.analysis_bundle(
            spectrogram=spec.values,
            mask=self.mask,
            stft_meta=stft_meta,
            params=params,
            source="segment",
        )
        return export.save_npz(path, bundle)

    @classmethod
    def load(cls, path: str | Path) -> Segmentation:
        """Load a ``tokeye-analysis/v1`` bundle that contains a mask.

        Bundles written by the app (flat ``n_fft``/``hop``/... params, no
        axes) load too; missing settings fall back to the defaults.
        """
        with np.load(Path(path), allow_pickle=False) as data:
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
            params = (
                json.loads(str(data["params_json"])) if "params_json" in files else {}
            )
            time_ms = data["time_ms"] if "time_ms" in files else None
            freq_khz = data["freq_khz"] if "freq_khz" in files else None

        if mask.ndim == 2:
            mask = mask[np.newaxis]
        cfg = _config_from_params(params)
        kind = params.get("kind") if params.get("kind") in KINDS else "spectrogram"
        fs = params.get("fs")
        fs = float(fs) if isinstance(fs, (int, float)) and fs > 0 else None
        if fs is not None and time_ms is not None and freq_khz is not None:
            freqs = np.asarray(freq_khz, dtype=np.float64) * 1e3
            times = np.asarray(time_ms, dtype=np.float64) / 1e3
        else:
            freqs, times = _axes(values.shape, kind, cfg, fs, None)
        spectrogram = Spectrogram(values, freqs, times, kind, cfg, fs)

        channels = params.get("channels")
        if not (isinstance(channels, list) and len(channels) == mask.shape[0]):
            channels = resolve_channels(DEFAULT_CHANNELS, mask.shape[0])
        model = params.get("model") or "unknown"
        return cls(mask, spectrogram, tuple(channels), str(model))

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


def _config_from_params(params: dict[str, Any]) -> SpectrogramConfig:
    raw = params.get("config")
    if isinstance(raw, dict):
        try:
            return SpectrogramConfig.from_dict(raw)
        except (TypeError, ValueError):
            pass
    flat = {key: params[key] for key in _FLAT_CONFIG_KEYS if key in params}
    try:
        return SpectrogramConfig.from_dict(flat)
    except (TypeError, ValueError):
        return SpectrogramConfig()
