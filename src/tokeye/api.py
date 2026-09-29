"""One-import Python API: ``from tokeye import TokEye``.

Gradio-free, so it is safe to embed in headless programs. ``TokEye()``
loads the default model (auto-downloading it from Hugging Face on first
use)::

    from tokeye import TokEye

    eye = TokEye()
    seg = eye.segment(signal, fs=500e3)  # Segmentation: mask + spectrogram + axes
    mask = eye(signal)                   # just the (2, H, W) mask
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import hub
from .config import SpectrogramConfig, resolve_channels
from .inference import check_tile, infer
from .preprocess import Spectrogram, _check_fs, prepare
from .result import Segmentation

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import numpy as np


class TokEye:
    """A loaded TokEye segmentation model plus its preprocessing settings.

    Parameters
    ----------
    model
        Registry name (downloaded and cached on first use) or a path to a
        local ``.pt``/``.pt2`` checkpoint. Instance models such as
        ``ae_tf_maskrcnn`` are rejected; use ``tokeye alfvenspec``.
    device
        ``"auto"`` (CUDA, then MPS, then CPU) or a torch device string.
    n_fft, hop, clip_dc, clip_low, clip_high, log
        Override single fields of ``config`` (see
        :class:`~tokeye.config.SpectrogramConfig`). ``None`` keeps the
        config's value.
    config
        Preprocessing settings (a config or a mapping); defaults when
        omitted.
    fs
        Default sampling rate in Hz for :meth:`segment` (labels axes only).
    tile
        Tiling policy passed to :func:`tokeye.inference.infer`.
    """

    def __init__(
        self,
        model: str | Path = hub.DEFAULT_MODEL,
        device: str = "auto",
        n_fft: int | None = None,
        hop: int | None = None,
        clip_dc: bool | None = None,
        clip_low: float | None = None,
        clip_high: float | None = None,
        log: bool | None = None,
        *,
        config: SpectrogramConfig | Mapping[str, Any] | None = None,
        fs: float | None = None,
        tile: int | str | None = "auto",
    ) -> None:
        overrides = {
            "n_fft": n_fft,
            "hop": hop,
            "clip_dc": clip_dc,
            "clip_low": clip_low,
            "clip_high": clip_high,
            "log": log,
        }
        cfg = SpectrogramConfig.coerce(config)
        explicit = {key: value for key, value in overrides.items() if value is not None}
        self.config = cfg.replace(**explicit) if explicit else cfg
        self.fs = _check_fs(fs)
        check_tile(tile)
        self.tile = tile

        hub.require_task(model, "segmentation")
        self.model_name = str(model)
        self.channels = hub.channels_for(model)
        self.model = hub.load_model(model, device)

    @property
    def log(self) -> bool:
        """Whether 2D inputs are ``log1p``-scaled (``config.log``)."""
        return self.config.log

    @log.setter
    def log(self, value: bool) -> None:
        self.config = self.config.replace(log=value)

    def _prepare(
        self,
        data: Any,
        *,
        fs: float | None = None,
        reference: Any = None,
        log: bool | None = None,
    ) -> Spectrogram:
        cfg = self.config if log is None else self.config.replace(log=log)
        return prepare(data, cfg, fs=self.fs if fs is None else fs, reference=reference)

    def spectrogram(self, data: Any, log: bool | None = None) -> np.ndarray:
        """The ``(H, W)`` float32 spectrogram the model will see.

        1D input goes through the STFT (which log-scales internally); 2D
        input is used as-is, ``log1p``-scaled first when ``log`` is on.
        ``log=None`` defers to the instance setting.
        """
        return self._prepare(data, log=log).values

    def segment(
        self,
        data: Any,
        *,
        fs: float | None = None,
        reference: Any = None,
        log: bool | None = None,
    ) -> Segmentation:
        """Segment a signal or spectrogram.

        Parameters
        ----------
        data
            1D signal or 2D spectrogram.
        fs
            Sampling rate in Hz (overrides the instance ``fs``).
        reference
            Second 1D signal for a cross-power spectrum.
        log
            Per-call override of ``config.log``.

        Returns
        -------
        Segmentation
            Mask, the spectrogram it was computed from, and its axes.
        """
        spec = self._prepare(data, fs=fs, reference=reference, log=log)
        mask = infer(self.model, spec.values, tile=self.tile)
        channels = resolve_channels(self.channels, mask.shape[0])
        return Segmentation(mask, spec, channels, self.model_name)

    def predict(self, data: Any, log: bool | None = None) -> np.ndarray:
        """Run inference; returns a float32 mask of shape ``(C, H, W)``.

        For ``big_tf_unet``, channel 0 = coherent and channel 1 =
        transient activity, both sigmoid scores in ``[0, 1]``.
        Standardization is applied internally.
        """
        return self.segment(data, log=log).mask

    def __call__(self, data: Any, log: bool | None = None) -> np.ndarray:
        return self.predict(data, log=log)
