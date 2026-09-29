"""Preprocessing configuration: the single source of spectrogram defaults.

Standard library only, so the CLI can build its flags from
:class:`SpectrogramConfig` without importing numpy, scipy or torch.
"""

from __future__ import annotations

import dataclasses
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = "big_tf_unet"
DEFAULT_CHANNELS: tuple[str, ...] = ("coherent", "transient")


def resolve_channels(preferred: tuple[str, ...], n: int) -> tuple[str, ...]:
    """Channel names for an ``n``-channel mask.

    Returns ``preferred`` when it has exactly ``n`` names, otherwise the
    generic ``("channel_0", "channel_1", ...)``.
    """
    if len(preferred) == n:
        return tuple(preferred)
    return tuple(f"channel_{i}" for i in range(n))


_INT_FIELDS = ("n_fft", "hop")
_FLOAT_FIELDS = ("clip_low", "clip_high")
_BOOL_FIELDS = ("clip_dc", "log")


@dataclass(frozen=True)
class SpectrogramConfig:
    """How a raw signal (or a stored spectrogram) becomes model input.

    Parameters
    ----------
    n_fft
        STFT window length in samples.
    hop
        STFT hop in samples. 128 matches the released model's training.
    window
        Window name understood by :func:`scipy.signal.get_window`.
    clip_dc
        Drop the DC (0 Hz) row of the STFT.
    clip_low, clip_high
        Percentiles the log-magnitude spectrogram is clipped to.
    log
        Apply ``log1p`` to 2D inputs stored in linear scale. 1D inputs
        are always log-scaled by the STFT, so this only affects 2D input.
    """

    n_fft: int = field(default=1024, metadata={"help": "STFT window length [samples]"})
    hop: int = field(default=128, metadata={"help": "STFT hop [samples]"})
    window: str = field(default="hann", metadata={"help": "STFT window name"})
    clip_dc: bool = field(default=True, metadata={"help": "drop the DC row"})
    clip_low: float = field(default=1.0, metadata={"help": "lower clip percentile"})
    clip_high: float = field(default=99.0, metadata={"help": "upper clip percentile"})
    log: bool = field(
        default=False,
        metadata={"help": "log1p 2D inputs stored in linear scale"},
    )

    def __post_init__(self) -> None:
        for name in _INT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{name} must be an integer, got {value!r}")
            object.__setattr__(self, name, int(value))
        for name in _FLOAT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be a number, got {value!r}")
            object.__setattr__(self, name, float(value))
        for name in _BOOL_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be True or False, got {value!r}")
        if not isinstance(self.window, str) or not self.window:
            raise TypeError(f"window must be a non-empty string, got {self.window!r}")

        if self.n_fft < 2:
            raise ValueError(f"n_fft must be >= 2, got {self.n_fft}")
        if self.hop < 1:
            raise ValueError(f"hop must be >= 1, got {self.hop}")
        if not 0.0 <= self.clip_low < self.clip_high <= 100.0:
            raise ValueError(
                "need 0 <= clip_low < clip_high <= 100, got "
                f"clip_low={self.clip_low}, clip_high={self.clip_high}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Plain-``dict`` form (JSON-serializable)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SpectrogramConfig:
        """Build from a mapping; unknown keys raise ``ValueError``."""
        valid = [f.name for f in dataclasses.fields(cls)]
        unknown = sorted(set(data) - set(valid))
        if unknown:
            raise ValueError(
                f"unknown SpectrogramConfig field(s) {unknown}; valid: {valid}"
            )
        return cls(**dict(data))

    def replace(self, **changes: Any) -> SpectrogramConfig:
        """Copy with ``changes`` applied (validated like the constructor)."""
        return self.from_dict({**self.to_dict(), **changes})

    @classmethod
    def coerce(
        cls, value: SpectrogramConfig | Mapping[str, Any] | None
    ) -> SpectrogramConfig:
        """Accept ``None`` (defaults), a config, or a mapping of fields."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        raise TypeError(
            f"expected SpectrogramConfig, a mapping or None, got {type(value).__name__}"
        )

    def stft_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :func:`tokeye.transforms.compute_stft`."""
        return {
            "n_fft": self.n_fft,
            "hop": self.hop,
            "window": self.window,
            "clip_dc": self.clip_dc,
            "clip_low": self.clip_low,
            "clip_high": self.clip_high,
        }


DEFAULT_CONFIG = SpectrogramConfig()
