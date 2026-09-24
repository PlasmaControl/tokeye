"""Matplotlib rendering shared by ``Segmentation.plot`` and batch previews.

Never imports ``matplotlib.pyplot``: figures are built with
:class:`matplotlib.figure.Figure`, which needs no GUI backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import DEFAULT_CHANNELS, resolve_channels

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .result import Segmentation

CHANNEL_COLORS: dict[str, tuple[float, float, float]] = {
    "coherent": (0.0, 1.0, 0.0),
    "transient": (1.0, 0.0, 0.0),
    "background": (0.0, 0.4, 1.0),
}
_FALLBACK_COLORS = ((1.0, 0.6, 0.0), (0.6, 0.0, 1.0), (0.0, 0.8, 0.8), (1.0, 0.0, 0.6))


def channel_color(name: str, index: int) -> tuple[float, float, float]:
    """RGB for a channel: its named colour, else a fallback by index."""
    return CHANNEL_COLORS.get(name, _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)])


def overlay_rgba(
    mask: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 0.4,
    channels: tuple[str, ...] | None = None,
) -> np.ndarray:
    """``(H, W, 4)`` RGBA overlay of a ``(C, H, W)`` mask thresholded at
    ``threshold``; later channels paint over earlier ones."""
    mask = np.asarray(mask)
    names = channels or resolve_channels(DEFAULT_CHANNELS, mask.shape[0])
    rgba = np.zeros((*mask.shape[1:], 4), dtype=np.float32)
    for index, name in enumerate(names):
        rgba[mask[index] >= threshold] = (*channel_color(name, index), alpha)
    return rgba


def _extent(times: np.ndarray, freqs: np.ndarray) -> tuple[float, float, float, float]:
    """imshow extent whose pixel centres sit on the given axis values."""

    def edges(centres: np.ndarray) -> tuple[float, float]:
        n = len(centres)
        step = (centres[-1] - centres[0]) / (n - 1) if n > 1 else 1.0
        return float(centres[0] - step / 2), float(centres[-1] + step / 2)

    x0, x1 = edges(times)
    y0, y1 = edges(freqs)
    return x0, x1, y0, y1


def draw_segmentation(
    ax: Axes, seg: Segmentation, *, threshold: float = 0.5, alpha: float = 0.4
) -> Axes:
    """Draw the spectrogram (grey, ``origin="lower"``) with the mask overlay."""
    from matplotlib.patches import Patch

    spec = seg.spectrogram
    extent = _extent(spec.times, spec.freqs)
    ax.imshow(spec.values, cmap="gray", origin="lower", aspect="auto", extent=extent)
    ax.imshow(
        overlay_rgba(seg.mask, threshold, alpha, seg.channels),
        origin="lower",
        aspect="auto",
        extent=extent,
    )
    if spec.fs is not None:
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
    else:
        ax.set_xlabel("Frame")
        ax.set_ylabel("Frequency bin")
    handles = [
        Patch(color=channel_color(name, i), label=name)
        for i, name in enumerate(seg.channels)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize="small")
    return ax


def save_preview(
    seg: Segmentation, path: str | Path, threshold: float = 0.5, dpi: int = 150
) -> Path:
    """Write a PNG preview of ``seg`` without touching ``pyplot``."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 4), layout="constrained")
    draw_segmentation(fig.add_subplot(), seg, threshold=threshold)
    fig.savefig(path, dpi=dpi)
    return Path(path)
