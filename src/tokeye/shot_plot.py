"""PNG for ``tokeye <shot> <signal>``: spectrogram + TokEye mask on real axes.

Two stacked panels sharing the time/frequency axes:

* top    — the log spectrogram (the exact model input), colour-mapped;
* bottom — the same spectrogram in grey with the thresholded mask on top
  (green = coherent modes, red = transient events).

Wide shots are reduced for display only (block-mean for the spectrogram,
block-max for the mask so thin modes survive); the saved NPZ keeps full
resolution. Headless (Agg) — safe on compute nodes without a display.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
    from typing import BinaryIO

DISPLAY_MAX_COLS = 4000

_COHERENT_RGBA = (0.0, 1.0, 0.25, 0.55)
_TRANSIENT_RGBA = (1.0, 0.15, 0.15, 0.55)


def _block_reduce(arr: np.ndarray, factor: int, how: str) -> np.ndarray:
    if factor <= 1:
        return arr
    w = (arr.shape[-1] // factor) * factor
    a = arr[..., :w].reshape(*arr.shape[:-1], w // factor, factor)
    return a.max(axis=-1) if how == "max" else a.mean(axis=-1)


def _band_rows(f_khz: np.ndarray, fmin: float | None, fmax: float | None) -> slice:
    lo, hi = 0, f_khz.size
    if fmin is not None and fmin > 0:
        lo = int(np.searchsorted(f_khz, fmin, side="left"))
    if fmax is not None and fmax > 0:
        hi = int(np.searchsorted(f_khz, fmax, side="right"))
    if hi - lo < 2:
        return slice(0, f_khz.size)
    return slice(lo, hi)


def save_shot_png(
    path: Path | BinaryIO,
    spec: np.ndarray,
    mask: np.ndarray,
    t_ms: np.ndarray,
    f_khz: np.ndarray,
    *,
    title: str = "",
    subtitle: str = "",
    threshold: float = 0.5,
    fmin_khz: float | None = None,
    fmax_khz: float | None = None,
    cmap: str = "viridis",
    dpi: int = 150,
) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = _band_rows(np.asarray(f_khz), fmin_khz, fmax_khz)
    spec = np.asarray(spec)[rows]
    mask = np.asarray(mask)[:, rows]
    f = np.asarray(f_khz)[rows]
    t = np.asarray(t_ms)

    factor = int(np.ceil(spec.shape[1] / DISPLAY_MAX_COLS)) if spec.shape[1] > DISPLAY_MAX_COLS else 1
    spec_d = _block_reduce(spec, factor, "mean")
    on = mask >= threshold
    coh = _block_reduce(on[0], factor, "max")
    tra = _block_reduce(on[1], factor, "max")

    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    df = float(np.median(np.diff(f))) if f.size > 1 else 1.0
    t_end = t[0] + spec_d.shape[1] * factor * dt
    extent = (t[0] - dt / 2, t_end - dt / 2, f[0] - df / 2, f[-1] + df / 2)

    overlay = np.zeros((*spec_d.shape, 4), dtype=np.float32)
    overlay[coh] = _COHERENT_RGBA
    overlay[tra] = _TRANSIENT_RGBA

    vmin, vmax = np.percentile(spec_d, [1, 99.5])
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(14, 8.5), sharex=True, sharey=True, constrained_layout=True
    )
    im = ax0.imshow(
        spec_d, origin="lower", aspect="auto", extent=extent, cmap=cmap,
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    fig.colorbar(im, ax=ax0, pad=0.01, label="log(1+|STFT|)")
    ax1.imshow(
        spec_d, origin="lower", aspect="auto", extent=extent, cmap="gray",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax1.imshow(overlay, origin="lower", aspect="auto", extent=extent, interpolation="nearest")
    # keep the two panels the same width as the colorbar'd top one
    fig.colorbar(mpl.cm.ScalarMappable(cmap="gray"), ax=ax1, pad=0.01).ax.set_visible(False)
    ax1.legend(
        handles=[
            Patch(color=_COHERENT_RGBA[:3], label=f"coherent (p ≥ {threshold:g})"),
            Patch(color=_TRANSIENT_RGBA[:3], label=f"transient (p ≥ {threshold:g})"),
        ],
        loc="upper right", fontsize=9, framealpha=0.85,
    )
    ax0.set_ylabel("frequency (kHz)")
    ax1.set_ylabel("frequency (kHz)")
    ax1.set_xlabel("time (ms)")
    ax0.set_title("spectrogram", fontsize=10, loc="left")
    ax1.set_title("TokEye mask", fontsize=10, loc="left")
    fig.suptitle(f"{title}\n{subtitle}" if subtitle else title, fontsize=11)
    fig.savefig(path, dpi=dpi, format="png")
    plt.close(fig)
