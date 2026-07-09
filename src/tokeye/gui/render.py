"""Pure array/geometry helpers for the GUI plots — no Qt, no torch.

Ports the real time-ms / frequency-kHz axis math from
:mod:`tokeye.sources.viz` (the Plotly/matplotlib renderers) so the native GUI
computes identical extents without importing plotly or matplotlib for axes.

pyqtgraph auto-downsamples wide images for display while keeping true data
coordinates, so — unlike the Plotly path — we never block-mean the data here and
the column factor is always 1. That makes :func:`spectrogram_rect` a clean
"pixel-centre → half-pixel-edge" mapping.
"""

from __future__ import annotations

import numpy as np

from tokeye.app.analyze.visualize import amplitude, enhance, mask

__all__ = [
    "freq_crop_rows",
    "spectrogram_rect",
    "auto_levels",
    "view_display_array",
    "is_rgb_view",
    "axis_rect",
    "mode_color_table",
    "discrete_mode_image",
    "nd_masked_for_display",
    "mode_ticks",
]


def freq_crop_rows(h: int, stft_meta: dict | None) -> tuple[int, int]:
    """Row bounds ``[r_lo, r_hi)`` cropping an STFT image of height ``h`` to the band.

    Row ``i`` maps to FFT bin ``i + offset`` (``offset=1`` if the DC bin was
    clipped), i.e. frequency ``(i+offset)·fs/n_fft``. Returns ``(0, h)`` (no crop)
    when the band is unset, ``fs`` is unknown, or the crop is degenerate, so a
    caller can always slice ``arr[r_lo:r_hi]`` safely. Ported verbatim from
    :func:`tokeye.sources.viz._freq_crop_rows`.
    """
    if not stft_meta:
        return 0, h
    fs = float(stft_meta.get("fs", 0.0))
    if fs <= 0:
        return 0, h
    n_fft = int(stft_meta.get("n_fft", 1024))
    offset = 1 if bool(stft_meta.get("clip_dc", True)) else 0
    fmin = stft_meta.get("fmin_khz")
    fmax = stft_meta.get("fmax_khz")
    r_lo, r_hi = 0, h
    if fmin is not None and float(fmin) > 0:
        r_lo = max(0, int(np.ceil(float(fmin) * 1e3 * n_fft / fs)) - offset)
    if fmax is not None and float(fmax) > 0:
        r_hi = min(h, int(np.floor(float(fmax) * 1e3 * n_fft / fs)) - offset + 1)
    if r_lo >= r_hi or (r_hi - r_lo) < 2:
        return 0, h
    return r_lo, r_hi


def spectrogram_rect(
    stft_meta: dict | None,
    n_rows: int,
    n_cols: int,
    r_lo: int = 0,
) -> tuple[float, float, float, float]:
    """``(x_left_ms, y_bottom_kHz, x_span_ms, y_span_kHz)`` for ``ImageItem.setRect``.

    ``n_rows``/``n_cols`` are the *cropped* image dimensions; ``r_lo`` the first
    kept FFT row (from :func:`freq_crop_rows`). The edges sit half a pixel beyond
    the first/last pixel centres, matching :func:`tokeye.sources.viz._view_axes`
    with a column factor of 1. Falls back to pixel indices when ``fs`` is unknown.
    """
    fs = float(stft_meta.get("fs", 0.0)) if stft_meta else 0.0
    if fs <= 0:
        return 0.0, 0.0, float(n_cols), float(n_rows)
    n_fft = int(stft_meta.get("n_fft", 1024))
    hop = int(stft_meta.get("hop", 256))
    t0 = float(stft_meta.get("t0_ms", 0.0))
    offset = 1 if bool(stft_meta.get("clip_dc", True)) else 0
    df = fs / n_fft / 1e3  # kHz per row
    dt = hop / fs * 1e3  # ms per column
    y0 = (r_lo + offset) * df  # centre freq of cropped row 0
    x_left = t0 - dt / 2.0
    y_bottom = y0 - df / 2.0
    return x_left, y_bottom, n_cols * dt, n_rows * df


def auto_levels(arr: np.ndarray, q: float = 0.95) -> tuple[float, float]:
    """(lo, hi) display levels for a scalar image: min → ``q``-quantile.

    Mirrors pyspecview's auto-scale (clip the top few percent so a handful of hot
    pixels don't wash the panel out). Robust to NaNs and flat inputs.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(finite.min())
    hi = float(np.quantile(finite, q))
    if hi <= lo:
        hi = float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def is_rgb_view(view_mode: str) -> bool:
    """Whether a view mode renders as an RGB overlay (vs a scalar+LUT image)."""
    return view_mode in ("Enhanced", "Mask")


def view_display_array(
    view_mode: str,
    arr: np.ndarray | None,
    arr_extract: np.ndarray | None,
    ch0_enabled: bool,
    ch1_enabled: bool,
    vmin: float,
    vmax: float,
    threshold: float,
) -> np.ndarray | None:
    """Scalar ``(H, W)`` or RGB ``(H, W, 3)`` float array for a view mode.

    Same dispatch as ``visualize.show_image`` (reuses ``enhance`` / ``mask`` /
    ``amplitude``), returning arrays (not a figure) for pyqtgraph. ``None`` when
    the needed inputs are missing. RGB channels: green = ch0 (coherent), red =
    ch1 (transient).
    """
    if arr is None:
        return None
    if view_mode == "Original":
        return np.asarray(arr, dtype=float)
    if arr_extract is None:
        return None
    if view_mode == "Enhanced":
        return enhance(arr_extract, ch0_enabled, ch1_enabled, vmin, vmax)
    if view_mode == "Mask":
        return mask(arr_extract, ch0_enabled, ch1_enabled, threshold)
    if view_mode == "Amplitude":
        return amplitude(arr, arr_extract, ch0_enabled, ch1_enabled, threshold)
    return None


# ── toroidal modespec (discrete mode-number map) ──────────────────────────────
def axis_rect(
    x_vec: np.ndarray, y_vec: np.ndarray
) -> tuple[float, float, float, float]:
    """``(x_left, y_bottom, x_span, y_span)`` for an image whose pixel *centres*
    fall on ``x_vec``/``y_vec`` (edges sit half a step beyond the ends)."""

    def edge(v: np.ndarray) -> tuple[float, float]:
        v = np.asarray(v, dtype=float)
        if v.size < 2:
            return (float(v[0]) - 0.5, 1.0) if v.size else (0.0, 1.0)
        step = (v[-1] - v[0]) / (v.size - 1)
        return float(v[0] - step / 2.0), float(v.size * step)

    x_left, x_span = edge(x_vec)
    y_bottom, y_span = edge(y_vec)
    return x_left, y_bottom, x_span, y_span


def mode_color_table(n_lo: int, n_hi: int, cmap: str = "RdBu_r") -> np.ndarray:
    """Discrete ``(ncolors, 3)`` uint8 table — one colour per integer mode ``n``.

    Diverging ``RdBu_r`` (blue = negative n, white ≈ 0, red = positive n) matches
    the classic modespec tool. One colour per integer so adjacent modes stay
    distinct rather than blending.
    """
    import matplotlib as mpl

    ncolors = max(1, int(n_hi) - int(n_lo) + 1)
    cm = mpl.colormaps[cmap].resampled(ncolors)
    return (np.array([cm(i)[:3] for i in range(ncolors)]) * 255).astype(np.uint8)


def nd_masked_for_display(
    result: dict, nd: np.ndarray | None, coh_thresh: float | None
) -> np.ndarray:
    """Dominant-``n`` ``(n_win, n_freq)`` float array, NaN where suppressed.

    If ``nd`` (already gated) is given it is returned as-is; otherwise the
    dominant mode is masked by ``coherence > coh_thresh`` (default
    ``max(c95, 0.3)``). Mirrors :func:`tokeye.sources.viz.plotly_modespec`.
    """
    if nd is not None:
        return np.asarray(nd, dtype=float)
    coh = np.asarray(result["coherence"], dtype=float)
    c95 = float(result.get("c95", 0.0))
    thresh = coh_thresh if coh_thresh is not None else max(c95, 0.3)
    return np.where(coh > thresh, np.asarray(result["n_dominant"], dtype=float), np.nan)


def discrete_mode_image(
    nd_masked: np.ndarray, n_lo: int, n_hi: int, cmap: str = "RdBu_r"
) -> np.ndarray:
    """``(n_win, n_freq)`` mode numbers (NaN = suppressed) → ``(n_freq, n_win, 4)``
    RGBA uint8: frequency on rows (row 0 = lowest), transparent where suppressed."""
    table = mode_color_table(n_lo, n_hi, cmap)
    z = np.asarray(nd_masked, dtype=float).T  # (n_freq, n_win): freq rows, time cols
    finite = np.isfinite(z)
    idx = np.clip(
        (np.nan_to_num(z, nan=float(n_lo)) - int(n_lo)).astype(int),
        0,
        table.shape[0] - 1,
    )
    rgba = np.zeros((*z.shape, 4), dtype=np.uint8)
    rgba[..., :3] = table[idx]
    rgba[..., 3] = np.where(finite, 255, 0).astype(np.uint8)
    return rgba


def mode_ticks(n_lo: int, n_hi: int) -> list[tuple[int, str]]:
    """``[(n, "±n"), …]`` integer colour-bar ticks for the mode range."""
    return [(n, f"{n:+d}") for n in range(int(n_lo), int(n_hi) + 1)]
