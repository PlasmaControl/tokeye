"""Plain-image renderers for the DIII-D tab — spectrogram views + modespec.

Deliberately **not** matplotlib figures: the views render as clean images (no
axes, ticks, frame, or baked-in colorbar), which is what the app wants and is
also faster and leak-free (no figure churn on the slider hot path). The frequency
and time range is read off the app's ``f-min/f-max`` + ``t-min/t-max`` fields
instead of axis labels.

For modespec, the dominant toroidal mode number ``n`` is drawn with a discrete
rainbow (``turbo``) so adjacent modes are visually distinct; the ``n`` → colour
legend lives *outside* the image, as a Gradio ``gr.HTML`` colour key built from
the **same** discrete colours via :func:`mode_color_legend_html`.

Gradio-free on purpose: the offline batch CLI (``tokeye diiid-batch``) imports the
same renderers to write PNGs on a compute node. The scalar/RGB view logic is
reused verbatim from :mod:`tokeye.app.analyze.visualize` (``enhance`` / ``mask`` /
``amplitude``) so that shared module stays untouched.

Wide arrays are binned to display width before rendering (a 512×8000 STFT only
ever shows on ~1200 px), so this stays fast without serializing whole arrays.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from PIL import Image

from tokeye.app.analyze.visualize import amplitude, enhance, mask

DISPLAY_MAX_COLS = 1500
_MODE_CMAP = "turbo"  # discrete rainbow for mode numbers; distinct adjacent bands
_BG_HEX = "#111111"  # masked / no-mode background
_BG_RGB = (0x11, 0x11, 0x11)


# ── column reduction ─────────────────────────────────────────────────────────────
def downsample_cols(arr: np.ndarray, max_cols: int = DISPLAY_MAX_COLS) -> np.ndarray:
    """Block-mean along the time axis so wide *continuous* arrays render fast.

    Works for 2-D ``(H, W)`` and 3-D ``(H, W, 3)`` (RGB) arrays.
    """
    w = arr.shape[1]
    if w <= max_cols:
        return arr
    factor = int(np.ceil(w / max_cols))
    w2 = (w // factor) * factor
    arr = arr[:, :w2]
    if arr.ndim == 2:
        return arr.reshape(arr.shape[0], w2 // factor, factor).mean(axis=2)
    return arr.reshape(arr.shape[0], w2 // factor, factor, arr.shape[2]).mean(axis=2)


def _decimate_cols_nearest(arr: np.ndarray, max_cols: int = DISPLAY_MAX_COLS) -> np.ndarray:
    """Nearest-neighbour column reduction — for *categorical* data (mode numbers).

    Block-mean would blend integer mode numbers into meaningless fractions, so the
    modespec image is decimated by picking representative columns instead.
    """
    w = arr.shape[1]
    if w <= max_cols:
        return arr
    idx = np.linspace(0, w - 1, max_cols).round().astype(int)
    return arr[:, idx]


# ── colour helpers ─────────────────────────────────────────────────────────────
def _to_uint8_rgb(rgb_float: np.ndarray) -> np.ndarray:
    """``(H, W, 3)`` float in [0, 1] → uint8 image array."""
    return (np.clip(rgb_float, 0.0, 1.0) * 255).astype(np.uint8)


def _colorize(arr2d: np.ndarray, cmap_name: str) -> np.ndarray:
    """Scalar ``(H, W)`` → ``(H, W, 3)`` uint8 via a matplotlib colormap (no figure)."""
    a = np.asarray(arr2d, dtype=float)
    finite = a[np.isfinite(a)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 1.0
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    rgba = mpl.colormaps[cmap_name](norm)  # (H, W, 4) float
    return _to_uint8_rgb(rgba[..., :3])


def _mode_colors(n_lo: int, n_hi: int, cmap: str = _MODE_CMAP) -> np.ndarray:
    """Discrete ``(ncolors, 3)`` uint8 table, one colour per integer mode number.

    The single source of truth for mode colours — used by both the modespec image
    and :func:`mode_color_legend_html`, so the legend always matches the plot.
    """
    ncolors = int(n_hi) - int(n_lo) + 1
    cm = mpl.colormaps[cmap].resampled(max(1, ncolors))
    return (np.array([cm(i)[:3] for i in range(max(1, ncolors))]) * 255).astype(np.uint8)


def mode_color_legend_html(n_lo: int, n_hi: int, cmap: str = _MODE_CMAP) -> str:
    """HTML colour key (``n`` → swatch) matching the modespec image colours.

    Rendered in a Gradio ``gr.HTML`` beside the plain modespec image so the rainbow
    is interpretable without a baked-in colorbar.
    """
    colors = _mode_colors(n_lo, n_hi, cmap)
    swatches = []
    for i, n in enumerate(range(int(n_lo), int(n_hi) + 1)):
        r, g, b = (int(v) for v in colors[i])
        swatches.append(
            '<span style="display:inline-flex;align-items:center;margin:0 8px 4px 0;">'
            f'<span style="width:14px;height:14px;border-radius:2px;background:'
            f'rgb({r},{g},{b});border:1px solid rgba(0,0,0,.35);"></span>'
            f'<span style="font-size:12px;margin-left:4px;">n={n:+d}</span></span>'
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;align-items:center;line-height:1.4;">'
        '<span style="font-size:12px;opacity:.7;margin-right:8px;">'
        "dominant toroidal mode n:</span>" + "".join(swatches) + "</div>"
    )


# ── frequency-band display crop ─────────────────────────────────────────────────
def _crop_freq(arr: np.ndarray, stft_meta: dict | None) -> np.ndarray:
    """Row-crop an STFT image to the ``[fmin_khz, fmax_khz]`` band (display only).

    Row ``i`` maps to FFT bin ``i + offset`` (``offset=1`` if the DC bin was
    clipped), i.e. frequency ``(i+offset)·fs/n_fft``. Returns the array unchanged
    when the band is unset, ``fs`` is unknown, or the crop would be degenerate.
    """
    if not stft_meta:
        return arr
    fs = float(stft_meta.get("fs", 0.0))
    if fs <= 0:
        return arr
    n_fft = int(stft_meta.get("n_fft", 1024))
    offset = 1 if bool(stft_meta.get("clip_dc", True)) else 0
    h = arr.shape[0]
    fmin = stft_meta.get("fmin_khz")
    fmax = stft_meta.get("fmax_khz")
    r_lo, r_hi = 0, h
    if fmin is not None and float(fmin) > 0:
        r_lo = max(0, int(np.ceil(float(fmin) * 1e3 * n_fft / fs)) - offset)
    if fmax is not None and float(fmax) > 0:
        r_hi = min(h, int(np.floor(float(fmax) * 1e3 * n_fft / fs)) - offset + 1)
    if r_lo >= r_hi or (r_hi - r_lo) < 2:
        return arr
    return arr[r_lo:r_hi]


# ── renderers ────────────────────────────────────────────────────────────────────
def render_view(
    view_mode: str,
    arr: np.ndarray | None,
    arr_extract: np.ndarray | None,
    out_1_enabled: bool,
    out_2_enabled: bool,
    vmin: float,
    vmax: float,
    threshold: float,
    stft_meta: dict | None = None,
):
    """Render a TokEye spectrogram view as a plain image (no axes/colorbar).

    Same pixel dispatch as ``visualize.show_image`` — reuses ``enhance``/``mask``/
    ``amplitude`` — then crops to the ``f-min/f-max`` band, bins to display width,
    applies the ``gist_heat`` colormap to scalar views, and flips so low frequency
    sits at the bottom. The model output ``arr_extract`` is unchanged (full band);
    only what is shown is cropped.
    """
    if arr is None:
        return None
    try:
        if view_mode == "Original":
            display_arr = np.asarray(arr)
        elif view_mode == "Enhanced":
            if arr_extract is None:
                return None
            display_arr = enhance(arr_extract, out_1_enabled, out_2_enabled, vmin, vmax)
        elif view_mode == "Mask":
            if arr_extract is None:
                return None
            display_arr = mask(arr_extract, out_1_enabled, out_2_enabled, threshold)
        elif view_mode == "Amplitude":
            if arr_extract is None:
                return None
            display_arr = amplitude(
                arr, arr_extract, out_1_enabled, out_2_enabled, threshold
            )
        else:
            return None

        display_arr = _crop_freq(display_arr, stft_meta)
        is_rgb = display_arr.ndim == 3
        display_arr = downsample_cols(display_arr)
        if is_rgb:
            rgb = _to_uint8_rgb(np.asarray(display_arr, dtype=float))
        else:
            rgb = _colorize(display_arr, "gist_heat")
        return Image.fromarray(np.flipud(rgb))
    except Exception:  # noqa: BLE001 - a render failure should degrade to no image
        return None


def render_modespec(
    result: dict,
    nd: np.ndarray | None = None,
    coh_thresh: float | None = None,
    shot: int | None = None,  # noqa: ARG001 - accepted for caller compat (plain image)
    title: str | None = None,  # noqa: ARG001 - accepted for caller compat (plain image)
):
    """Render dominant toroidal mode ``n`` as a plain discrete-rainbow image.

    If ``nd`` (a pre-gated ``(n_win, n_freq)`` array, NaN where suppressed) is given
    it is shown as-is; otherwise the dominant mode is masked by
    ``coherence > coh_thresh`` (default ``max(c95, 0.3)``). Freq is on the vertical
    axis (low at the bottom), time on the horizontal. Colours come from
    :func:`_mode_colors`; the matching legend is :func:`mode_color_legend_html`.
    """
    try:
        coh = np.asarray(result["coherence"])
        n_lo, n_hi = (int(v) for v in result["n_range"])
        c95 = float(result.get("c95", 0.0))
        thresh = coh_thresh if coh_thresh is not None else max(c95, 0.3)

        if nd is None:
            nd_src = np.asarray(result["n_dominant"], dtype=float)
            nd_masked = np.where(coh > thresh, nd_src, np.nan)
        else:
            nd_masked = np.asarray(nd, dtype=float)

        # (n_win, n_freq) -> image (n_freq, n_win): freq on y, time on x.
        img2d = _decimate_cols_nearest(nd_masked.T)
        colors = _mode_colors(n_lo, n_hi)
        ncolors = colors.shape[0]

        out = np.empty(img2d.shape + (3,), dtype=np.uint8)
        out[:] = _BG_RGB
        finite = np.isfinite(img2d)
        idx = np.clip(np.round(img2d[finite]).astype(int) - n_lo, 0, ncolors - 1)
        out[finite] = colors[idx]
        return Image.fromarray(np.flipud(out))
    except Exception:  # noqa: BLE001
        return None
