"""Renderers for the DIII-D tabs — interactive Plotly online, matplotlib offline.

**Online** (the web app) uses Plotly so the plots have real kHz/ms axes plus
zoom/pan/hover, rendered client-side by the browser (Gradio bundles ``plotly.js``):

* :func:`plotly_view` — a TokEye spectrogram view (Original / Enhanced / Mask /
  Amplitude). The pixels are built exactly as before (reusing
  :mod:`tokeye.app.analyze.visualize`'s ``enhance`` / ``mask`` / ``amplitude``),
  colour-composited to an RGB image, then placed on a real time/frequency grid as a
  compact ``go.Image`` (base64 PNG data-URI, so slider re-renders stay small).
* :func:`plotly_modespec` — the dominant toroidal mode number ``n`` as a
  ``go.Heatmap`` with a **discrete** ``turbo`` colorscale (adjacent modes stay
  distinct), an **integer colorbar** (this replaces the old external HTML legend),
  and hover-to-read ``n``.

**Offline** (the batch CLI on a compute node) keeps matplotlib: :func:`render_modespec_png`
renders the same discrete-``n`` map to a PNG with clean axes + a discrete colorbar.
Plotly's static export needs Kaleido/Chromium, which the node lacks, so Plotly is
online-only.

``plotly`` is imported lazily inside the ``plotly_*`` functions so importing this
module stays cheap and the offline CLI (which only needs matplotlib) never requires
it. The scalar/RGB view logic is reused verbatim from
:mod:`tokeye.app.analyze.visualize` so that shared module stays untouched.
"""

from __future__ import annotations

import base64
import io

import matplotlib as mpl
import numpy as np
from PIL import Image

from tokeye.app.analyze.visualize import amplitude, enhance, mask

DISPLAY_MAX_COLS = 1500
_MODE_CMAP = "turbo"  # discrete rainbow for mode numbers; distinct adjacent bands

# Control-room palette — mirrors gui/theme.py::COLORS — keep in sync. Shared with
# the web PALETTE (app/utils/theme.py) so the interactive Plotly figures, the app
# shell, and the native GUI read as one dark theme. The parity test in
# tests/test_app_main.py locks these three copies together.
_PAPER_HEX = "#13151a"  # figure paper / window            (COLORS["bg"])
_PLOT_HEX = "#0c0d11"  # plot canvas (darkest)             (COLORS["plot"])
_LINE_HEX = "#2a2f3a"  # hairline grid / axis lines        (COLORS["line"])
_MUTED_HEX = "#8b93a1"  # axis + label text                (COLORS["muted"])
_ACCENT_HEX = "#45b8cb"  # interactive / active modebar     (COLORS["accent"])
_BG_HEX = _PLOT_HEX  # masked / no-mode bins read as the plot canvas, not a box


# ── column reduction ─────────────────────────────────────────────────────────────
def downsample_cols(arr: np.ndarray, max_cols: int = DISPLAY_MAX_COLS) -> np.ndarray:
    """Block-mean along the time axis so wide *continuous* arrays render fast.

    Works for 2-D ``(H, W)`` and 3-D ``(H, W, 3)`` (RGB) arrays. Rows (frequency)
    are untouched, so a frequency axis stays exact after this reduction.
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


def _col_factor(w: int, max_cols: int = DISPLAY_MAX_COLS) -> int:
    """The block-mean factor :func:`downsample_cols` will use for width ``w``."""
    return int(np.ceil(w / max_cols)) if w > max_cols else 1


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

    The single source of truth for mode colours — used by both the Plotly
    discrete colorscale (:func:`plotly_modespec`) and the matplotlib offline PNG
    (:func:`render_modespec_png`), so plot + colorbar always agree.
    """
    ncolors = int(n_hi) - int(n_lo) + 1
    cm = mpl.colormaps[cmap].resampled(max(1, ncolors))
    return (np.array([cm(i)[:3] for i in range(max(1, ncolors))]) * 255).astype(np.uint8)


# ── frequency-band display crop ─────────────────────────────────────────────────
def _freq_crop_rows(h: int, stft_meta: dict | None) -> tuple[int, int]:
    """Row bounds ``[r_lo, r_hi)`` cropping an STFT image of height ``h`` to the band.

    Row ``i`` maps to FFT bin ``i + offset`` (``offset=1`` if the DC bin was
    clipped), i.e. frequency ``(i+offset)·fs/n_fft``. Returns ``(0, h)`` (no crop)
    when the band is unset, ``fs`` is unknown, or the crop would be degenerate — so
    a caller can always slice ``arr[r_lo:r_hi]`` safely.
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


# ── PNG data-URI ─────────────────────────────────────────────────────────────────
def _png_data_uri(rgb: np.ndarray) -> str:
    """Encode an ``(H, W, 3)`` uint8 image as a ``data:image/png;base64,…`` URI."""
    im = Image.fromarray(np.ascontiguousarray(rgb))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ── TokEye view → display array ────────────────────────────────────────────────────
def _view_display_arr(
    view_mode: str,
    arr: np.ndarray | None,
    arr_extract: np.ndarray | None,
    out_1_enabled: bool,
    out_2_enabled: bool,
    vmin: float,
    vmax: float,
    threshold: float,
) -> np.ndarray | None:
    """Scalar ``(H, W)`` or RGB ``(H, W, 3)`` array for a view mode (same dispatch as
    ``visualize.show_image``), or ``None`` if inputs are missing."""
    if arr is None:
        return None
    if view_mode == "Original":
        return np.asarray(arr)
    if arr_extract is None:
        return None
    if view_mode == "Enhanced":
        return enhance(arr_extract, out_1_enabled, out_2_enabled, vmin, vmax)
    if view_mode == "Mask":
        return mask(arr_extract, out_1_enabled, out_2_enabled, threshold)
    if view_mode == "Amplitude":
        return amplitude(arr, arr_extract, out_1_enabled, out_2_enabled, threshold)
    return None


def _view_axes(stft_meta: dict | None, r_lo: int, factor: int):
    """``(x0, dx, y0, dy, x_title, y_title)`` real time/frequency grid for a view.

    ``r_lo`` is the first kept FFT row (from :func:`_freq_crop_rows`) and ``factor``
    the time-column block-mean factor (:func:`_col_factor`). Falls back to pixel
    indices when the sampling rate is unknown (so the plot still renders).
    """
    fs = float(stft_meta.get("fs", 0.0)) if stft_meta else 0.0
    if fs <= 0:
        return 0.0, 1.0, 0.0, 1.0, "time (window)", "frequency (bin)"
    n_fft = int(stft_meta.get("n_fft", 1024))
    hop = int(stft_meta.get("hop", 256))
    t0 = float(stft_meta.get("t0_ms", 0.0))
    offset = 1 if bool(stft_meta.get("clip_dc", True)) else 0
    df_khz = fs / n_fft / 1e3
    dt0_ms = hop / fs * 1e3  # per original STFT column
    y0 = (r_lo + offset) * df_khz  # centre frequency of cropped row 0
    x0 = t0 + (factor - 1) / 2.0 * dt0_ms  # centre time of binned column 0
    return x0, factor * dt0_ms, y0, df_khz, "time (ms)", "frequency (kHz)"


def _apply_dark_layout(fig) -> None:
    """Paint ``fig`` in the shared control-room dark palette (in place).

    One helper so every interactive DIII-D figure matches the app shell and the
    native GUI instead of stock ``plotly_dark`` blue-grey. Preserves any margin /
    dragmode / titles / ranges already set — it only touches colours.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_PAPER_HEX,
        plot_bgcolor=_PLOT_HEX,
        font={"family": "Inter, Segoe UI, sans-serif", "color": _MUTED_HEX},
        modebar={
            "bgcolor": "rgba(0,0,0,0)",
            "color": _MUTED_HEX,
            "activecolor": _ACCENT_HEX,
        },
    )
    axis = {"gridcolor": _LINE_HEX, "linecolor": _LINE_HEX, "zerolinecolor": _LINE_HEX}
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)


def _empty_fig(message: str):
    """A blank Plotly figure carrying a centred hint (e.g. \"Load a shot\")."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
        font={"size": 14, "color": _MUTED_HEX},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin={"l": 10, "r": 10, "t": 10, "b": 10})
    _apply_dark_layout(fig)
    return fig


# ── renderers ────────────────────────────────────────────────────────────────────
def plotly_view(
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
    """Render a TokEye spectrogram view as an interactive Plotly image.

    Same pixel dispatch as ``visualize.show_image`` (reuses ``enhance`` / ``mask`` /
    ``amplitude``), cropped to the ``f-min/f-max`` band and binned to display width,
    then drawn as a ``go.Image`` on a real time (ms) / frequency (kHz) grid with
    zoom + pan. Scalar views (Original) get the ``gist_heat`` colormap. Returns a
    ``plotly.graph_objects.Figure`` (empty-with-hint on missing inputs).
    """
    import plotly.graph_objects as go

    display_arr = _view_display_arr(
        view_mode, arr, arr_extract, out_1_enabled, out_2_enabled, vmin, vmax, threshold
    )
    if display_arr is None:
        return _empty_fig("Load a shot, then Analyze")

    r_lo, r_hi = _freq_crop_rows(display_arr.shape[0], stft_meta)
    display_arr = display_arr[r_lo:r_hi]
    is_rgb = display_arr.ndim == 3
    factor = _col_factor(display_arr.shape[1])
    display_arr = downsample_cols(display_arr)
    if is_rgb:
        rgb = _to_uint8_rgb(np.asarray(display_arr, dtype=float))
    else:
        rgb = _colorize(display_arr, "gist_heat")

    x0, dx, y0, dy, x_title, y_title = _view_axes(stft_meta, r_lo, factor)
    h, w = rgb.shape[0], rgb.shape[1]
    x_lo, x_hi = x0 - dx / 2.0, x0 + (w - 0.5) * dx
    y_lo, y_hi = y0 - dy / 2.0, y0 + (h - 0.5) * dy

    # Draw the pre-coloured spectrogram as a STRETCHED layout image over normal
    # (aspect-unlocked) axes. A go.Image *trace* forces square pixels (scaleanchor
    # can't be reliably cleared) and squashes the panel into a thin strip; a layout
    # image stretches to fill the axis ranges. The PNG's top row must be the highest
    # frequency, so flip (rgb row 0 is the lowest kept bin); the y-axis stays
    # ascending so low frequency sits at the bottom (standard spectrogram look).
    fig = go.Figure()
    fig.add_layout_image(
        source=_png_data_uri(np.flipud(rgb)),
        xref="x", yref="y", x=x_lo, y=y_hi,
        sizex=(x_hi - x_lo), sizey=(y_hi - y_lo), sizing="stretch", layer="below",
    )
    fig.update_xaxes(title_text=x_title, range=[x_lo, x_hi])
    fig.update_yaxes(title_text=y_title, range=[y_lo, y_hi])
    fig.update_layout(margin={"l": 60, "r": 20, "t": 20, "b": 45}, dragmode="pan")
    _apply_dark_layout(fig)
    return fig


def _discrete_colorscale(n_lo: int, n_hi: int) -> list[list]:
    """Stepwise ``turbo`` colorscale, one flat band per integer mode number.

    Paired with ``zmin=n_lo-0.5``/``zmax=n_hi+0.5`` on the heatmap, integer ``n``
    maps to the centre of its band, so adjacent modes read as distinct colours
    rather than a smooth gradient. Colours come from :func:`_mode_colors`.
    """
    colors = _mode_colors(n_lo, n_hi)
    ncolors = colors.shape[0]
    scale: list[list] = []
    for i in range(ncolors):
        r, g, b = (int(v) for v in colors[i])
        c = f"rgb({r},{g},{b})"
        scale.append([i / ncolors, c])
        scale.append([(i + 1) / ncolors, c])
    return scale


def plotly_modespec(
    result: dict,
    nd: np.ndarray | None = None,
    coh_thresh: float | None = None,
):
    """Dominant toroidal mode ``n`` as an interactive discrete-rainbow heatmap.

    If ``nd`` (a pre-gated ``(n_win, n_freq)`` array, NaN where suppressed) is given
    it is shown as-is; otherwise the dominant mode is masked by
    ``coherence > coh_thresh`` (default ``max(c95, 0.3)``). Frequency is on the
    vertical axis (kHz, low at the bottom), time on the horizontal (ms). The
    integer colorbar replaces the old external HTML legend; hover reads off
    ``t / f / n``. Suppressed bins are transparent over a dark panel.
    """
    import plotly.graph_objects as go

    coh = np.asarray(result["coherence"])
    n_lo, n_hi = (int(v) for v in result["n_range"])
    c95 = float(result.get("c95", 0.0))
    thresh = coh_thresh if coh_thresh is not None else max(c95, 0.3)

    if nd is None:
        nd_src = np.asarray(result["n_dominant"], dtype=float)
        nd_masked = np.where(coh > thresh, nd_src, np.nan)
    else:
        nd_masked = np.asarray(nd, dtype=float)

    # (n_win, n_freq) -> heatmap z (n_freq, n_win): freq on y, time on x.
    z = nd_masked.T
    f_khz = np.asarray(result["freq_khz"], dtype=float)
    t_ms = np.asarray(result["t_win_ms"], dtype=float)
    ticks = list(range(n_lo, n_hi + 1))

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=t_ms,
            y=f_khz,
            zmin=n_lo - 0.5,
            zmax=n_hi + 0.5,
            colorscale=_discrete_colorscale(n_lo, n_hi),
            colorbar={
                "title": "mode n",
                "tickmode": "array",
                "tickvals": ticks,
                "ticktext": [f"{n:+d}" for n in ticks],
            },
            zsmooth=False,
            hoverongaps=False,
            hovertemplate="t=%{x:.1f} ms<br>f=%{y:.0f} kHz<br>n=%{z:.0f}<extra></extra>",
        )
    )
    fig.update_layout(margin={"l": 60, "r": 20, "t": 20, "b": 45}, dragmode="pan")
    fig.update_xaxes(title_text="time (ms)")
    fig.update_yaxes(title_text="frequency (kHz)")
    # Suppressed (NaN) bins fall through to plot_bgcolor == _BG_HEX == _PLOT_HEX.
    _apply_dark_layout(fig)
    return fig


def render_modespec_png(
    result: dict,
    nd: np.ndarray | None = None,
    coh_thresh: float | None = None,
    shot: int | None = None,
    title: str | None = None,
):
    """Matplotlib dominant-``n`` map as a ``PIL.Image`` — for the offline batch PNGs.

    Plotly's static export needs Kaleido/Chromium (absent on compute nodes), so the
    offline gallery stays matplotlib. Renders the same discrete-``turbo`` mode map
    as :func:`plotly_modespec` but with clean freq/time axes and a discrete colorbar
    baked in. Gating semantics (``nd`` / ``coh_thresh``) match exactly.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.figure import Figure

        coh = np.asarray(result["coherence"])
        n_lo, n_hi = (int(v) for v in result["n_range"])
        c95 = float(result.get("c95", 0.0))
        thresh = coh_thresh if coh_thresh is not None else max(c95, 0.3)

        if nd is None:
            nd_src = np.asarray(result["n_dominant"], dtype=float)
            nd_masked = np.where(coh > thresh, nd_src, np.nan)
        else:
            nd_masked = np.asarray(nd, dtype=float)

        z = np.ma.masked_invalid(nd_masked.T)  # (n_freq, n_win); masked = transparent
        f_khz = np.asarray(result["freq_khz"], dtype=float)
        t_ms = np.asarray(result["t_win_ms"], dtype=float)
        extent = [float(t_ms[0]), float(t_ms[-1]), float(f_khz[0]), float(f_khz[-1])]

        cmap = ListedColormap(_mode_colors(n_lo, n_hi) / 255.0).with_extremes(bad=_BG_HEX)
        bounds = np.arange(n_lo - 0.5, n_hi + 1.5, 1.0)
        norm = BoundaryNorm(bounds, cmap.N)

        fig = Figure(figsize=(7.5, 4.0), facecolor=_BG_HEX, layout="constrained")
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, facecolor=_BG_HEX)
        im = ax.imshow(
            z, aspect="auto", origin="lower", extent=extent, cmap=cmap, norm=norm,
            interpolation="nearest",
        )
        ax.set_xlabel("time (ms)", color="w")
        ax.set_ylabel("frequency (kHz)", color="w")
        ttl = title or (f"Shot {shot} — toroidal mode n" if shot else "toroidal mode n")
        ax.set_title(ttl, color="w")
        ax.tick_params(colors="w")
        for spine in ax.spines.values():
            spine.set_color("w")
        cbar = fig.colorbar(im, ax=ax, ticks=list(range(n_lo, n_hi + 1)))
        cbar.set_label("mode n", color="w")
        cbar.ax.yaxis.set_tick_params(color="w")
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_color("w")

        canvas.draw()
        w, h = canvas.get_width_height()
        rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        return Image.fromarray(rgba[..., :3].copy())
    except Exception:  # noqa: BLE001 - a render failure should degrade to no image
        return None
