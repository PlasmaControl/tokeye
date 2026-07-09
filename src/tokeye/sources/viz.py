"""Matplotlib renderers for the DIII-D tab — spectrogram views + modespec, with
real frequency (kHz) and time (ms) axes.

Gradio-free on purpose: the offline batch CLI (``tokeye diiid-batch``) imports the
same renderers to write PNGs on a compute node. The scalar/RGB view logic is
reused verbatim from :mod:`tokeye.app.analyze.visualize` (``enhance`` / ``mask`` /
``amplitude`` / ``render_image``) — only the final plotting step differs (axes,
labels, colorbar) so the shared module stays untouched.

Wide arrays are binned to display width before plotting (Plotly-style full-array
serialization is exactly what we avoid); a 512×8000 STFT only ever shows on ~1200
px, so binning to ~1500 columns is visually lossless and keeps rendering fast.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")  # headless: safe on batch nodes and inside the app worker
import matplotlib.pyplot as plt
import numpy as np

from tokeye.app.analyze.visualize import amplitude, enhance, mask, render_image

DISPLAY_MAX_COLS = 1500


def downsample_cols(arr: np.ndarray, max_cols: int = DISPLAY_MAX_COLS) -> np.ndarray:
    """Block-mean along the time axis so wide arrays render fast. Extent-preserving.

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


def stft_axes(shape: tuple[int, ...], stft_meta: dict | None) -> tuple[list | None, str, str]:
    """Return ``(extent, xlabel, ylabel)`` for an STFT image ``(H, W[, 3])``.

    ``extent`` is ``[t0_ms, t1_ms, f0_khz, f1_khz]`` when a sampling rate is known,
    else ``None`` (labels fall back to frame/bin indices).
    """
    w = shape[1]
    fs = float(stft_meta.get("fs", 0.0)) if stft_meta else 0.0
    if not stft_meta or fs <= 0:
        return None, "Time (frames)", "Frequency (bin)"
    n_fft = int(stft_meta.get("n_fft", 1024))
    hop = int(stft_meta.get("hop", 256))
    t0 = float(stft_meta.get("t0_ms", 0.0))
    clip_dc = bool(stft_meta.get("clip_dc", True))

    f0_khz = (fs / n_fft if clip_dc else 0.0) / 1e3
    f1_khz = (fs / 2.0) / 1e3
    t1 = t0 + (w - 1) * hop / fs * 1e3  # frame spacing = hop/fs seconds
    return [t0, t1, f0_khz, f1_khz], "Time (ms)", "Frequency (kHz)"


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
    """Render a TokEye spectrogram view (Original/Enhanced/Mask/Amplitude) with axes.

    Same dispatch as ``visualize.show_image`` — reuses ``enhance``/``mask``/
    ``amplitude`` for the pixel data — but plots with real kHz/ms axes (and a
    colorbar for scalar views) instead of a bare ``axis("off")`` heatmap.
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

        is_rgb = display_arr.ndim == 3
        extent, xlabel, ylabel = stft_axes(display_arr.shape, stft_meta)
        display_arr = downsample_cols(display_arr)

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=(12, 4))
            im = ax.imshow(
                display_arr,
                aspect="auto",
                origin="lower",
                cmap=None if is_rgb else "gist_heat",
                extent=extent,
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if not is_rgb:
                fig.colorbar(im, ax=ax, pad=0.01)
            fig.tight_layout()
        return render_image(fig)
    except Exception:  # noqa: BLE001 - a render failure should degrade to no image
        return None


def render_modespec(
    result: dict,
    nd: np.ndarray | None = None,
    coh_thresh: float | None = None,
    shot: int | None = None,
    title: str | None = None,
):
    """Render the dominant toroidal mode number ``n`` vs (freq kHz, time ms).

    Single-panel analogue of ``modespec.plot_modespec`` panel 2. If ``nd`` (a
    pre-gated ``(n_win, n_freq)`` array, NaN where suppressed) is given it is shown
    as-is; otherwise the dominant mode is masked by ``coherence > coh_thresh``
    (defaulting to ``max(c95, 0.3)``).
    """
    try:
        t = np.asarray(result["t_win_ms"])
        f = np.asarray(result["freq_khz"])
        coh = np.asarray(result["coherence"])
        n_lo, n_hi = result["n_range"]
        c95 = float(result.get("c95", 0.0))
        thresh = coh_thresh if coh_thresh is not None else max(c95, 0.3)

        if nd is None:
            nd_src = np.asarray(result["n_dominant"], dtype=float)
            nd_masked = np.where(coh > thresh, nd_src, np.nan)
        else:
            nd_masked = np.asarray(nd, dtype=float)

        # (n_win, n_freq) -> image (n_freq, n_win): freq on y, time on x.
        img = downsample_cols(nd_masked.T)
        extent = [float(t[0]), float(t[-1]), float(f[0]), float(f[-1])]
        ncolors = int(n_hi) - int(n_lo) + 1

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=(12, 4))
            cmap = plt.get_cmap("RdBu_r", ncolors).with_extremes(bad="#111111")
            im = ax.imshow(
                img,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                extent=extent,
                vmin=n_lo - 0.5,
                vmax=n_hi + 0.5,
            )
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Frequency (kHz)")
            ax.set_title(
                title
                if title is not None
                else (f"Shot {shot} — toroidal mode n" if shot else "Toroidal mode n")
            )
            cb = fig.colorbar(im, ax=ax, pad=0.01, ticks=range(int(n_lo), int(n_hi) + 1))
            cb.set_label("n")
            ax.text(
                0.01,
                0.97,
                f"coh > {thresh:.2f} (c95={c95:.2f})",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
                color="white",
            )
            fig.tight_layout()
        return render_image(fig)
    except Exception:  # noqa: BLE001
        return None
