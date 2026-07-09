"""DIII-D tab — load a shot from MDSplus and overlay the model's mode mask.

This is the "pyspecview + TokEye" view: enter a shot, pick a diagnostic + probe
(the time window auto-fills to the data range), **Load shot** to see the
spectrogram, then **Analyze** to overlay the segmentation mask — the same STFT →
model → render pipeline the Analyze tab uses. A **Modespec** view mode runs the
classic toroidal mode-number analysis and can be **gated by the TokEye mask** for
much cleaner modes.

Views render as plain images (no axis chrome); the frequency/time range is read
off the **f-min/f-max** (STFT settings) + **t-min/t-max** fields. Everything
network/MDSplus-touching is deferred to callbacks, so building this tab (and the
app) does no I/O.
"""

from __future__ import annotations

import logging

import gradio as gr

from tokeye.app.analyze.analyze import (
    ensure_model,
    setup_stft_transform,
    wrapper_model_load,
)
from tokeye.app.analyze.load import find_models, model_infer
from tokeye.hub import DEFAULT_MODEL, MODEL_REGISTRY
from tokeye.inference import signal_to_spectrogram
from tokeye.sources import DIAGNOSTICS, diagnostic_dropdown_choices
from tokeye.sources.viz import mode_color_legend_html, render_modespec, render_view
from tokeye.transforms import (
    DEFAULT_CLIP_DC,
    DEFAULT_CLIP_HIGH,
    DEFAULT_CLIP_LOW,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
)

logger = logging.getLogger(__name__)

_DEFAULT_DIAG = "mag"
_VIEW_MODES = ["Original", "Enhanced", "Mask", "Amplitude", "Modespec"]
_DEFAULT_FMIN_KHZ = 5.0
_DEFAULT_FMAX_KHZ = 250.0

DIIID_INTRO_MD = """\
Load a **DIII-D shot** straight from MDSplus (`atlas.gat.com`) and see its \
spectrogram with TokEye's mode mask overlaid — like pyspecview, served in your \
browser.

1. Enter a **shot** (defaults to the latest), pick a **diagnostic** + **probe** — \
the **t-min/t-max** fields auto-fill to that signal's data window (narrow them to \
go faster).
2. **Load shot** fetches the signal (cached under `$TOKEYE_CACHE`) and shows the \
spectrogram. Use **STFT settings → f-min/f-max** to crop the band you care about.
3. **Analyze** overlays the coherent/transient mode mask. Threshold / clip / band \
sliders re-render live — no recompute.
4. **View Mode → Modespec** runs the classic toroidal mode-number analysis on the \
Mirnov array; tick **Gate with TokEye** to keep only modes the mask confirms.

Fetching needs a node that can reach `atlas.gat.com` (login / somega).
"""

MODESPEC_NOTE_MD = (
    "Modespec uses the **toroidal Mirnov array** (independent of the probe above). "
    "Its frequency band + decimation come from **STFT settings** (decimation speeds "
    "it up a lot). **Gate with TokEye** intersects the loaded probe's coherent mask "
    "with the mode coherence — load a probe and it gates on Analyze."
)


def _pointname_update(diag_key: str) -> gr.Dropdown:
    """Repopulate the probe dropdown when the diagnostic changes."""
    diag = DIAGNOSTICS.get(diag_key)
    if diag is None:
        return gr.Dropdown(choices=[], value=None)
    return gr.Dropdown(choices=list(diag.pointnames), value=diag.default)


def _tlim(t_min, t_max) -> tuple[float, float] | None:
    if t_min is not None and t_max is not None and float(t_max) > float(t_min):
        return (float(t_min), float(t_max))
    return None


def _meta_band(stft_meta: dict | None, fmin, fmax) -> dict | None:
    """Copy ``stft_meta`` with the live display band merged in (display-only crop)."""
    if not stft_meta:
        return stft_meta
    m = dict(stft_meta)
    m["fmin_khz"] = float(fmin) if fmin not in (None, "") else None
    m["fmax_khz"] = float(fmax) if fmax not in (None, "") else None
    return m


def _toggle_view_groups(mode: str) -> list[gr.Group]:
    """Show the control group for the selected view (enhanced / mask / modespec)."""
    return [
        gr.Group(visible=(mode == "Enhanced")),
        gr.Group(visible=(mode in ("Mask", "Amplitude"))),
        gr.Group(visible=(mode == "Modespec")),
    ]


def fill_window(shot, pointname):
    """Auto-fill (t_min, t_max) with the signal's data window on shot/probe change.

    Uses a cheap scalar-TDI bounds query (only the endpoints cross the wire). Best
    effort: leaves the fields as-is if the shot/probe is empty or bounds can't be
    read (atlas unreachable, off-cluster), so it never clobbers a manual entry.
    """
    if not shot or not pointname:
        return gr.update(), gr.update()
    from tokeye.sources import MDSSource

    bounds = MDSSource.time_bounds(int(shot), str(pointname))
    if not bounds:
        return gr.update(), gr.update()
    return round(float(bounds[0]), 2), round(float(bounds[1]), 2)


def load_shot(
    shot,
    diag_key,
    pointname,
    t_min,
    t_max,
    transform_args,
    decimation,
    progress=gr.Progress(),
):
    """Fetch one DIII-D signal → ``(spectrogram, stft_meta)`` for the states.

    Optionally decimates the signal first (STFT-settings **Decimation** knob, like
    ``n_fft``/``hop``) and records the (post-decimation) sampling rate + window
    start in ``stft_meta`` so views can crop/label in real kHz/ms. Degrades to
    ``(None, None)`` + a ``gr.Warning`` on any bad shot / missing MDSplus.
    """
    if not shot or not pointname or transform_args is None:
        gr.Warning("Enter a shot number and pick a diagnostic/probe first.")
        return None, None

    from tokeye.sources import MDSSource

    tlim = _tlim(t_min, t_max)
    progress(0.1, desc=f"Fetching {int(shot)}/{pointname} …")
    try:
        t, x, fs = MDSSource().fetch(int(shot), str(pointname), tlim)
    except Exception as exc:  # noqa: BLE001 - surface any fetch failure as a toast
        gr.Warning(f"Fetch failed for {int(shot)}/{pointname}: {exc}")
        return None, None

    if x.size == 0:
        gr.Warning(f"{int(shot)}/{pointname} returned no samples in that window.")
        return None, None

    # Decimation (anti-aliased) — a transform knob; default 1 = off. Lowers the
    # sample count (and Nyquist) fed to the STFT/model, mirroring n_fft/hop.
    d = int(decimation) if decimation else 1
    if d > 1 and x.size > 64:
        progress(0.5, desc=f"Decimating ×{d} …")
        try:
            from scipy.signal import decimate

            x = decimate(x, d, ftype="fir")
            t = t[::d][: x.size]
            fs = fs / d
        except Exception as exc:  # noqa: BLE001 - fall back to full rate
            gr.Warning(f"Decimation ×{d} failed, using full rate: {exc}")

    n_fft = transform_args.get("n_fft", DEFAULT_N_FFT)
    hop = transform_args.get("hop_length", DEFAULT_HOP)
    clip_dc = transform_args.get("clip_dc", DEFAULT_CLIP_DC)
    progress(0.7, desc="Computing spectrogram …")
    spec = signal_to_spectrogram(
        x,
        n_fft=n_fft,
        hop=hop,
        clip_dc=clip_dc,
        clip_low=transform_args.get("percentile_low", DEFAULT_CLIP_LOW),
        clip_high=transform_args.get("percentile_high", DEFAULT_CLIP_HIGH),
    )
    stft_meta = {
        "fs": float(fs),
        "t0_ms": float(t[0]) if t.size else 0.0,
        "n_fft": int(n_fft),
        "hop": int(hop),
        "clip_dc": bool(clip_dc),
    }
    progress(1.0, desc="Done")
    return spec, stft_meta


def render_spectrogram(signal_transform, stft_meta, stft_fmin, stft_fmax):
    """Plain pre-analysis spectrogram (Original view), cropped to the display band."""
    return render_view(
        "Original",
        signal_transform,
        None,
        False,
        False,
        0,
        100,
        0.5,
        _meta_band(stft_meta, stft_fmin, stft_fmax),
    )


def run_analyze(
    view_mode,
    shot,
    t_min,
    t_max,
    model,
    model_file,
    signal_transform,
    stft_meta,
    inference_output,
    modespec_result,
    out_1,
    out_2,
    vmin,
    vmax,
    threshold,
    stft_fmin,
    stft_fmax,
    decimation,
    ms_nmin,
    ms_nmax,
    ms_coh,
    ms_gate,
    progress=gr.Progress(),
):
    """Analyze button: branch on the view mode.

    TokEye views run the model on the loaded probe; **Modespec** runs the classic
    toroidal analysis (band + decimation from STFT settings; optionally gated by
    the TokEye mask). Returns
    ``[model, inference_output, modespec_result, image, legend_html]``.
    """
    meta = _meta_band(stft_meta, stft_fmin, stft_fmax)

    if view_mode == "Modespec":
        if not shot:
            gr.Warning("Enter a shot number first.")
            return model, inference_output, modespec_result, None, ""
        from tokeye.sources.mirnov import gate_dominant, run_mode_spectrogram

        progress(0.15, desc="Fetching Mirnov array + computing mode spectrogram …")
        try:
            result = run_mode_spectrogram(
                int(shot),
                "toroidal",
                _tlim(t_min, t_max),
                decimation=int(decimation) if decimation else None,
                n_range=(int(ms_nmin), int(ms_nmax)),
                f_min_khz=float(stft_fmin),
                f_max_khz=float(stft_fmax),
            )
        except Exception as exc:  # noqa: BLE001
            gr.Warning(f"Modespec failed for shot {int(shot)}: {exc}")
            return model, inference_output, None, None, ""

        nd = None
        if ms_gate:
            if signal_transform is None:
                gr.Warning("Load a probe first to gate with TokEye (showing ungated).")
            else:
                progress(0.8, desc="Gating with TokEye …")
                model = ensure_model(model, model_file, signal_transform)
                if model is not None:
                    inference_output = model_infer(signal_transform, model)
                    try:
                        nd = gate_dominant(
                            result,
                            inference_output,
                            stft_meta,
                            mask_threshold=float(threshold),
                            coh_thresh=float(ms_coh),
                        )
                    except Exception as exc:  # noqa: BLE001
                        gr.Warning(f"Gating failed (showing ungated): {exc}")
        progress(0.95, desc="Rendering modes …")
        img = render_modespec(result, nd=nd, coh_thresh=float(ms_coh))
        n_lo, n_hi = result["n_range"]
        return model, inference_output, result, img, mode_color_legend_html(n_lo, n_hi)

    # TokEye views
    if signal_transform is None:
        gr.Warning("Load a shot first.")
        return model, inference_output, modespec_result, None, ""

    if view_mode == "Original":
        img = render_view(
            "Original", signal_transform, None, False, False, vmin, vmax, threshold,
            meta,
        )
        return model, inference_output, modespec_result, img, ""

    progress(0.3, desc="Running model …")
    model = ensure_model(model, model_file, signal_transform)
    if model is None:
        return model, inference_output, modespec_result, None, ""
    inference_output = model_infer(signal_transform, model)
    progress(0.9, desc="Rendering …")
    img = render_view(
        view_mode, signal_transform, inference_output, out_1, out_2, vmin, vmax,
        threshold, meta,
    )
    return model, inference_output, modespec_result, img, ""


def rerender(
    view_mode,
    signal_transform,
    stft_meta,
    inference_output,
    modespec_result,
    out_1,
    out_2,
    vmin,
    vmax,
    threshold,
    stft_fmin,
    stft_fmax,
    ms_coh,
    ms_gate,
):
    """Cheap re-render from cached state (slider release / view switch / band change).

    No fetch, no inference — re-colors the already-computed spectrogram/mask (or
    re-gates the cached modespec result). Returns ``(image, legend_html)``.
    """
    if view_mode == "Modespec":
        if modespec_result is None:
            return None, ""
        nd = None
        if ms_gate and inference_output is not None and stft_meta is not None:
            from tokeye.sources.mirnov import gate_dominant

            try:
                nd = gate_dominant(
                    modespec_result,
                    inference_output,
                    stft_meta,
                    mask_threshold=float(threshold),
                    coh_thresh=float(ms_coh),
                )
            except Exception:  # noqa: BLE001
                nd = None
        img = render_modespec(modespec_result, nd=nd, coh_thresh=float(ms_coh))
        n_lo, n_hi = modespec_result["n_range"]
        return img, mode_color_legend_html(n_lo, n_hi)

    img = render_view(
        view_mode, signal_transform, inference_output, out_1, out_2, vmin, vmax,
        threshold, _meta_band(stft_meta, stft_fmin, stft_fmax),
    )
    return img, ""


def diiid_tab():
    # User Interface
    with gr.Column():
        with gr.Accordion("What this tab does", open=False):
            gr.Markdown(DIIID_INTRO_MD)

        ## Model
        with gr.Group():
            model_file = gr.Dropdown(
                label="Analysis Model",
                info=(
                    "Built-in models download from Hugging Face on first load "
                    "(~30 MB, cached). Local model/*.pt files also listed."
                ),
                choices=list(MODEL_REGISTRY) + find_models(),
                value=DEFAULT_MODEL,
                interactive=True,
                allow_custom_value=True,
            )
            load_model_btn = gr.Button("Load Model")

        ## Shot input
        with gr.Group():
            with gr.Row():
                shot = gr.Number(label="Shot", value=None, precision=0)
                diagnostic = gr.Dropdown(
                    label="Diagnostic",
                    choices=diagnostic_dropdown_choices(),
                    value=_DEFAULT_DIAG,
                    interactive=True,
                )
                pointname = gr.Dropdown(
                    label="Probe / pointname",
                    choices=list(DIAGNOSTICS[_DEFAULT_DIAG].pointnames),
                    value=DIAGNOSTICS[_DEFAULT_DIAG].default,
                    interactive=True,
                    allow_custom_value=True,
                )
            with gr.Row():
                t_min = gr.Number(label="t min (ms)", value=None)
                t_max = gr.Number(label="t max (ms)", value=None)

        ## Transform settings — ABOVE the spectrogram (controls above their output)
        with gr.Accordion("STFT settings", open=False), gr.Group():
            with gr.Row():
                stft_fmin = gr.Number(value=_DEFAULT_FMIN_KHZ, label="f min (kHz)")
                stft_fmax = gr.Number(value=_DEFAULT_FMAX_KHZ, label="f max (kHz)")
                decimation = gr.Number(
                    value=1,
                    precision=0,
                    minimum=1,
                    label="Decimation (1 = off)",
                    info="Applied on Load; also speeds up Modespec.",
                )
            clip_low_sld = gr.Slider(
                0, 100, value=DEFAULT_CLIP_LOW, step=1, label="% Clip Low"
            )
            clip_high_sld = gr.Slider(
                0, 100, value=DEFAULT_CLIP_HIGH, step=1, label="% Clip High"
            )
            n_fft = gr.Slider(
                256, 2048, value=DEFAULT_N_FFT, step=256, label="Number of Bins"
            )
            hop_length = gr.Slider(64, 512, value=DEFAULT_HOP, step=64, label="Hop Size")
            clip_dc = gr.Checkbox(value=DEFAULT_CLIP_DC, label="Remove DC (Bottom) Bin")
            setup_transform_btn = gr.Button("Apply Transform Settings")

        ## Load shot
        load_shot_btn = gr.Button("Load shot", variant="primary")

        ## Spectrogram (pre-analysis)
        extract_out = gr.Image(label="Spectrogram", type="pil")

        ## Visualization (mirror of the Analyze-tab controls + Modespec)
        with gr.Column():
            with gr.Row():
                view_mode = gr.Radio(
                    choices=_VIEW_MODES,
                    value="Enhanced",
                    label="View Mode",
                )
                with gr.Group():
                    out_1_chk = gr.Checkbox(value=True, label="Coherent Events")
                    out_2_chk = gr.Checkbox(value=True, label="Transient Events")
            with gr.Group(visible=True) as enhanced_grp, gr.Column():
                vmin_sld = gr.Slider(0, 100, value=0, step=1, label="% Min Clip")
                vmax_sld = gr.Slider(0, 100, value=100, step=1, label="% Max Clip")
            with gr.Group(visible=False) as mask_grp, gr.Column():
                threshold_sld = gr.Slider(0, 1, value=0.5, step=0.01, label="Threshold")
            with gr.Group(visible=False) as modespec_grp, gr.Column():
                gr.Markdown(MODESPEC_NOTE_MD)
                with gr.Row():
                    ms_nmin = gr.Number(value=-5, precision=0, label="n min")
                    ms_nmax = gr.Number(value=5, precision=0, label="n max")
                ms_coh = gr.Slider(
                    0, 1, value=0.5, step=0.01, label="Coherence threshold"
                )
                ms_gate = gr.Checkbox(value=True, label="Gate with TokEye")

            analyze_btn = gr.Button("Analyze", variant="primary")
            visualize_out = gr.Image(label="Visualization", type="pil")
            modespec_legend = gr.HTML()

    # State variables (same shape as the Analyze tab + stft_meta / modespec)
    model = gr.State()
    signal_transform = gr.State()
    stft_meta = gr.State()
    inference_output = gr.State()
    modespec_result = gr.State()
    transform_args = gr.State(
        setup_stft_transform(
            DEFAULT_N_FFT,
            DEFAULT_HOP,
            DEFAULT_CLIP_DC,
            DEFAULT_CLIP_LOW,
            DEFAULT_CLIP_HIGH,
        )
    )

    # Shared input bundles
    rerender_inputs = [
        view_mode,
        signal_transform,
        stft_meta,
        inference_output,
        modespec_result,
        out_1_chk,
        out_2_chk,
        vmin_sld,
        vmax_sld,
        threshold_sld,
        stft_fmin,
        stft_fmax,
        ms_coh,
        ms_gate,
    ]
    spectrogram_inputs = [signal_transform, stft_meta, stft_fmin, stft_fmax]

    # Event handling
    diagnostic.change(fn=_pointname_update, inputs=[diagnostic], outputs=[pointname])

    # Auto-fill the time window to the signal's data range on shot/probe change.
    shot.change(fn=fill_window, inputs=[shot, pointname], outputs=[t_min, t_max])
    pointname.change(fn=fill_window, inputs=[shot, pointname], outputs=[t_min, t_max])

    load_model_btn.click(fn=wrapper_model_load, inputs=[model_file], outputs=[model])

    setup_transform_btn.click(
        fn=setup_stft_transform,
        inputs=[n_fft, hop_length, clip_dc, clip_low_sld, clip_high_sld],
        outputs=[transform_args],
    )

    ## Load shot -> spectrogram (plain image, cropped to the display band)
    load_shot_btn.click(
        fn=load_shot,
        inputs=[shot, diagnostic, pointname, t_min, t_max, transform_args, decimation],
        outputs=[signal_transform, stft_meta],
    ).then(
        fn=render_spectrogram, inputs=spectrogram_inputs, outputs=[extract_out]
    )

    ## Switching view mode toggles its controls and re-renders from cache
    view_mode.change(
        fn=_toggle_view_groups,
        inputs=[view_mode],
        outputs=[enhanced_grp, mask_grp, modespec_grp],
    ).then(fn=rerender, inputs=rerender_inputs, outputs=[visualize_out, modespec_legend])

    ## One-click Analyze: branch on view mode (TokEye model vs Modespec)
    analyze_btn.click(
        fn=run_analyze,
        inputs=[
            view_mode,
            shot,
            t_min,
            t_max,
            model,
            model_file,
            signal_transform,
            stft_meta,
            inference_output,
            modespec_result,
            out_1_chk,
            out_2_chk,
            vmin_sld,
            vmax_sld,
            threshold_sld,
            stft_fmin,
            stft_fmax,
            decimation,
            ms_nmin,
            ms_nmax,
            ms_coh,
            ms_gate,
        ],
        outputs=[model, inference_output, modespec_result, visualize_out, modespec_legend],
    )

    ## Live re-render on slider release (re-color only, no recompute)
    for sld in (vmin_sld, vmax_sld, threshold_sld, ms_coh):
        sld.release(
            fn=rerender, inputs=rerender_inputs, outputs=[visualize_out, modespec_legend]
        )

    ## Frequency band changes re-crop both images live (display-only, no recompute)
    for band in (stft_fmin, stft_fmax):
        band.change(fn=render_spectrogram, inputs=spectrogram_inputs, outputs=[extract_out])
        band.change(
            fn=rerender, inputs=rerender_inputs, outputs=[visualize_out, modespec_legend]
        )

    return shot
