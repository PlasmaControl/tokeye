"""DIII-D tab — load a shot from MDSplus and overlay the model's mode mask.

This is the "pyspecview + TokEye" view: enter a shot, pick a diagnostic + probe
and a time window, **Load shot** to see the spectrogram (with real kHz/ms axes),
then **Analyze** to overlay the segmentation mask — the same STFT → model →
render pipeline the Analyze tab uses. A **Modespec** view mode runs the classic
toroidal mode-number analysis and can be **gated by the TokEye mask** for much
cleaner modes.

Everything network/MDSplus-touching is deferred to button clicks, so building
this tab (and the app) does no I/O.
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
from tokeye.sources.viz import render_modespec, render_view
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

DIIID_INTRO_MD = """\
Load a **DIII-D shot** straight from MDSplus (`atlas.gat.com`) and see its \
spectrogram with TokEye's mode mask overlaid — like pyspecview, but served in \
your browser and auto-labelled.

1. Enter a **shot** (defaults to the latest), pick a **diagnostic** + **probe**, \
and (optionally) a **time window** in ms.
2. **Load shot** fetches the signal (cached under `$TOKEYE_CACHE`) and shows the \
spectrogram with real frequency/time axes.
3. **Analyze** overlays the coherent/transient mode mask. Threshold / clip \
sliders re-render live (on release) — no recompute.
4. **View Mode → Modespec** runs the classic toroidal mode-number analysis on the \
Mirnov array; tick **Gate with TokEye** to keep only modes the mask confirms.

Fetching needs a node that can reach `atlas.gat.com` (login / somega). Only \
**Fast Magnetics / Mirnov** is verified end-to-end; the others are being wired in.
"""

MODESPEC_NOTE_MD = (
    "Modespec uses the **toroidal Mirnov array** (independent of the probe above). "
    "**Gate with TokEye** intersects the loaded probe's coherent mask with the "
    "mode coherence — load a probe and it gates on Analyze."
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


def _toggle_view_groups(mode: str) -> list[gr.Group]:
    """Show the control group for the selected view (enhanced / mask / modespec)."""
    return [
        gr.Group(visible=(mode == "Enhanced")),
        gr.Group(visible=(mode in ("Mask", "Amplitude"))),
        gr.Group(visible=(mode == "Modespec")),
    ]


def load_shot(shot, diag_key, pointname, t_min, t_max, transform_args):
    """Fetch one DIII-D signal → ``(spectrogram, stft_meta)`` for the states.

    ``stft_meta`` carries the sampling rate + window start so views can label real
    kHz/ms axes; the spectrogram itself is computed exactly as before (the model
    input is unchanged). Degrades to ``(None, None)`` + a ``gr.Warning`` on any
    bad shot / missing MDSplus — mirroring ``wrapper_model_load``.
    """
    if not shot or not pointname or transform_args is None:
        gr.Warning("Enter a shot number and pick a diagnostic/probe first.")
        return None, None

    from tokeye.sources import MDSSource

    tlim = _tlim(t_min, t_max)
    try:
        t, x, fs = MDSSource().fetch(int(shot), str(pointname), tlim)
    except Exception as exc:  # noqa: BLE001 - surface any fetch failure as a toast
        gr.Warning(f"Fetch failed for {int(shot)}/{pointname}: {exc}")
        return None, None

    if x.size == 0:
        gr.Warning(f"{int(shot)}/{pointname} returned no samples in that window.")
        return None, None

    n_fft = transform_args.get("n_fft", DEFAULT_N_FFT)
    hop = transform_args.get("hop_length", DEFAULT_HOP)
    clip_dc = transform_args.get("clip_dc", DEFAULT_CLIP_DC)
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
    return spec, stft_meta


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
    ms_nmin,
    ms_nmax,
    ms_fmin,
    ms_fmax,
    ms_coh,
    ms_gate,
):
    """Analyze button: branch on the view mode.

    TokEye views run the model on the loaded probe; **Modespec** runs the classic
    toroidal analysis (optionally gated by the TokEye mask). Returns
    ``[model, inference_output, modespec_result, image]``.
    """
    if view_mode == "Modespec":
        if not shot:
            gr.Warning("Enter a shot number first.")
            return model, inference_output, modespec_result, None
        from tokeye.sources.mirnov import gate_dominant, run_mode_spectrogram

        try:
            result = run_mode_spectrogram(
                int(shot),
                "toroidal",
                _tlim(t_min, t_max),
                n_range=(int(ms_nmin), int(ms_nmax)),
                f_min_khz=float(ms_fmin),
                f_max_khz=float(ms_fmax),
            )
        except Exception as exc:  # noqa: BLE001
            gr.Warning(f"Modespec failed for shot {int(shot)}: {exc}")
            return model, inference_output, None, None

        nd = None
        if ms_gate:
            if signal_transform is None:
                gr.Warning("Load a probe first to gate with TokEye (showing ungated).")
            else:
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
        img = render_modespec(result, nd=nd, coh_thresh=float(ms_coh), shot=int(shot))
        return model, inference_output, result, img

    # TokEye views
    if signal_transform is None:
        gr.Warning("Load a shot first.")
        return model, inference_output, modespec_result, None

    if view_mode == "Original":
        img = render_view(
            "Original", signal_transform, None, False, False, vmin, vmax, threshold,
            stft_meta,
        )
        return model, inference_output, modespec_result, img

    model = ensure_model(model, model_file, signal_transform)
    if model is None:
        return model, inference_output, modespec_result, None
    inference_output = model_infer(signal_transform, model)
    img = render_view(
        view_mode, signal_transform, inference_output, out_1, out_2, vmin, vmax,
        threshold, stft_meta,
    )
    return model, inference_output, modespec_result, img


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
    ms_coh,
    ms_gate,
):
    """Cheap re-render from cached state (slider release / view switch).

    No fetch, no inference — just re-colors the already-computed spectrogram/mask
    (or re-gates the cached modespec result).
    """
    if view_mode == "Modespec":
        if modespec_result is None:
            return None
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
        return render_modespec(modespec_result, nd=nd, coh_thresh=float(ms_coh))

    return render_view(
        view_mode, signal_transform, inference_output, out_1, out_2, vmin, vmax,
        threshold, stft_meta,
    )


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
            load_shot_btn = gr.Button("Load shot", variant="primary")

        ## Spectrogram (pre-analysis)
        extract_out = gr.Image(label="Spectrogram", type="pil")

        ## Transform settings (same knobs as the Analyze tab)
        with gr.Accordion("STFT settings", open=False), gr.Group():
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
                    ms_fmin = gr.Number(value=5, label="f min (kHz)")
                    ms_fmax = gr.Number(value=200, label="f max (kHz)")
                ms_coh = gr.Slider(
                    0, 1, value=0.5, step=0.01, label="Coherence threshold"
                )
                ms_gate = gr.Checkbox(value=True, label="Gate with TokEye")

            analyze_btn = gr.Button("Analyze", variant="primary")
            visualize_out = gr.Image(label="Visualization", type="pil")

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
        ms_coh,
        ms_gate,
    ]

    # Event handling
    diagnostic.change(fn=_pointname_update, inputs=[diagnostic], outputs=[pointname])

    load_model_btn.click(fn=wrapper_model_load, inputs=[model_file], outputs=[model])

    setup_transform_btn.click(
        fn=setup_stft_transform,
        inputs=[n_fft, hop_length, clip_dc, clip_low_sld, clip_high_sld],
        outputs=[transform_args],
    )

    ## Load shot -> spectrogram (with axes)
    load_shot_btn.click(
        fn=load_shot,
        inputs=[shot, diagnostic, pointname, t_min, t_max, transform_args],
        outputs=[signal_transform, stft_meta],
    ).then(
        fn=render_view,
        inputs=[
            gr.State("Original"),
            signal_transform,
            inference_output,
            gr.State(False),
            gr.State(False),
            vmin_sld,
            vmax_sld,
            threshold_sld,
            stft_meta,
        ],
        outputs=[extract_out],
    )

    ## Switching view mode toggles its controls and re-renders from cache
    view_mode.change(
        fn=_toggle_view_groups,
        inputs=[view_mode],
        outputs=[enhanced_grp, mask_grp, modespec_grp],
    ).then(fn=rerender, inputs=rerender_inputs, outputs=[visualize_out])

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
            ms_nmin,
            ms_nmax,
            ms_fmin,
            ms_fmax,
            ms_coh,
            ms_gate,
        ],
        outputs=[model, inference_output, modespec_result, visualize_out],
    )

    ## Live re-render on slider release (re-color only, no recompute)
    for sld in (vmin_sld, vmax_sld, threshold_sld, ms_coh):
        sld.release(fn=rerender, inputs=rerender_inputs, outputs=[visualize_out])

    return shot
