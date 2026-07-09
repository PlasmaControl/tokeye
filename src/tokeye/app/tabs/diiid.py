"""DIII-D tab — load a shot from MDSplus and overlay the model's mode mask.

This is the "pyspecview + TokEye" view: enter a shot, pick a diagnostic + probe
and a time window, **Load shot** to see the spectrogram, then **Analyze** to
overlay the segmentation mask — the same STFT → model → render pipeline the
Analyze tab uses, so the fetched signal plugs into unchanged downstream code.

Everything network/MDSplus-touching is deferred to button clicks, so building
this tab (and the app) does no I/O.
"""

from __future__ import annotations

import logging

import gradio as gr

from tokeye.app.analyze.analyze import (
    ensure_model,
    setup_stft_transform,
    toggle_view_groups,
    wrapper_model_load,
)
from tokeye.app.analyze.load import find_models, model_infer
from tokeye.app.analyze.visualize import show_image
from tokeye.hub import DEFAULT_MODEL, MODEL_REGISTRY
from tokeye.inference import signal_to_spectrogram
from tokeye.sources import DIAGNOSTICS, diagnostic_dropdown_choices
from tokeye.transforms import (
    DEFAULT_CLIP_DC,
    DEFAULT_CLIP_HIGH,
    DEFAULT_CLIP_LOW,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
)

logger = logging.getLogger(__name__)

_DEFAULT_DIAG = "mag"

DIIID_INTRO_MD = """\
Load a **DIII-D shot** straight from MDSplus (`atlas.gat.com`) and see its \
spectrogram with TokEye's mode mask overlaid — like pyspecview, but served in \
your browser and auto-labelled.

1. Enter a **shot**, pick a **diagnostic** + **probe**, and (optionally) a \
**time window** in ms.
2. **Load shot** fetches the signal (cached under `$TOKEYE_CACHE`) and shows the \
spectrogram.
3. **Analyze** downloads the model on first use (~30 MB, cached) and overlays \
the coherent/transient mode mask.

Fetching needs a node that can reach `atlas.gat.com` (login / somega). Only \
**Fast Magnetics / Mirnov** is verified end-to-end; ECE/CO2/BES are scaffolds.
"""


def _pointname_update(diag_key: str) -> gr.Dropdown:
    """Repopulate the probe dropdown when the diagnostic changes."""
    diag = DIAGNOSTICS.get(diag_key)
    if diag is None:
        return gr.Dropdown(choices=[], value=None)
    return gr.Dropdown(choices=list(diag.pointnames), value=diag.default)


def load_shot(shot, diag_key, pointname, t_min, t_max, transform_args):
    """Fetch one DIII-D signal and return its spectrogram for the state.

    Degrades to a ``gr.Warning`` + ``None`` (rather than raising) so a bad
    shot/pointname or missing MDSplus surfaces as a toast instead of crashing
    the pipeline — mirroring ``wrapper_model_load`` in the Analyze tab.
    """
    if not shot or not pointname or transform_args is None:
        gr.Warning("Enter a shot number and pick a diagnostic/probe first.")
        return None

    from tokeye.sources import MDSSource

    tlim = None
    if t_min is not None and t_max is not None and float(t_max) > float(t_min):
        tlim = (float(t_min), float(t_max))

    try:
        _t, x, _fs = MDSSource().fetch(int(shot), str(pointname), tlim)
    except Exception as exc:  # noqa: BLE001 - surface any fetch failure as a toast
        gr.Warning(f"Fetch failed for {int(shot)}/{pointname}: {exc}")
        return None

    if x.size == 0:
        gr.Warning(f"{int(shot)}/{pointname} returned no samples in that window.")
        return None

    return signal_to_spectrogram(
        x,
        n_fft=transform_args.get("n_fft", DEFAULT_N_FFT),
        hop=transform_args.get("hop_length", DEFAULT_HOP),
        clip_dc=transform_args.get("clip_dc", DEFAULT_CLIP_DC),
        clip_low=transform_args.get("percentile_low", DEFAULT_CLIP_LOW),
        clip_high=transform_args.get("percentile_high", DEFAULT_CLIP_HIGH),
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

        ## Visualization (mirror of the Analyze-tab controls)
        with gr.Column():
            with gr.Row():
                view_mode = gr.Radio(
                    choices=["Original", "Enhanced", "Mask", "Amplitude"],
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

            analyze_btn = gr.Button("Analyze", variant="primary")
            visualize_out = gr.Image(label="Visualization", type="pil")

    # State variables (same shape as the Analyze tab)
    model = gr.State()
    signal_transform = gr.State()
    inference_output = gr.State()
    transform_args = gr.State(
        setup_stft_transform(
            DEFAULT_N_FFT,
            DEFAULT_HOP,
            DEFAULT_CLIP_DC,
            DEFAULT_CLIP_LOW,
            DEFAULT_CLIP_HIGH,
        )
    )

    # Event handling
    diagnostic.change(
        fn=_pointname_update, inputs=[diagnostic], outputs=[pointname]
    )

    load_model_btn.click(
        fn=wrapper_model_load, inputs=[model_file], outputs=[model]
    )

    setup_transform_btn.click(
        fn=setup_stft_transform,
        inputs=[n_fft, hop_length, clip_dc, clip_low_sld, clip_high_sld],
        outputs=[transform_args],
    )

    ## Load shot -> spectrogram
    load_shot_btn.click(
        fn=load_shot,
        inputs=[shot, diagnostic, pointname, t_min, t_max, transform_args],
        outputs=[signal_transform],
    ).then(
        fn=show_image,
        inputs=[
            gr.State("Original"),
            signal_transform,
            inference_output,
            gr.State(False),
            gr.State(False),
            vmin_sld,
            vmax_sld,
            threshold_sld,
        ],
        outputs=[extract_out],
    )

    view_mode.change(
        fn=toggle_view_groups,
        inputs=[view_mode],
        outputs=[enhanced_grp, mask_grp],
    )

    ## One-click Analyze: ensure a model, run inference, visualize
    analyze_btn.click(
        fn=ensure_model,
        inputs=[model, model_file, signal_transform],
        outputs=[model],
    ).then(
        fn=model_infer,
        inputs=[signal_transform, model],
        outputs=[inference_output],
    ).then(
        fn=show_image,
        inputs=[
            view_mode,
            signal_transform,
            inference_output,
            out_1_chk,
            out_2_chk,
            vmin_sld,
            vmax_sld,
            threshold_sld,
        ],
        outputs=[visualize_out],
    )
