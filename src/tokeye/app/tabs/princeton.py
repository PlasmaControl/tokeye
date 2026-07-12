"""Princeton tab — load a shot from the local foundation_model archive.

The stellar twin of the DIII-D tab: enter a shot, pick a signal group +
channel (the time window auto-fills to the data range), **Load shot** to see
the spectrogram, then **Analyze** to overlay the segmentation mask. Data comes
straight from ``/scratch/gpfs/EKOLEMEN/foundation_model`` (``$TOKEYE_FOUNDATION_DIR``)
— no network fetch, no cache needed.

The view/render/model plumbing is shared with the DIII-D tab
(:mod:`tokeye.app.tabs.diiid` — importing it does no I/O); only the
source-facing handlers differ. Pointnames are ``group/index`` because the
archive does not record channel identity, and the probe dropdown refreshes
from the selected shot's file so it lists exactly what that shot carries.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import gradio as gr

from tokeye import export
from tokeye.app.analyze.analyze import wrapper_model_load_pair
from tokeye.app.analyze.load import find_models
from tokeye.app.tabs.diiid import (
    _VIEW_MODES,
    _tlim,
    _toggle_view_groups,
    render_spectrogram,
    rerender,
    rerender_band,
    run_analyze,
)
from tokeye.hub import DEFAULT_MODEL, MODEL_REGISTRY
from tokeye.inference import signal_to_spectrogram
from tokeye.sources.foundation import pointname_slug
from tokeye.sources.foundation_presets import (
    FOUNDATION_DIAGNOSTICS,
    foundation_dropdown_choices,
    signals_for_shot,
)
from tokeye.transforms import (
    DEFAULT_CLIP_DC,
    DEFAULT_CLIP_HIGH,
    DEFAULT_CLIP_LOW,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
)

logger = logging.getLogger(__name__)

_DEFAULT_DIAG = "mirnov"
_DEFAULT_FMIN_KHZ = 5.0
_DEFAULT_FMAX_KHZ = 250.0

PRINCETON_INTRO_MD = """\
Load a **DIII-D shot from the local foundation_model archive** \
(`/scratch/gpfs/EKOLEMEN/foundation_model`, ~17k shots) and see its \
spectrogram with TokEye's mode mask overlaid. Everything is read from GPFS — \
no MDSplus, no network.

1. Enter a **shot** (defaults to the newest in the archive), pick a **signal \
group** + **channel** — the channel list refreshes to what that shot's file \
actually carries, and **t-min/t-max** auto-fill to the signal's data window \
(narrow them to go faster).
2. **Load shot** reads the channel and shows the spectrogram — an interactive \
Plotly plot with real kHz/ms axes (zoom + pan). Use **STFT settings → \
f-min/f-max** to crop the band you care about.
3. **Analyze** overlays the coherent/transient mode mask. Threshold / clip / \
band sliders re-render live — no recompute.

Pointnames are `group/index`: the archive stores channels as rows in \
sorted-original-name order but does not record which probe each row is, so \
probe-identity analyses (mode numbers / Modespec) are not available here.
"""


def _pointname_update(diag_key: str, shot) -> gr.Dropdown:
    """Repopulate the channel dropdown for the diagnostic — per-shot when possible.

    With a shot entered, the choices come from that shot's file (channel counts
    vary and some groups are absent); otherwise, or if the file is unreadable,
    the static presets are used.
    """
    diags = signals_for_shot(int(shot)) if shot else FOUNDATION_DIAGNOSTICS
    diag = diags.get(diag_key) or FOUNDATION_DIAGNOSTICS.get(diag_key)
    if diag is None:
        return gr.Dropdown(choices=[], value=None)
    return gr.Dropdown(choices=list(diag.pointnames), value=diag.default)


def fill_window(shot, pointname):
    """Auto-fill (t_min, t_max) with the signal's data window on shot/probe change.

    Reads only the time base's two endpoint samples from the shot file. Best
    effort: leaves the fields as-is when the shot/probe is empty or unreadable,
    so it never clobbers a manual entry.
    """
    if not shot or not pointname:
        return gr.update(), gr.update()
    from tokeye.sources.foundation import FoundationSource

    bounds = FoundationSource.time_bounds(int(shot), str(pointname))
    if not bounds:
        return gr.update(), gr.update()
    return round(float(bounds[0]), 2), round(float(bounds[1]), 2)


def load_shot(
    shot,
    diag_key,
    pointname,
    t_min,
    t_max,
    n_fft,
    hop_length,
    clip_dc,
    clip_low,
    clip_high,
    decimation,
    progress=gr.Progress(),
):
    """Read one archive channel → ``(spectrogram, stft_meta)`` for the states.

    Same contract as the DIII-D tab's loader (STFT knobs read live, optional
    anti-aliased decimation, real ``fs``/``t0_ms`` recorded in ``stft_meta``);
    only the source differs. Degrades to ``(None, None)`` + a ``gr.Warning``
    on a missing shot/group/channel or an empty window.
    """
    if not shot or not pointname:
        gr.Warning("Enter a shot number and pick a signal group/channel first.")
        return None, None

    from tokeye.sources.foundation import FoundationSource

    tlim = _tlim(t_min, t_max)
    progress(0.1, desc=f"Reading {int(shot)}/{pointname} …")
    try:
        t, x, fs = FoundationSource().fetch(int(shot), str(pointname), tlim)
    except Exception as exc:  # noqa: BLE001 - surface any read failure as a toast
        gr.Warning(f"Read failed for {int(shot)}/{pointname}: {exc}")
        return None, None

    if x.size == 0:
        gr.Warning(
            f"{int(shot)}/{pointname} has no samples"
            + (" in that window." if tlim else " (signal absent for this shot).")
        )
        return None, None

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

    progress(0.7, desc="Computing spectrogram …")
    spec = signal_to_spectrogram(
        x,
        n_fft=n_fft,
        hop=hop_length,
        clip_dc=clip_dc,
        clip_low=clip_low,
        clip_high=clip_high,
    )
    stft_meta = {
        "fs": float(fs),
        "t0_ms": float(t[0]) if t.size else 0.0,
        "n_fft": int(n_fft),
        "hop": int(hop_length),
        "clip_dc": bool(clip_dc),
    }
    progress(1.0, desc="Done")
    return spec, stft_meta


def export_princeton_analysis(
    shot,
    pointname,
    model_file,
    signal_transform,
    stft_meta,
    inference_output,
    n_fft,
    hop_length,
    clip_dc,
    clip_low,
    clip_high,
    decimation,
    stft_fmin,
    stft_fmax,
    threshold,
    view_mode,
):
    """Save the loaded spectrogram (+ mask, if inferred) as a ``.npz`` bundle.

    Mirrors ``export_diiid_analysis`` (real ``time_ms``/``freq_khz`` axes from
    ``stft_meta``; ``None`` on no data so the download slot clears), with
    ``source="princeton"`` and a slugged stem — ``group/index`` pointnames
    contain a ``/`` that must not reach the filename.
    """
    if signal_transform is None:
        gr.Warning("Load a shot first.")
        return None

    params = {
        "shot": int(shot) if shot else None,
        "pointname": pointname,
        "model": model_file,
        "n_fft": n_fft,
        "hop": hop_length,
        "clip_dc": clip_dc,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "decimation": decimation,
        "fmin_khz": stft_fmin,
        "fmax_khz": stft_fmax,
        "threshold": threshold,
        "view_mode": view_mode,
    }
    bundle = export.analysis_bundle(
        spectrogram=signal_transform,
        mask=inference_output,
        stft_meta=stft_meta,
        params=params,
        source="princeton",
    )
    if shot and pointname:
        stem = f"{int(shot)}_{pointname_slug(pointname)}_analysis"
    else:
        stem = export.default_stem("analysis")
    out_dir = Path(tempfile.mkdtemp(prefix="tokeye-export-"))
    path = export.save_npz(out_dir / f"{stem}.npz", bundle)
    return str(path)


def princeton_tab():
    # User Interface
    with gr.Column():
        with gr.Accordion("What this tab does", open=False):
            gr.Markdown(PRINCETON_INTRO_MD)

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
                    label="Signal group",
                    choices=foundation_dropdown_choices(),
                    value=_DEFAULT_DIAG,
                    interactive=True,
                )
                pointname = gr.Dropdown(
                    label="Channel (group/index)",
                    choices=list(FOUNDATION_DIAGNOSTICS[_DEFAULT_DIAG].pointnames),
                    value=FOUNDATION_DIAGNOSTICS[_DEFAULT_DIAG].default,
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
                    info="Applied on Load.",
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

        ## Load shot
        load_shot_btn = gr.Button("Load shot", variant="primary")

        ## Spectrogram (pre-analysis)
        extract_out = gr.Plot(label="Spectrogram")

        ## Visualization (mirror of the Analyze-tab controls)
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

            analyze_btn = gr.Button("Analyze", variant="primary")
            visualize_out = gr.Plot(label="Visualization")

            save_export_btn = gr.Button("Save results (.npz)")
            export_out = gr.File(
                label="Download analysis (.npz)", interactive=False
            )

    # State variables (same shape as the DIII-D tab)
    model = gr.State()
    model_name = gr.State(None)
    signal_transform = gr.State()
    stft_meta = gr.State()
    inference_output = gr.State()

    # Shared input bundles
    rerender_inputs = [
        view_mode,
        signal_transform,
        stft_meta,
        inference_output,
        out_1_chk,
        out_2_chk,
        vmin_sld,
        vmax_sld,
        threshold_sld,
        stft_fmin,
        stft_fmax,
    ]
    spectrogram_inputs = [signal_transform, stft_meta, stft_fmin, stft_fmax]

    # Event handling: the channel list depends on BOTH the diagnostic and the
    # shot (availability + channel counts vary per shot file).
    diagnostic.change(
        fn=_pointname_update, inputs=[diagnostic, shot], outputs=[pointname]
    )
    shot.change(
        fn=_pointname_update, inputs=[diagnostic, shot], outputs=[pointname]
    ).then(fn=fill_window, inputs=[shot, pointname], outputs=[t_min, t_max])
    pointname.change(fn=fill_window, inputs=[shot, pointname], outputs=[t_min, t_max])

    load_model_btn.click(
        fn=wrapper_model_load_pair,
        inputs=[model_file],
        outputs=[model, model_name],
    )

    ## Load shot -> spectrogram (Plotly image, cropped to the display band).
    load_shot_btn.click(
        fn=load_shot,
        inputs=[
            shot,
            diagnostic,
            pointname,
            t_min,
            t_max,
            n_fft,
            hop_length,
            clip_dc,
            clip_low_sld,
            clip_high_sld,
            decimation,
        ],
        outputs=[signal_transform, stft_meta],
    ).then(
        fn=render_spectrogram, inputs=spectrogram_inputs, outputs=[extract_out]
    )

    ## Switching view mode toggles its controls and re-renders from cache
    view_mode.change(
        fn=_toggle_view_groups,
        inputs=[view_mode],
        outputs=[enhanced_grp, mask_grp],
    ).then(fn=rerender, inputs=rerender_inputs, outputs=[visualize_out])

    ## One-click Analyze: run the model + render the chosen view
    analyze_btn.click(
        fn=run_analyze,
        inputs=[
            view_mode,
            model,
            model_name,
            model_file,
            signal_transform,
            stft_meta,
            inference_output,
            out_1_chk,
            out_2_chk,
            vmin_sld,
            vmax_sld,
            threshold_sld,
            stft_fmin,
            stft_fmax,
        ],
        outputs=[model, model_name, inference_output, visualize_out],
    )

    ## Live re-render on slider release (re-color only, no recompute)
    for sld in (vmin_sld, vmax_sld, threshold_sld):
        sld.release(fn=rerender, inputs=rerender_inputs, outputs=[visualize_out])

    ## Frequency band changes re-crop both plots on Enter / focus-leave
    for band in (stft_fmin, stft_fmax):
        band.submit(
            fn=rerender_band,
            inputs=rerender_inputs,
            outputs=[extract_out, visualize_out],
        )
        band.blur(
            fn=rerender_band,
            inputs=rerender_inputs,
            outputs=[extract_out, visualize_out],
        )

    ## Export the loaded spectrogram (+ mask, if inferred) as an .npz bundle.
    save_export_btn.click(
        fn=export_princeton_analysis,
        inputs=[
            shot,
            pointname,
            model_file,
            signal_transform,
            stft_meta,
            inference_output,
            n_fft,
            hop_length,
            clip_dc,
            clip_low_sld,
            clip_high_sld,
            decimation,
            stft_fmin,
            stft_fmax,
            threshold_sld,
            view_mode,
        ],
        outputs=[export_out],
    )

    return shot
