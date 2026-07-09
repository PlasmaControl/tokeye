"""DIII-D Modespec tab — the classic toroidal mode-number analysis, self-contained.

This mirrors how ``modespec`` lives in the source: its own thing. Enter a shot and
it fetches the **toroidal Mirnov array**, runs the classic mode-spectrogram, and
draws the dominant toroidal mode number ``n`` as an interactive Plotly heatmap
(discrete rainbow, integer colorbar, hover-to-read ``n``, zoom/pan).

The gate is **band-matched** — the fix for the old DIII-D-tab problem where the
gate came from a single, possibly-different-band probe. **Gate with TokEye** runs
the segmentation model on the *same* array and keeps only the modes it confirms,
using either the **average across all probes** (default; cancels single-probe
pickup/harmonic horizontal-line artifacts) or a chosen **reference probe**.

Everything MDSplus/model-touching is deferred to the callbacks, so building this
tab (and the app) does no I/O and no model load.
"""

from __future__ import annotations

import logging

import gradio as gr

from tokeye.app.analyze.analyze import wrapper_model_load
from tokeye.app.analyze.load import find_models, model_infer
from tokeye.hub import DEFAULT_MODEL, MODEL_REGISTRY
from tokeye.sources.presets import MIRNOV_TOROIDAL
from tokeye.transforms import (
    DEFAULT_CLIP_DC,
    DEFAULT_CLIP_HIGH,
    DEFAULT_CLIP_LOW,
    DEFAULT_HOP,
    DEFAULT_N_FFT,
)

logger = logging.getLogger(__name__)

_ARRAY = "toroidal"
_DEFAULT_REF = "MPI66M067D"
_GATE_AVERAGE = "Array average"
_GATE_REFERENCE = "Reference probe"
_MASK_THRESHOLD = 0.5  # TokEye coherent-channel cutoff for the gate mask

MODESPEC_INTRO_MD = """\
The classic **toroidal mode-number** analysis on the DIII-D Mirnov array, served \
interactively. Enter a **shot** (the time window auto-fills from the reference \
probe), set the frequency band + mode range, and **Analyze**.

* The plot is the dominant toroidal mode ``n`` vs time/frequency — a discrete \
rainbow with an integer colorbar; **hover** to read ``n``, scroll/drag to zoom + pan.
* **Gate with TokEye** keeps only the modes the segmentation model confirms, \
computed from the **same array** (band-matched). **Array average** cancels \
single-probe horizontal-line artifacts; **Reference probe** gates from one probe.
* The **coherence threshold** re-renders instantly from the cached result (no recompute).

Fetching needs a node that can reach `atlas.gat.com` (login / somega).
"""


def _stft_kwargs() -> dict:
    """STFT settings for the gate's per-probe spectrograms (the model's defaults)."""
    return {
        "n_fft": DEFAULT_N_FFT,
        "hop": DEFAULT_HOP,
        "clip_dc": DEFAULT_CLIP_DC,
        "clip_low": DEFAULT_CLIP_LOW,
        "clip_high": DEFAULT_CLIP_HIGH,
    }


def _tlim(t_min, t_max) -> tuple[float, float] | None:
    if t_min is not None and t_max is not None and float(t_max) > float(t_min):
        return (float(t_min), float(t_max))
    return None


def _ensure_model(model, model_file):
    """Load the model on first gated Analyze; pass an already-loaded model through."""
    if model is not None:
        return model
    return wrapper_model_load(model_file)


def fill_window(shot, ref_probe):
    """Auto-fill (t_min, t_max) from the reference probe's data window on change.

    Cheap scalar-TDI bounds query (only endpoints cross the wire). Best effort:
    leaves the fields as-is if unreadable (atlas unreachable, off-cluster), never
    clobbering a manual entry.
    """
    if not shot or not ref_probe:
        return gr.update(), gr.update()
    from tokeye.sources import MDSSource

    bounds = MDSSource.time_bounds(int(shot), str(ref_probe))
    if not bounds:
        return gr.update(), gr.update()
    return round(float(bounds[0]), 2), round(float(bounds[1]), 2)


def run_modespec(
    shot,
    ref_probe,
    t_min,
    t_max,
    model,
    model_file,
    f_min,
    f_max,
    n_min,
    n_max,
    decimation,
    coh_thresh,
    gate,
    gate_source,
    progress=gr.Progress(),
):
    """Analyze: fetch array → mode-spectrogram → (optional band-matched gate) → figure.

    Returns ``[model, result, tok_mask, gate_meta, figure]`` — ``result``/``tok_mask``/
    ``gate_meta`` are cached in state so the coherence slider re-renders without
    recompute (see :func:`rerender_coh`).
    """
    if not shot:
        gr.Warning("Enter a shot number first.")
        return model, None, None, None, None

    from tokeye.sources.mirnov import (
        array_gate_mask,
        gate_dominant_mask,
        run_mode_spectrogram,
    )
    from tokeye.sources.viz import plotly_modespec

    tlim = _tlim(t_min, t_max)
    dec = int(decimation) if decimation else None
    progress(0.1, desc="Fetching Mirnov array + computing mode spectrogram …")
    try:
        result = run_mode_spectrogram(
            int(shot),
            _ARRAY,
            tlim,
            decimation=dec,
            n_range=(int(n_min), int(n_max)),
            f_min_khz=float(f_min),
            f_max_khz=float(f_max),
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure as a toast
        gr.Warning(f"Modespec failed for shot {int(shot)}: {exc}")
        return model, None, None, None, None

    tok_mask = None
    gate_meta = None
    if gate:
        model = _ensure_model(model, model_file)
        if model is None:
            gr.Warning("Gate needs a model (load failed); showing ungated.")
        else:
            source = "reference" if gate_source == _GATE_REFERENCE else "average"
            gr.Info("Running TokEye on the Mirnov array to gate the modes …")
            try:
                tok_mask, gate_meta = array_gate_mask(
                    int(shot),
                    _ARRAY,
                    tlim,
                    _stft_kwargs(),
                    lambda s: model_infer(s, model),
                    threshold=_MASK_THRESHOLD,
                    decimation=dec,
                    source=source,
                    reference=str(ref_probe) if ref_probe else None,
                    f_max_khz=float(f_max),
                    data_dir=None,
                    on_progress=lambda frac, desc: progress(0.3 + 0.5 * frac, desc=desc),
                )
            except Exception as exc:  # noqa: BLE001
                gr.Warning(f"Array gate failed (showing ungated): {exc}")
                tok_mask, gate_meta = None, None

    nd = None
    if tok_mask is not None and gate_meta is not None:
        try:
            nd = gate_dominant_mask(result, tok_mask, gate_meta, coh_thresh=float(coh_thresh))
        except Exception as exc:  # noqa: BLE001
            gr.Warning(f"Gating failed (showing ungated): {exc}")
            nd = None

    progress(0.95, desc="Rendering modes …")
    fig = plotly_modespec(result, nd=nd, coh_thresh=float(coh_thresh))
    return model, result, tok_mask, gate_meta, fig


def rerender_coh(result, tok_mask, gate_meta, coh_thresh, gate):
    """Cheap re-render on coherence-slider release — cached result/mask, no recompute."""
    if result is None:
        return None
    from tokeye.sources.mirnov import gate_dominant_mask
    from tokeye.sources.viz import plotly_modespec

    nd = None
    if gate and tok_mask is not None and gate_meta is not None:
        try:
            nd = gate_dominant_mask(result, tok_mask, gate_meta, coh_thresh=float(coh_thresh))
        except Exception:  # noqa: BLE001
            nd = None
    return plotly_modespec(result, nd=nd, coh_thresh=float(coh_thresh))


def diiid_modespec_tab():
    with gr.Column():
        with gr.Accordion("What this tab does", open=False):
            gr.Markdown(MODESPEC_INTRO_MD)

        ## Model
        with gr.Group():
            model_file = gr.Dropdown(
                label="Analysis Model",
                info=(
                    "Used only for the TokEye gate. Built-in models download from "
                    "Hugging Face on first gated Analyze (~30 MB, cached)."
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
                ref_probe = gr.Dropdown(
                    label="Reference probe (window autofill + gate reference)",
                    choices=list(MIRNOV_TOROIDAL),
                    value=_DEFAULT_REF,
                    interactive=True,
                    allow_custom_value=True,
                )
            with gr.Row():
                t_min = gr.Number(label="t min (ms)", value=None)
                t_max = gr.Number(label="t max (ms)", value=None)

        ## Modespec settings
        with gr.Group():
            with gr.Row():
                f_min = gr.Number(value=5.0, label="f min (kHz)")
                f_max = gr.Number(value=200.0, label="f max (kHz)")
                decimation = gr.Number(
                    value=1,
                    precision=0,
                    minimum=1,
                    label="Decimation (1 = off)",
                    info="Speeds up modespec; ≥ auto f-max-safe value.",
                )
            with gr.Row():
                n_min = gr.Number(value=-5, precision=0, label="n min")
                n_max = gr.Number(value=5, precision=0, label="n max")

        ## Gate settings
        with gr.Group():
            with gr.Row():
                gate = gr.Checkbox(value=True, label="Gate with TokEye")
                gate_source = gr.Radio(
                    choices=[_GATE_AVERAGE, _GATE_REFERENCE],
                    value=_GATE_AVERAGE,
                    label="Gate source",
                    info="Average cancels single-probe horizontal-line artifacts.",
                )
            coh_thresh = gr.Slider(
                0, 1, value=0.5, step=0.01, label="Coherence threshold"
            )

        analyze_btn = gr.Button("Analyze", variant="primary")
        modespec_out = gr.Plot(label="Toroidal mode number n")

    # State (cached so the coherence slider re-renders without recompute)
    model = gr.State()
    result_state = gr.State()
    tok_mask_state = gr.State()
    gate_meta_state = gr.State()

    # Auto-fill the time window from the reference probe on shot/probe change.
    shot.change(fn=fill_window, inputs=[shot, ref_probe], outputs=[t_min, t_max])
    ref_probe.change(fn=fill_window, inputs=[shot, ref_probe], outputs=[t_min, t_max])

    load_model_btn.click(fn=wrapper_model_load, inputs=[model_file], outputs=[model])

    analyze_btn.click(
        fn=run_modespec,
        inputs=[
            shot, ref_probe, t_min, t_max, model, model_file,
            f_min, f_max, n_min, n_max, decimation, coh_thresh, gate, gate_source,
        ],
        outputs=[model, result_state, tok_mask_state, gate_meta_state, modespec_out],
    )

    # Coherence slider re-renders instantly from the cached result + gate mask.
    coh_thresh.release(
        fn=rerender_coh,
        inputs=[result_state, tok_mask_state, gate_meta_state, coh_thresh, gate],
        outputs=[modespec_out],
    )

    return shot
