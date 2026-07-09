"""DIII-D Offline tab — batch many shots on the cluster, then view the results.

Enter a shot **range** (``150000-150010``) or **list** (``150000,150002``), pick
what to run (TokEye masks / modespec / TokEye-gated modespec), and Submit. Because
compute nodes cannot reach atlas, the app **prefetches every shot here on somega**
(warming the on-disk cache + the model), then submits ONE Slurm job that runs
``tokeye diiid-batch`` over the cached data. Refresh to see job status and the
gallery of per-shot result images.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import gradio as gr

from tokeye.cli.diiid_batch import build_sbatch_script, parse_shots
from tokeye.hub import DEFAULT_MODEL, MODEL_REGISTRY
from tokeye.sources import DIAGNOSTICS, diagnostic_dropdown_choices

logger = logging.getLogger(__name__)

_DEFAULT_DIAG = "mag"

OFFLINE_INTRO_MD = """\
Analyze **many shots** at once. The app prefetches each shot on somega (which can \
reach `atlas.gat.com`), then submits a single Slurm job on the `gpus` partition to \
run TokEye + modespec over the cached data. Results land in your output folder; \
**Refresh** shows job status and the image gallery when it finishes.
"""


def _default_outdir() -> str:
    user = os.environ.get("USER", "user")
    return f"/cscratch/{user}/tokeye/data/runs"


def _tlim(t_min, t_max):
    if t_min is not None and t_max is not None and float(t_max) > float(t_min):
        return (float(t_min), float(t_max))
    return None


def _pointname_update(diag_key: str) -> gr.Dropdown:
    diag = DIAGNOSTICS.get(diag_key)
    if diag is None:
        return gr.Dropdown(choices=[], value=None)
    return gr.Dropdown(choices=list(diag.pointnames), value=diag.default)


def submit_batch(
    shots_text,
    diag_key,
    pointname,
    t_min,
    t_max,
    model_file,
    do_tokeye,
    do_modespec,
    do_gate,
    gate_source,
    ms_nmin,
    ms_nmax,
    ms_fmin,
    ms_fmax,
    decimation,
    outdir_base,
    partition,
    gres,
    time_limit,
    progress=gr.Progress(),
):
    """Prefetch on somega, write the sbatch script, submit → (status, job_id, outdir)."""
    try:
        shots = parse_shots(shots_text)
    except ValueError as exc:
        gr.Warning(f"Bad shot list: {exc}")
        return f"❌ Bad shot list: {exc}", None, None
    if not shots:
        gr.Warning("Enter a shot range (a-b) or comma-separated list.")
        return "❌ No shots given.", None, None
    if not (do_tokeye or do_modespec or do_gate):
        gr.Warning("Pick at least one analysis.")
        return "❌ Pick at least one analysis.", None, None

    diag = DIAGNOSTICS.get(diag_key)
    probe = pointname or (diag.default if diag else None)
    model = model_file or DEFAULT_MODEL
    tlim = _tlim(t_min, t_max)
    need_probe = do_tokeye or do_gate
    need_array = do_modespec or do_gate

    tag = f"{shots[0]}-{shots[-1]}-{diag_key}"
    outdir = Path(outdir_base) / tag
    inputs_dir = outdir / "inputs"
    try:
        inputs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        gr.Warning(f"Cannot create {outdir}: {exc}")
        return f"❌ Cannot create {outdir}: {exc}", None, None

    # ── Prefetch on somega (has atlas); the Slurm job runs cache-only ────────────
    import numpy as np

    from tokeye.sources import MDSSource

    src = MDSSource()
    prefetched = 0
    for i, shot in enumerate(shots):
        progress((i + 1) / (len(shots) + 1), desc=f"Prefetch {shot}")
        try:
            if need_probe:
                _t, x, _fs = src.fetch(int(shot), str(probe), tlim)
                if x.size:
                    np.save(inputs_dir / f"{shot}_{probe}.npy", x)
            if need_array:
                from tokeye.sources.mirnov import fetch_mirnov_cached

                fetch_mirnov_cached(int(shot), "toroidal", tlim)
            prefetched += 1
        except Exception as exc:  # noqa: BLE001 - skip a bad shot, keep going
            logger.warning("prefetch failed for %s: %s", shot, exc)

    if prefetched == 0:
        gr.Warning("Prefetch fetched nothing — is atlas reachable from this node?")
        return "❌ Prefetch fetched no shots (atlas unreachable?).", None, None

    # Warm the model cache here (compute nodes have no internet to download it).
    progress(1.0, desc="Caching model")
    try:
        from tokeye.app.analyze.load import model_load

        model_load(model)
    except Exception as exc:  # noqa: BLE001
        gr.Warning(f"Could not pre-cache model '{model}': {exc}")

    # ── Write + submit the Slurm job ─────────────────────────────────────────────
    script = build_sbatch_script(
        outdir=str(outdir),
        shots=shots,
        diag=diag_key,
        probe=str(probe),
        tlim=tlim,
        model=model,
        threshold=0.5,
        do_tokeye=do_tokeye,
        do_modespec=do_modespec,
        do_gate=do_gate,
        n_range=(int(ms_nmin), int(ms_nmax)),
        f_min=float(ms_fmin),
        f_max=float(ms_fmax),
        decimation=int(decimation) if decimation else 1,
        gate_source=gate_source,
        reference_probe=str(probe),
        partition=partition,
        gres=gres,
        time_limit=time_limit,
    )
    script_path = outdir / "submit.sh"
    script_path.write_text(script)

    try:
        res = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        gr.Warning("sbatch not found — run the app on a Slurm submit node (somega).")
        return f"⚠️ Prefetched {prefetched} shot(s), but `sbatch` is unavailable.\n\nScript: {script_path}", None, str(outdir)
    except subprocess.CalledProcessError as exc:
        gr.Warning(f"sbatch failed: {exc.stderr.strip()}")
        return f"❌ sbatch failed: {exc.stderr.strip()}", None, str(outdir)

    job_id = res.stdout.strip().split(";")[0]
    (outdir / "job.txt").write_text(f"{job_id}\n{outdir}\n")
    status = (
        f"✅ Submitted Slurm job **{job_id}** ({prefetched}/{len(shots)} shots "
        f"prefetched).\n\nResults → `{outdir}`\n\nClick **Refresh** for status."
    )
    return status, job_id, str(outdir)


def refresh_results(job_id, outdir):
    """Poll Slurm state + gather the per-shot result images from ``outdir``."""
    if not outdir:
        return "No job submitted yet.", []

    state = "unknown"
    if job_id:
        try:
            q = subprocess.run(
                ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
                capture_output=True, text=True,
            )
            state = q.stdout.strip()
            if not state:  # finished → sacct
                a = subprocess.run(
                    ["sacct", "-j", str(job_id), "--format=State", "-n", "-P", "-X"],
                    capture_output=True, text=True,
                )
                state = a.stdout.strip().splitlines()[0].strip() if a.stdout.strip() else "COMPLETED?"
        except FileNotFoundError:
            state = "squeue unavailable"

    out = Path(outdir)
    images = sorted(
        str(p)
        for p in [*out.glob("*_preview.png"), *out.glob("*_modespec*.png")]
    )

    tail = ""
    logs = sorted(out.glob(f"{job_id}*.out")) if job_id else []
    if logs:
        lines = Path(logs[-1]).read_text(errors="replace").splitlines()
        tail = "\n".join(lines[-20:])

    status = f"**Job {job_id}** — state: `{state}`\n\n{len(images)} result image(s)."
    if tail:
        status += f"\n\n```\n{tail}\n```"
    return status, images


def diiid_offline_tab():
    with gr.Column():
        with gr.Accordion("What this tab does", open=False):
            gr.Markdown(OFFLINE_INTRO_MD)

        with gr.Group():
            shots_text = gr.Textbox(
                label="Shots",
                placeholder="150000-150010   or   150000,150002,150005",
                info="Inclusive range (a-b) and/or comma-separated list.",
            )
            with gr.Row():
                diagnostic = gr.Dropdown(
                    label="Diagnostic",
                    choices=diagnostic_dropdown_choices(),
                    value=_DEFAULT_DIAG,
                )
                pointname = gr.Dropdown(
                    label="Probe / pointname",
                    choices=list(DIAGNOSTICS[_DEFAULT_DIAG].pointnames),
                    value=DIAGNOSTICS[_DEFAULT_DIAG].default,
                    allow_custom_value=True,
                )
                model_file = gr.Dropdown(
                    label="Model",
                    choices=list(MODEL_REGISTRY),
                    value=DEFAULT_MODEL,
                    allow_custom_value=True,
                )
            with gr.Row():
                t_min = gr.Number(label="t min (ms)", value=None)
                t_max = gr.Number(label="t max (ms)", value=None)

        with gr.Group():
            with gr.Row():
                do_tokeye = gr.Checkbox(value=True, label="TokEye mask")
                do_modespec = gr.Checkbox(value=True, label="Modespec")
                do_gate = gr.Checkbox(value=True, label="TokEye-gated modespec")
            gate_source = gr.Radio(
                choices=["average", "reference"],
                value="average",
                label="Gate source",
                info="Band-matched gate: average over the array (cancels single-probe "
                "artifacts) or one reference probe (the probe above).",
            )
            with gr.Row():
                ms_nmin = gr.Number(value=-5, precision=0, label="n min")
                ms_nmax = gr.Number(value=5, precision=0, label="n max")
                ms_fmin = gr.Number(value=5, label="f min (kHz)")
                ms_fmax = gr.Number(value=200, label="f max (kHz)")
                ms_decim = gr.Number(
                    value=1, precision=0, minimum=1, label="Decimation",
                    info="≥ auto f-max-safe value; speeds up modespec.",
                )

        with gr.Accordion("Cluster / output", open=False), gr.Group():
            outdir_base = gr.Textbox(label="Output folder", value=_default_outdir())
            with gr.Row():
                partition = gr.Textbox(label="Partition", value="gpus")
                gres = gr.Textbox(label="GRES (blank = CPU)", value="gpu:v100:1")
                time_limit = gr.Textbox(label="Time limit", value="0-02:00:00")

        submit_btn = gr.Button("Prefetch + Submit job", variant="primary")
        status_md = gr.Markdown()
        with gr.Row():
            refresh_btn = gr.Button("Refresh status / results")
        gallery = gr.Gallery(label="Results", columns=3, height="auto")

    job_id = gr.State()
    outdir_state = gr.State()

    diagnostic.change(fn=_pointname_update, inputs=[diagnostic], outputs=[pointname])

    submit_btn.click(
        fn=submit_batch,
        inputs=[
            shots_text, diagnostic, pointname, t_min, t_max, model_file,
            do_tokeye, do_modespec, do_gate, gate_source,
            ms_nmin, ms_nmax, ms_fmin, ms_fmax, ms_decim,
            outdir_base, partition, gres, time_limit,
        ],
        outputs=[status_md, job_id, outdir_state],
    )

    refresh_btn.click(
        fn=refresh_results,
        inputs=[job_id, outdir_state],
        outputs=[status_md, gallery],
    )
