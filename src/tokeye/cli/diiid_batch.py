"""``tokeye diiid-batch`` — offline multi-shot DIII-D analysis (cache-only).

Runs TokEye inference and/or the classic toroidal mode-spectrogram (optionally
gated by the TokEye mask) over many shots, reading prefetched signals from the
on-disk cache — so it works on a compute node with no atlas/internet. The
``DIII-D Offline`` web tab prefetches on somega and submits this as one Slurm job.

Per shot, into ``--outdir``:
  * ``<shot>_<probe>_mask.npy`` + ``<shot>_<probe>_preview.png``  (TokEye)
  * ``<shot>_modespec.png`` + ``<shot>_modes.csv``               (modespec)
  * ``<shot>_modespec_gated.png``                                (gated modespec)

Process exit code = number of shots that failed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

MAX_SHOTS = 1000  # guardrail against an accidentally huge range

# Cluster module incantation the Slurm job body uses.
MODULE_USE = "/cscratch/share/tokeye/modulefiles"
MODULE_NAME = "tokeye"


def parse_shots(text: str) -> list[int]:
    """Expand ``"150000-150010"`` (inclusive range) and/or ``"1,2,3"`` → int list.

    Deduplicates preserving order. Raises ``ValueError`` on garbage or an
    over-large expansion (> ``MAX_SHOTS``).
    """
    text = (text or "").strip()
    if not text:
        return []
    shots: list[int] = []
    for tok in text.replace(" ", "").split(","):
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            lo, hi = sorted((int(a), int(b)))
            if hi - lo + 1 > MAX_SHOTS:
                raise ValueError(f"range {tok} expands to > {MAX_SHOTS} shots")
            shots.extend(range(lo, hi + 1))
        else:
            shots.append(int(tok))
    seen: set[int] = set()
    out: list[int] = []
    for s in shots:
        if s not in seen:
            seen.add(s)
            out.append(s)
    if len(out) > MAX_SHOTS:
        raise ValueError(f"{len(out)} shots exceeds the {MAX_SHOTS} limit")
    return out


def build_sbatch_script(
    *,
    outdir: str,
    shots: list[int],
    diag: str,
    probe: str,
    tlim: tuple[float, float] | None,
    model: str,
    threshold: float,
    do_tokeye: bool,
    do_modespec: bool,
    do_gate: bool,
    n_range: tuple[int, int],
    f_min: float,
    f_max: float,
    partition: str,
    gres: str,
    time_limit: str,
    decimation: int = 1,
    device: str = "auto",
    job_name: str = "tokeye_offline",
) -> str:
    """Render the sbatch script that re-invokes ``tokeye diiid-batch`` on a node."""
    shots_csv = ",".join(str(s) for s in shots)
    header = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
    ]
    if gres.strip():
        header.append(f"#SBATCH --gres={gres.strip()}")
    header += [
        f"#SBATCH --time={time_limit}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --output={outdir}/%j.out",
    ]

    cmd = [
        "srun",
        "tokeye",
        "diiid-batch",
        f"--shots {shots_csv}",
        f"--outdir {outdir}",
        f"--diag {diag}",
        f"--probe {probe}",
        f"--model {model}",
        f"--threshold {threshold}",
        f"--n-range {n_range[0]} {n_range[1]}",
        f"--f-min {f_min}",
        f"--f-max {f_max}",
        f"--decimation {decimation}",
        f"--device {device}",
    ]
    if tlim is not None:
        cmd.append(f"--tlim {tlim[0]} {tlim[1]}")
    if do_tokeye:
        cmd.append("--tokeye")
    if do_modespec:
        cmd.append("--modespec")
    if do_gate:
        cmd.append("--gate")

    body = [
        "",
        f"module use {MODULE_USE} && module load {MODULE_NAME}",
        " ".join(cmd),
        "",
    ]
    return "\n".join(header + body)


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "diiid-batch",
        help="Offline multi-shot DIII-D analysis (TokEye + modespec) from cache.",
    )
    parser.add_argument("--shots", required=True, help="Comma list and/or a-b ranges.")
    parser.add_argument("--outdir", required=True, help="Results directory.")
    parser.add_argument("--diag", default="mag", help="Diagnostic preset (default mag).")
    parser.add_argument("--probe", default=None, help="Probe (default: diag default).")
    parser.add_argument(
        "--tlim", type=float, nargs=2, metavar=("T0", "T1"), default=None
    )
    parser.add_argument("--model", default=None, help="Model (default big_tf_unet).")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--array", default="toroidal")
    parser.add_argument("--n-range", type=int, nargs=2, metavar=("N0", "N1"),
                        default=[-5, 5], dest="n_range")
    parser.add_argument("--f-min", type=float, default=5.0, dest="f_min")
    parser.add_argument("--f-max", type=float, default=200.0, dest="f_max")
    parser.add_argument(
        "--decimation", type=int, default=1,
        help="Signal decimation before modespec (>= auto f-max-safe value).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokeye", action="store_true", help="Run TokEye masks.")
    parser.add_argument("--modespec", action="store_true", help="Run modespec.")
    parser.add_argument("--gate", action="store_true", help="Gated modespec (+TokEye).")
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:  # noqa: C901 - linear per-shot driver
    import csv as _csv

    import numpy as np

    from tokeye import batch
    from tokeye.modespec.classic.generate_modes import (
        CSV_COLUMNS,
        PARAM_DEFAULTS,
        detect_modes,
    )
    from tokeye.sources import DIAGNOSTICS, MDSSource
    from tokeye.sources.mirnov import gate_dominant, run_mode_spectrogram
    from tokeye.sources.viz import render_modespec
    from tokeye.transforms import (
        DEFAULT_CLIP_DC,
        DEFAULT_CLIP_HIGH,
        DEFAULT_CLIP_LOW,
        DEFAULT_HOP,
        DEFAULT_N_FFT,
    )

    try:
        shots = parse_shots(args.shots)
    except ValueError as exc:
        print(f"error: bad --shots: {exc}", file=sys.stderr)
        return 2
    if not shots:
        print("error: no shots given", file=sys.stderr)
        return 2

    diag = DIAGNOSTICS.get(args.diag)
    probe = args.probe or (diag.default if diag else None)
    do_gate = args.gate
    do_tokeye = args.tokeye or do_gate  # gating needs the mask
    do_modespec = args.modespec or do_gate
    model = args.model or "big_tf_unet"
    tlim = tuple(args.tlim) if args.tlim is not None else None
    n_range = (int(args.n_range[0]), int(args.n_range[1]))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    inputs_dir = outdir / "inputs"

    stft_kwargs = {
        "n_fft": DEFAULT_N_FFT,
        "hop": DEFAULT_HOP,
        "clip_dc": DEFAULT_CLIP_DC,
        "clip_low": DEFAULT_CLIP_LOW,
        "clip_high": DEFAULT_CLIP_HIGH,
    }

    # ── TokEye masks (one batch → one model load) ────────────────────────────────
    if do_tokeye:
        npys = [str(inputs_dir / f"{s}_{probe}.npy") for s in shots]
        existing = [p for p in npys if Path(p).is_file()]
        if not existing:
            print(f"error: no prefetched inputs in {inputs_dir}", file=sys.stderr)
            return 2
        batch.run_batch(
            existing,
            model=model,
            out_dir=outdir,
            stft_kwargs=stft_kwargs,
            save_png=True,
            threshold=args.threshold,
            device=args.device,
        )

    # ── Modespec (+ optional gated) per shot ─────────────────────────────────────
    failures = 0
    for shot in shots:
        try:
            if do_modespec:
                result = run_mode_spectrogram(
                    shot, args.array, tlim,
                    decimation=int(args.decimation) if args.decimation else None,
                    n_range=n_range, f_min_khz=args.f_min, f_max_khz=args.f_max,
                )
                img = render_modespec(result, coh_thresh=None, shot=shot)
                if img is not None:
                    img.save(outdir / f"{shot}_modespec.png")

                cfg = {**PARAM_DEFAULTS, "n_range": list(n_range)}
                rows = detect_modes(result, cfg)
                with (outdir / f"{shot}_modes.csv").open("w", newline="") as fh:
                    w = _csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
                    w.writeheader()
                    for ev in rows:
                        w.writerow({
                            "array": args.array,
                            "mode_label": "n",
                            "f_min_khz": args.f_min,
                            "f_max_khz": args.f_max,
                            **ev,
                        })

                if do_gate:
                    mask_path = outdir / f"{shot}_{probe}_mask.npy"
                    if not mask_path.is_file():
                        print(f"[{shot}] no mask for gating; skipping gated view",
                              file=sys.stderr)
                    else:
                        mask = np.load(mask_path)
                        t, _x, fs = MDSSource().fetch(shot, str(probe), tlim)
                        meta = {
                            "fs": float(fs),
                            "t0_ms": float(t[0]) if t.size else 0.0,
                            "n_fft": DEFAULT_N_FFT,
                            "hop": DEFAULT_HOP,
                            "clip_dc": DEFAULT_CLIP_DC,
                        }
                        nd = gate_dominant(
                            result, mask, meta, mask_threshold=args.threshold
                        )
                        gimg = render_modespec(
                            result, nd=nd, shot=shot,
                            title=f"Shot {shot} — toroidal n (TokEye-gated)",
                        )
                        if gimg is not None:
                            gimg.save(outdir / f"{shot}_modespec_gated.png")
            print(f"[{shot}] done")
        except Exception as exc:  # noqa: BLE001 - one bad shot must not kill the run
            print(f"[{shot}] ERROR: {exc}", file=sys.stderr)
            failures += 1

    print(f"diiid-batch: {len(shots) - failures}/{len(shots)} shot(s) ok -> {outdir}")
    return failures
