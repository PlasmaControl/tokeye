"""``tokeye princeton-batch`` — multi-shot TokEye masks from the foundation archive.

Reads channels straight from ``/scratch/gpfs/EKOLEMEN/foundation_model``
(``$TOKEYE_FOUNDATION_DIR``) — no prefetch step, the data is already on GPFS.
By default it renders a Slurm sbatch script into ``--outdir`` and submits it to
stellar's A100 ``gpu`` partition; ``--local`` runs the same work in-process
(e.g. on a stellar-vis node's V100S), and ``--dry-run`` prints the script
without submitting.

Per shot, into ``--outdir``:
  * ``inputs/<shot>_<group>-<index>.npy``   (the extracted channel)
  * ``<shot>_<group>-<index>_mask.npy`` + ``..._preview.png``  (TokEye)

No mode-number/modespec flags: the archive stores channels by row index
without probe identity, so the toroidal mode analysis is not available here.

Process exit code = number of shots that failed (``--local`` / job body).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tokeye.cli.diiid_batch import parse_shots

if TYPE_CHECKING:
    import argparse

# Cluster module incantation the Slurm job body uses. The modulefiles dir is
# the deploy default; ``TOKEYE_MODULE_DIR`` (set by the modulefile) overrides.
MODULE_USE = "/projects/EKOLEMEN/Modules/modulefiles-shared"
MODULE_NAME = "tokeye"

# Slurm defaults for stellar's A100 batch partition; the deploy modulefile
# sets the TOKEYE_SLURM_* overrides.
DEFAULT_PARTITION = "gpu"
DEFAULT_GRES = "gpu:a100:1"
DEFAULT_TIME = "0-02:00:00"


def _module_use() -> str:
    return os.environ.get("TOKEYE_MODULE_DIR", MODULE_USE)


def _default_partition() -> str:
    return os.environ.get("TOKEYE_SLURM_PARTITION", DEFAULT_PARTITION)


def _default_gres() -> str:
    return os.environ.get("TOKEYE_SLURM_GRES", DEFAULT_GRES)


def _default_time() -> str:
    return os.environ.get("TOKEYE_SLURM_TIME", DEFAULT_TIME)


def build_sbatch_script(
    *,
    outdir: str,
    shots: list[int],
    probe: str,
    tlim: tuple[float, float] | None,
    model: str,
    threshold: float,
    partition: str,
    gres: str,
    time_limit: str,
    decimation: int = 1,
    device: str = "auto",
    job_name: str = "tokeye_princeton",
) -> str:
    """Render the sbatch script that re-invokes ``tokeye princeton-batch --local``."""
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
        "princeton-batch",
        "--local",
        f"--shots {shots_csv}",
        f"--outdir {outdir}",
        f"--probe {probe}",
        f"--model {model}",
        f"--threshold {threshold}",
        f"--decimation {decimation}",
        f"--device {device}",
    ]
    if tlim is not None:
        cmd.append(f"--tlim {tlim[0]} {tlim[1]}")

    body = [
        "",
        # Compute nodes have no internet: use the module's prefetched HF cache
        # (deploy/princeton/setup.sh downloads the weights). Overridable so a
        # user with their own cache can opt out.
        "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}",
        f"module use {_module_use()} && module load {MODULE_NAME}",
        " ".join(cmd),
        "",
    ]
    return "\n".join(header + body)


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "princeton-batch",
        help="Multi-shot TokEye masks from the foundation_model archive (Slurm).",
        description=(
            "Run TokEye segmentation over many shots read from the local "
            "foundation_model archive. Default: render + submit an sbatch job "
            "on the A100 'gpu' partition; --local runs in-process (vis-node "
            "GPU). No modespec/mode-number options: the archive does not "
            "record probe identity."
        ),
    )
    parser.add_argument("--shots", required=True, help="Comma list and/or a-b ranges.")
    parser.add_argument("--outdir", required=True, help="Results directory.")
    parser.add_argument(
        "--diag", default="mirnov", help="Signal group preset (default mirnov)."
    )
    parser.add_argument(
        "--probe", default=None,
        help="Channel as group/index, e.g. mirnov/07 (default: the group's default).",
    )
    parser.add_argument(
        "--tlim", type=float, nargs=2, metavar=("T0", "T1"), default=None,
        help="Time window in ms.",
    )
    parser.add_argument("--model", default=None, help="Model (default big_tf_unet).")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--decimation", type=int, default=1,
        help="Anti-aliased decimation before the STFT (1 = off).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--local", action="store_true",
        help="Run now in this process instead of submitting a Slurm job.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Print the sbatch script and exit without submitting.",
    )
    parser.add_argument(
        "--partition", default=None,
        help=f"Slurm partition (default $TOKEYE_SLURM_PARTITION or {DEFAULT_PARTITION}).",
    )
    parser.add_argument(
        "--gres", default=None,
        help=f"Slurm --gres (default $TOKEYE_SLURM_GRES or {DEFAULT_GRES}).",
    )
    parser.add_argument(
        "--time", default=None, dest="time_limit",
        help=f"Slurm time limit (default $TOKEYE_SLURM_TIME or {DEFAULT_TIME}).",
    )
    parser.set_defaults(handler=_handle)


def _resolve_probe(args: argparse.Namespace) -> str | None:
    if args.probe:
        return str(args.probe)
    from tokeye.sources.foundation_presets import FOUNDATION_DIAGNOSTICS

    diag = FOUNDATION_DIAGNOSTICS.get(args.diag)
    return diag.default if diag else None


def _run_local(
    shots: list[int],
    outdir: Path,
    probe: str,
    tlim: tuple[float, float] | None,
    model: str,
    threshold: float,
    decimation: int,
    device: str,
) -> int:
    """Extract every shot's channel to ``inputs/`` then run one batch (one model
    load). Returns the failed-shot count."""
    import numpy as np

    from tokeye import batch
    from tokeye.sources.foundation import FoundationSource, pointname_slug

    inputs_dir = outdir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    slug = pointname_slug(probe)
    source = FoundationSource()

    npys: list[str] = []
    failures = 0
    for shot in shots:
        try:
            _t, x, fs = source.fetch(shot, probe, tlim)
            if x.size == 0:
                raise ValueError("no samples (signal absent or window empty)")
            d = int(decimation) if decimation else 1
            if d > 1 and x.size > 64:
                from scipy.signal import decimate

                x = decimate(x, d, ftype="fir")
                fs = fs / d
            path = inputs_dir / f"{shot}_{slug}.npy"
            np.save(path, x.astype(np.float32))
            npys.append(str(path))
            print(f"[{shot}] {x.size} samples @ {fs / 1e3:.1f} kHz -> {path.name}")
        except Exception as exc:  # noqa: BLE001 - one bad shot must not kill the run
            print(f"[{shot}] ERROR: {exc}", file=sys.stderr)
            failures += 1

    if npys:
        failures += batch.run_batch(
            npys,
            model=model,
            out_dir=outdir,
            save_png=True,
            threshold=threshold,
            device=device,
        )

    ok = len(shots) - failures
    print(f"princeton-batch: {ok}/{len(shots)} shot(s) ok -> {outdir}")
    return failures


def _handle(args: argparse.Namespace) -> int:
    try:
        shots = parse_shots(args.shots)
    except ValueError as exc:
        print(f"error: bad --shots: {exc}", file=sys.stderr)
        return 2
    if not shots:
        print("error: no shots given", file=sys.stderr)
        return 2

    probe = _resolve_probe(args)
    if not probe:
        print(f"error: unknown --diag '{args.diag}' and no --probe", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    model = args.model or "big_tf_unet"
    tlim = tuple(args.tlim) if args.tlim is not None else None

    if args.local:
        outdir.mkdir(parents=True, exist_ok=True)
        return _run_local(
            shots, outdir, probe, tlim, model,
            float(args.threshold), int(args.decimation), str(args.device),
        )

    script = build_sbatch_script(
        outdir=str(outdir),
        shots=shots,
        probe=probe,
        tlim=tlim,
        model=model,
        threshold=float(args.threshold),
        partition=args.partition or _default_partition(),
        gres=args.gres if args.gres is not None else _default_gres(),
        time_limit=args.time_limit or _default_time(),
        decimation=int(args.decimation),
        device=str(args.device),
    )

    if args.dry_run:
        print(script)
        return 0

    outdir.mkdir(parents=True, exist_ok=True)
    script_path = outdir / "submit.sh"
    script_path.write_text(script)
    try:
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print(
            "error: sbatch not found — run on a Slurm submit node, or use "
            "--local / --dry-run",
            file=sys.stderr,
        )
        return 2
    if result.returncode != 0:
        print(f"error: sbatch failed: {result.stderr.strip()}", file=sys.stderr)
        return result.returncode or 2
    print(result.stdout.strip())
    print(f"script: {script_path}")
    return 0
