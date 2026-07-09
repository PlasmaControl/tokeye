"""``tokeye fetch`` — pre-cache a DIII-D shot signal to a ``.npy`` file.

Fetches one diagnostic pointname from MDSplus (atlas.gat.com) via
``tokeye.sources`` and writes a 1-D ``.npy``. Run it on a node that can reach
atlas (a login / somega node); the path-based ``tokeye run`` and the app's
Analyze tab can then consume the file offline (e.g. on a batch GPU node).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "fetch",
        help="Fetch a DIII-D shot signal from MDSplus and cache it as .npy.",
    )
    parser.add_argument("--shot", type=int, required=True, help="DIII-D shot number.")
    parser.add_argument(
        "--diag",
        default="mag",
        help=(
            "Diagnostic preset: mag|mag_pol|mhr|ece|co2|bes "
            "(default: mag, toroidal Mirnov)."
        ),
    )
    parser.add_argument(
        "--pointname",
        default=None,
        help="PTDATA pointname (default: the diagnostic's default probe).",
    )
    parser.add_argument(
        "--tlim",
        type=float,
        nargs=2,
        metavar=("T_MIN_MS", "T_MAX_MS"),
        default=None,
        help="Optional time window in ms.",
    )
    parser.add_argument(
        "--out",
        default="data/input",
        help="Output directory for the .npy (default: data/input).",
    )
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    import numpy as np

    from tokeye.sources import DIAGNOSTICS, MDSSource

    diag = DIAGNOSTICS.get(args.diag)
    if diag is None:
        print(
            f"error: unknown diagnostic '{args.diag}' "
            f"(choices: {', '.join(DIAGNOSTICS)})",
            file=sys.stderr,
        )
        return 2
    pointname = args.pointname or diag.default
    tlim = tuple(args.tlim) if args.tlim is not None else None

    try:
        t_ms, x, fs = MDSSource().fetch(args.shot, pointname, tlim)
    except RuntimeError as exc:  # MDSplus unavailable and no cache
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # atlas / tree / pointname failures
        print(
            f"error: failed to fetch {args.shot}/{pointname}: {exc}",
            file=sys.stderr,
        )
        return 2

    if x.size == 0:
        print(
            f"error: {args.shot}/{pointname} returned no samples"
            + (" in that window" if tlim else ""),
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.shot}_{pointname}.npy"
    np.save(out_path, x)

    span_ms = float(t_ms[-1] - t_ms[0]) if t_ms.size > 1 else 0.0
    print(f"{out_path}  ({x.size} samples, {span_ms:.1f} ms, fs={fs / 1e3:.1f} kHz)")
    return 0
