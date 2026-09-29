"""``tokeye example`` — write a synthetic demo signal."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tokeye.cli import _common, _options

if TYPE_CHECKING:
    import argparse

DEFAULT_FS = 200_000.0


def default_output(fs: float) -> str:
    """``tokeye_example_sr<fs>.npy``: the ``_sr`` suffix records the rate,
    so ``tokeye run`` and the app pick it up without ``--fs``."""
    rate = str(int(fs)) if float(fs).is_integer() else str(fs)
    return f"tokeye_example_sr{rate}.npy"


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "example", help="Write a synthetic example signal to a .npy file."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output path (default: tokeye_example_sr<FS>.npy)",
    )
    parser.add_argument(
        "--duration",
        type=_options.positive_float,
        default=2.0,
        help="length [s] (default: %(default)s)",
    )
    parser.add_argument(
        "--fs",
        type=_options.positive_float,
        default=DEFAULT_FS,
        help="sampling rate [Hz] (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    import numpy as np

    from tokeye.examples import make_example_signal

    output_path = Path(args.output or default_output(args.fs))
    if output_path.suffix != ".npy":
        # np.save silently appends ".npy" to paths without that suffix;
        # normalize up-front so the printed path is the file that exists.
        output_path = output_path.with_suffix(".npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signal = make_example_signal(duration_s=args.duration, fs=args.fs, seed=args.seed)
    np.save(output_path, signal)
    print(output_path)
    print(f"next: tokeye run {output_path}", file=sys.stderr)
    return _common.EXIT_OK
