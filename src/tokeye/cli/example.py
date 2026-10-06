"""``tokeye example`` — write a synthetic demo signal."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tokeye.cli import _common, _options

if TYPE_CHECKING:
    import argparse

DEFAULT_FS = 200_000.0


def default_output(fs: float) -> str:
    """``tokeye_example_sr<fs>.npy``: the ``_sr`` suffix lets ``tokeye run``
    read the rate from the name (the app does not)."""
    from tokeye.examples import example_filename

    return example_filename(fs)


def _shell_quote(path: str) -> str:
    """``path`` quoted for the platform's shell: double quotes on Windows
    (``cmd.exe`` and PowerShell), POSIX quoting elsewhere."""
    if os.name == "nt":
        return subprocess.list2cmdline([path])
    return shlex.quote(path)


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "example",
        parents=[_options.VERBOSE],
        help="Write a synthetic example signal to a .npy file.",
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

    from tokeye.examples import format_rate, make_example_signal
    from tokeye.io import fs_from_name

    output_path = Path(args.output or default_output(args.fs))
    if not output_path.name.endswith(".npy"):
        # np.save appends ".npy" to any name that does not end with it (so
        # "demo_sr1234.5" becomes "demo_sr1234.5.npy", not "demo_sr1234.npy");
        # do the same up-front so the printed path is the file that exists.
        output_path = output_path.with_name(output_path.name + ".npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signal = make_example_signal(duration_s=args.duration, fs=args.fs, seed=args.seed)
    np.save(output_path, signal)
    print(output_path)
    command = f"tokeye run {_shell_quote(str(output_path))}"
    if fs_from_name(output_path) != args.fs:  # --fs overrides the name
        command += f" --fs {format_rate(args.fs)}"
    print(f"next: {command}", file=sys.stderr)
    return _common.EXIT_OK
