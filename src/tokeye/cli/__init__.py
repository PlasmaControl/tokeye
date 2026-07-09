"""``tokeye`` console entry point.

Argparse only (no new dependencies). Heavy imports (torch, ``tokeye.batch``,
``tokeye.app``) are deferred into each subcommand handler so ``tokeye --help``
returns instantly and ``tokeye run`` never imports gradio.

Each subcommand lives in its own module under ``tokeye.cli`` and exposes
``add_subcommand(subparsers)``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

from tokeye.cli import (
    alfvenspec,
    app,
    diiid_batch,
    download,
    eigspec,
    elmspec,
    example,
    fetch,
    gui,
    modesearch,
    modespec,
    run,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokeye",
        description=(
            "Automatic classification and localization of fluctuating signals "
            "in spectrograms."
        ),
    )
    parser.add_argument(
        "--version", action="store_true", help="Print the tokeye version and exit."
    )
    subparsers = parser.add_subparsers(dest="command")
    app.add_subcommand(subparsers)
    gui.add_subcommand(subparsers)
    run.add_subcommand(subparsers)
    download.add_subcommand(subparsers)
    example.add_subcommand(subparsers)
    fetch.add_subcommand(subparsers)
    diiid_batch.add_subcommand(subparsers)
    modespec.add_subcommand(subparsers)
    elmspec.add_subcommand(subparsers)
    alfvenspec.add_subcommand(subparsers)
    eigspec.add_subcommand(subparsers)
    modesearch.add_subcommand(subparsers)
    return parser


def _should_autolaunch_gui() -> bool:
    """Whether bare ``tokeye`` (no subcommand) should open the desktop GUI.

    Only in an interactive session with a display. The ``isatty`` conjunct keeps
    ``main([])`` returning 2 under pytest/CI (captured stdout, often no display),
    so ``tests/test_cli`` stays green; ``TOKEYE_NO_GUI`` is a scriptable escape
    hatch.
    """
    if os.environ.get("TOKEYE_NO_GUI"):
        return False
    display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    return bool(display) and sys.stdout.isatty()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from importlib.metadata import version

        print(version("tokeye"))
        return 0

    if args.command is None:
        if _should_autolaunch_gui():
            return gui.launch_default()
        parser.print_help()
        return 2

    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
