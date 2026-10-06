"""``tokeye`` console entry point.

Argparse only (no new dependencies). Heavy imports (torch, ``tokeye.batch``,
``tokeye.app``) are deferred into each subcommand handler so ``tokeye --help``
returns instantly and ``tokeye run`` never imports gradio. At module level,
CLI modules import only :mod:`tokeye.config` (standard library).

Each subcommand lives in its own module under ``tokeye.cli`` and exposes
``add_subcommand(subparsers)``.

Exit codes: 0 = every input succeeded, 1 = at least one input failed,
2 = usage or configuration error, 130 = interrupted. Errors are one
``error:`` line on stderr, and Python warnings one ``warning:`` line; ``-v``
(before or after the subcommand) adds the debug logs and tracebacks and
shows warnings in full.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from tokeye._version import __version__
from tokeye.cli import (
    _common,
    _options,
    alfvenspec,
    app,
    download,
    elmspec,
    example,
    info,
    run,
)
from tokeye.cli._common import EXIT_INTERRUPTED, EXIT_USAGE

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokeye",
        parents=[_options.VERBOSE],
        description=(
            "Automatic classification and localization of fluctuating signals "
            "in spectrograms."
        ),
    )
    parser.add_argument("--version", action="version", version=f"tokeye {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    run.add_subcommand(subparsers)
    app.add_subcommand(subparsers)
    download.add_subcommand(subparsers)
    example.add_subcommand(subparsers)
    info.add_subcommand(subparsers)
    elmspec.add_subcommand(subparsers)
    alfvenspec.add_subcommand(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE

    verbose = getattr(args, "verbose", False)
    with _common.verbose_logging(verbose), _common.one_line_warnings(verbose):
        try:
            return args.handler(args)
        except KeyboardInterrupt:
            print("interrupted", file=sys.stderr)
            return EXIT_INTERRUPTED
        except Exception as exc:  # noqa: BLE001 - the last resort: one line
            return _common.report_unexpected(exc)


if __name__ == "__main__":
    sys.exit(main())
