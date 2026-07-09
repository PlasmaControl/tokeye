"""``tokeye gui`` — launch the native desktop GUI (PySide6 + pyqtgraph).

Mirrors ``tokeye.cli.app``: stdlib-only at module scope, with the heavy import
deferred into the handler and a friendly message when the ``gui`` extra is
missing.
"""

from __future__ import annotations

import argparse
import sys

_MISSING_EXTRA = (
    "`tokeye gui` needs the 'gui' extra (PySide6, pyqtgraph), which is not "
    "installed.\n"
    "Install it with:\n"
    "    pip install 'tokeye[gui]'      # or:  uv sync --extra gui\n"
)


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "gui",
        help="Launch the native desktop GUI (PySide6 + pyqtgraph).",
    )
    parser.add_argument(
        "--view",
        choices=["spectrogram", "modespec"],
        default=None,
        help="Open directly on a specific view (default: spectrogram).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Build the window, run one event-loop tick, then exit 0 "
        "(for headless/offscreen CI).",
    )
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    try:
        import pyqtgraph  # noqa: F401
        import PySide6  # noqa: F401
    except ImportError as exc:
        print(f"{_MISSING_EXTRA}(underlying import error: {exc})", file=sys.stderr)
        return 1

    from tokeye.gui.app import run

    return run(view=args.view, self_test=args.self_test)


def launch_default() -> int:
    """Entry for bare-``tokeye`` autolaunch: default view, no self-test."""
    return _handle(argparse.Namespace(view=None, self_test=False))
