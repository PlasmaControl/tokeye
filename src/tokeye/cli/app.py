"""``tokeye app`` — launch the Gradio web app."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tokeye.cli import _common, _options

if TYPE_CHECKING:
    import argparse

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860

MISSING_EXTRA = (
    "`tokeye app` needs the 'app' extra (gradio), which is not installed.\n"
    "Install it with:\n"
    "    pip install 'tokeye[app]'      # or:  uv sync --extra app\n"
    "(underlying import error: {exc})"
)


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "app",
        parents=[_options.VERBOSE],
        help="Launch the TokEye web app in your browser.",
        description=(
            "Serve the app on http://HOST:PORT (the next free port if PORT is "
            "busy). On a remote machine, forward the port over SSH instead of "
            "using --share."
        ),
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="interface to bind; 0.0.0.0 exposes the app to your network "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="first port to try (default: %(default)s)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="also create a public gradio.live link (anyone with it can use the app)",
    )
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument(
        "--open",
        dest="browser",
        action="store_const",
        const=True,
        help="open a browser tab (default when not in an SSH session)",
    )
    browser.add_argument(
        "--no-browser",
        dest="browser",
        action="store_const",
        const=False,
        help="do not open a browser tab",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        metavar="DIR",
        help="directory the app reads and writes files in (created if missing; "
        "default: the current directory)",
    )
    parser.set_defaults(handler=_handle, browser=None)


def _handle(args: argparse.Namespace) -> int:
    try:
        from tokeye.app.__main__ import main as app_main
    except ImportError as exc:
        print(MISSING_EXTRA.format(exc=exc), file=sys.stderr)
        return _common.EXIT_USAGE

    remote = "SSH_CONNECTION" in os.environ
    open_browser = args.browser if args.browser is not None else not remote
    if remote and not args.share:
        print(
            f"note: SSH session detected; forward the port from your laptop: "
            f"ssh -L {args.port}:localhost:{args.port} <this-host>",
            file=sys.stderr,
        )
    if args.share:
        print(
            "warning: --share makes the app reachable by anyone with the link",
            file=sys.stderr,
        )
    if args.workspace is not None:
        args.workspace.mkdir(parents=True, exist_ok=True)
        os.chdir(args.workspace)

    app_main(
        port=args.port, share=args.share, open_browser=open_browser, host=args.host
    )
    return _common.EXIT_OK
