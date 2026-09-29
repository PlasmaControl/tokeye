"""``tokeye download`` — pre-fetch model checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tokeye.cli import _common, _options
from tokeye.config import DEFAULT_MODEL

if TYPE_CHECKING:
    import argparse


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "download",
        parents=[_options.VERBOSE],
        help="Download model weights into the Hugging Face cache.",
        description=(
            "Download model weights ahead of time, e.g. on an HPC login node "
            "before running on an offline compute node."
        ),
    )
    parser.add_argument(
        "models",
        nargs="*",
        metavar="MODEL",
        help=f"model registry name(s) (default: {DEFAULT_MODEL})",
    )
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    from tokeye import hub

    verbose = getattr(args, "verbose", False)
    for name in args.models or [DEFAULT_MODEL]:
        try:
            with _common.quiet_hub_logs(verbose):
                path = hub.download_model(name)
        except Exception as exc:  # noqa: BLE001 - report_model_error maps every type
            # One line, then stop: exit 2 at the first failure.
            return _common.report_model_error(name, exc, download=True)
        print(path)
    return _common.EXIT_OK
