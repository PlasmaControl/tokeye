"""``tokeye download`` — pre-fetch model checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tokeye.cli import _common
from tokeye.config import DEFAULT_MODEL

if TYPE_CHECKING:
    import argparse


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "download",
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
    from huggingface_hub.errors import HfHubHTTPError

    from tokeye import hub

    for name in args.models or [DEFAULT_MODEL]:
        try:
            path = hub.download_model(name)
        except ValueError as exc:
            return _common.error(str(exc))
        except (HfHubHTTPError, OSError) as exc:
            _common.print_hub_error(name, exc)
            return _common.EXIT_USAGE
        print(path)
    return _common.EXIT_OK
