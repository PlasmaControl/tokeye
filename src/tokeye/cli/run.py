"""``tokeye run`` — headless batch segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tokeye.cli import _common, _options

if TYPE_CHECKING:
    import argparse


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "run",
        help="Segment one or more inputs (files, directories or globs).",
        description=(
            "Segment each input and write <stem>_mask.npy (or <stem>_tokeye.npz "
            "with --format npz), <stem>_preview.png and <stem>_params.json."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="files (.npy .npz .wav .flac .ogg .csv .txt .mat .h5), "
        "directories, or glob patterns",
    )
    _options.add_model_options(parser)
    _options.add_tile_option(parser)
    _options.add_output_options(parser, default_dir="tokeye_output", png_default=True)
    _options.add_fs_option(parser)
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=("npy", "npz"),
        default="npy",
        help="npy = mask only; npz = mask + spectrogram + axes bundle "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="preview threshold (default: %(default)s)",
    )
    _options.add_spectrogram_options(parser)
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    from huggingface_hub.errors import HfHubHTTPError

    from tokeye import batch

    try:
        config = _options.config_from_args(args)
    except ValueError as exc:
        return _common.error(str(exc))

    try:
        failures = batch.run_batch(
            args.inputs,
            model=args.model,
            out_dir=Path(args.output_dir),
            config=config,
            save_png=args.png,
            threshold=args.threshold,
            device=args.device,
            fs=args.fs,
            fmt=args.fmt,
            tile=args.tile,
        )
    except ValueError as exc:
        hint = _common.NO_INPUT_HINT if "No input files found" in str(exc) else ""
        return _common.error(f"{exc}{hint}")
    except (FileNotFoundError, ImportError) as exc:
        return _common.error(str(exc))
    except (HfHubHTTPError, OSError) as exc:
        _common.print_hub_error(args.model, exc)
        return _common.EXIT_USAGE

    return _common.EXIT_FAILED if failures else _common.EXIT_OK
