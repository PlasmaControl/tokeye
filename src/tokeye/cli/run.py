"""``tokeye run`` — headless batch segmentation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tokeye.cli import _common, _options

if TYPE_CHECKING:
    import argparse


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "run",
        parents=[_options.VERBOSE],
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
    from tokeye import batch

    setup = _common.setup_or_report(args, "segmentation")
    if setup is None:
        return _common.EXIT_USAGE
    failures = batch.process_files(
        setup.paths,
        setup.model,
        setup.config,
        setup.out_dir,
        save_png=args.png,
        threshold=args.threshold,
        fs=args.fs,
        fmt=args.fmt,
        tile=args.tile,
        model_name=str(args.model),
        channels=setup.channels,
        on_error=_common.report_failure,
    )
    return _common.EXIT_FAILED if failures else _common.EXIT_OK
