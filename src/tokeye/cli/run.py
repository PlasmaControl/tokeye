"""``tokeye run`` — headless batch segmentation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tokeye.cli import _common, _options

if TYPE_CHECKING:
    import argparse
    from pathlib import Path


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
    _options.add_key_option(parser)
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
        type=_options.unit_float,
        default=0.5,
        help="preview threshold, in [0, 1] (default: %(default)s)",
    )
    _options.add_spectrogram_options(parser)
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    from tokeye import batch

    setup = _common.setup_or_report(args, "segmentation", unique_stems=True)
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
        model_name=args.model,
        channels=setup.channels,
        key=args.key,
        on_error=_common.report_failure,
    )
    written = len(setup.paths) - failures
    print(_summary(written, failures, setup.out_dir, fmt=args.fmt, png=args.png))
    return _common.EXIT_FAILED if failures else _common.EXIT_OK


def _summary(written: int, failed: int, out_dir: Path, *, fmt: str, png: bool) -> str:
    """The line ``tokeye run`` ends with: how many inputs, where, which files.

    For example ``wrote 3 inputs -> tokeye_output/ (mask, preview, params)``,
    with ``; 1 failed`` appended when inputs failed.
    """
    kinds = ["mask" if fmt == "npy" else "bundle", *(["preview"] if png else [])]
    inputs = "input" if written == 1 else "inputs"
    line = f"wrote {written} {inputs} -> {out_dir}{os.sep} ({', '.join(kinds)}, params)"
    return f"{line}; {failed} failed" if failed else line
