"""``tokeye alfvenspec`` — Alfvén-eigenmode detection with ae_tf_maskrcnn."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tokeye.cli import _common, _options

DEFAULT_WINDOW_COLS = 710  # = tokeye.alfvenspec.DEFAULT_WINDOW_COLS (no torch here)


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "alfvenspec",
        help="Detect Alfvén-eigenmode activity (boxes + masks via ae_tf_maskrcnn).",
        description=(
            "Run the ae_tf_maskrcnn instance model (needs the 'ae' extra: "
            "pip install 'tokeye[ae]'). Writes ae_detections.csv and, unless "
            "--no-masks, <stem>_ae_instances.npy: an (H, W) int32 map where "
            "i + 1 marks detection i of that input."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="files, directories, or glob patterns",
    )
    _options.add_model_options(parser, default="ae_tf_maskrcnn")
    _options.add_output_options(parser, default_dir="tokeye_ae")
    parser.add_argument(
        "--score-min",
        type=float,
        default=0.5,
        help="keep detections with at least this score (default: %(default)s)",
    )
    parser.add_argument(
        "--window-cols",
        type=int,
        default=DEFAULT_WINDOW_COLS,
        help=(
            "process wide spectrograms in windows of this many columns "
            "(training width; 0 disables windowing; default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=None,
        help="standardization mean (default: per-input statistics)",
    )
    parser.add_argument(
        "--std",
        type=_options.positive_float,
        default=None,
        help="standardization std (default: per-input statistics)",
    )
    parser.add_argument(
        "--masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write <stem>_ae_instances.npy per input (default: on)",
    )
    _options.add_spectrogram_options(parser)
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    import numpy as np

    from tokeye import batch
    from tokeye.alfvenspec import detect_windowed, write_detections_csv

    try:
        config = _options.config_from_args(args)
    except ValueError as exc:
        return _common.error(str(exc))
    paths = _common.collect_inputs_or_report(args.inputs)
    if paths is None:
        return _common.EXIT_USAGE
    model = _common.load_model_or_report(args.model, args.device, "instance")
    if model is None:
        return _common.EXIT_USAGE

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_detections = []
    failures = 0
    for path in paths:
        try:
            spec = batch.load_spectrogram(path, config)
            detections = detect_windowed(
                spec.values,
                model,
                window_cols=args.window_cols,
                score_min=args.score_min,
                mean=args.mean,
                std=args.std,
            )
        except Exception as exc:  # noqa: BLE001 - mirror `tokeye run`: keep batch going
            print(f"error: failed to process {path}: {exc}", file=sys.stderr)
            failures += 1
            continue

        all_detections.append((str(path), detections))
        print(f"{path}: {len(detections['boxes'])} detection(s)")
        if args.masks:
            np.save(
                out_dir / f"{path.stem}_ae_instances.npy", detections["instance_map"]
            )

    detections_csv = out_dir / "ae_detections.csv"
    write_detections_csv(detections_csv, all_detections)
    print(detections_csv)
    return _common.EXIT_FAILED if failures else _common.EXIT_OK
