"""``tokeye alfvenspec`` — Alfvén-eigenmode detection with ae_tf_maskrcnn."""

from __future__ import annotations

import argparse
import sys

from tokeye.cli import _common, _options

# = tokeye.alfvenspec.inference.DEFAULT_WINDOW_COLS and _MIN_WINDOW_COLS;
# importing them would import torch.
DEFAULT_WINDOW_COLS = 710
MIN_WINDOW_COLS = 32


def window_cols(text: str) -> int:
    """argparse ``type=`` for ``--window-cols``: 0 or an integer >= 32."""
    rule = f"must be 0 (no windowing) or an integer >= {MIN_WINDOW_COLS}"
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{rule}, got {text!r}") from None
    if value != 0 and value < MIN_WINDOW_COLS:
        raise argparse.ArgumentTypeError(f"{rule}, got {text}")
    return value


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "alfvenspec",
        parents=[_options.VERBOSE],
        help="Detect Alfvén-eigenmode activity (boxes + masks via ae_tf_maskrcnn).",
        description=(
            "Run the ae_tf_maskrcnn instance model (needs the 'ae' extra: "
            "pip install 'tokeye[ae]'). Writes ae_detections.csv and, unless "
            "--no-masks, <stem>_ae_instances.npy: an (H, W) int32 map where "
            "i + 1 marks detection i of that input. Unwindowed inputs with "
            "detections also get <stem>_ae_masks.npy, the per-detection soft "
            "masks (deprecated; no longer written from 2.0)."
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
    _options.add_key_option(parser)
    parser.add_argument(
        "--score-min",
        type=_options.unit_float,
        default=0.5,
        help="keep detections with at least this score, in [0, 1] "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--window-cols",
        type=window_cols,
        default=DEFAULT_WINDOW_COLS,
        help=(
            "process wide spectrograms in windows of this many columns (the "
            "default is the training width): 0 disables windowing, else >= "
            f"{MIN_WINDOW_COLS} (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--mean",
        type=_options.finite_float,
        default=None,
        help="standardization mean, a finite number (default: per-input statistics)",
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
        help=(
            "write <stem>_ae_instances.npy per input, and the deprecated "
            "<stem>_ae_masks.npy for unwindowed inputs with detections "
            "(default: on)"
        ),
    )
    _options.add_spectrogram_options(parser)
    parser.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    import numpy as np

    from tokeye import batch
    from tokeye.alfvenspec import detect_windowed, write_detections_csv

    setup = _common.setup_or_report(args, "instance", unique_stems=args.masks)
    if setup is None:
        return _common.EXIT_USAGE
    out_dir = setup.out_dir

    # Only what the CSV needs: an input's instance map and masks stack are
    # dropped once its files are written.
    all_detections = []
    failures = 0
    wrote_masks = False
    for path in setup.paths:
        try:
            spec = batch.load_spectrogram(path, setup.config, key=args.key)
            detections = detect_windowed(
                spec.values,
                setup.model,
                window_cols=args.window_cols,
                score_min=args.score_min,
                mean=args.mean,
                std=args.std,
            )
            masks = detections["masks"]  # None when the input was windowed
            has_masks = args.masks and masks is not None and len(masks) > 0
            if args.masks:
                np.save(
                    out_dir / f"{path.stem}_ae_instances.npy",
                    detections["instance_map"],
                )
            if has_masks:
                np.save(out_dir / f"{path.stem}_ae_masks.npy", masks)
        except Exception as exc:  # noqa: BLE001 - mirror `tokeye run`: keep batch going
            _common.report_failure(path, exc)
            failures += 1
            continue

        wrote_masks = wrote_masks or has_masks
        kept = {key: detections[key] for key in ("boxes", "labels", "scores")}
        all_detections.append((str(path), kept))
        print(f"{path}: {len(kept['boxes'])} detection(s)")

    detections_csv = out_dir / "ae_detections.csv"
    write_detections_csv(detections_csv, all_detections)
    print(detections_csv)
    if wrote_masks:
        print(
            "note: <stem>_ae_masks.npy (per-detection soft masks) is deprecated "
            "and is no longer written from 2.0; use <stem>_ae_instances.npy",
            file=sys.stderr,
        )
    return _common.EXIT_FAILED if failures else _common.EXIT_OK
