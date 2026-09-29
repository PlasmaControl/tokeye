"""``tokeye elmspec`` — detect ELM events via the transient-activity channel."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from tokeye.cli import _common, _options

if TYPE_CHECKING:
    import argparse


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "elmspec",
        parents=[_options.VERBOSE],
        help="Detect ELM events (transient-channel intervals, count, frequency).",
        description=(
            "Segment each input, then turn broadband stripes in the transient "
            "channel into ELM events. Writes elm_events.csv and "
            "elm_summary.csv (plus <stem>_elm_preview.png with --png)."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="files, directories, or glob patterns",
    )
    _options.add_model_options(parser)
    _options.add_tile_option(parser)
    _options.add_output_options(parser, default_dir="tokeye_elms", png_default=False)
    _options.add_fs_option(parser)
    parser.add_argument(
        "--dt",
        type=_options.positive_float,
        default=None,
        help=(
            "seconds per spectrogram column; overrides --hop/--fs (use it for "
            "2D inputs whose columns are not --hop samples apart)"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=_options.unit_float,
        default=0.5,
        help="mask binarization threshold, in [0, 1] (default: %(default)s)",
    )
    parser.add_argument(
        "--activity-min",
        type=_options.unit_float,
        default=0.1,
        help=(
            "minimum fraction of active frequency bins for a time column to "
            "belong to an ELM, in [0, 1] (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--min-gap-cols",
        type=_options.nonnegative_int,
        default=3,
        help="merge events separated by at most this many columns, >= 0 "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--min-duration-cols",
        type=_options.positive_int,
        default=1,
        help="drop events shorter than this many columns, >= 1 (default: %(default)s)",
    )
    _options.add_spectrogram_options(parser)
    parser.set_defaults(handler=_handle)


def _transient(seg):
    """The transient channel: by name, else channel 1 of a 2+ channel mask."""
    if "transient" in seg.channels:
        return seg["transient"]
    if seg.mask.shape[0] >= 2:
        return seg.mask[1]
    raise ValueError(
        f"model {seg.model!r} has no transient channel (channels: {seg.channels})"
    )


def _handle(args: argparse.Namespace) -> int:
    from tokeye import batch
    from tokeye._plotting import save_preview
    from tokeye.config import resolve_channels
    from tokeye.elmspec import (
        event_rows,
        extract_elm_events,
        seconds_per_col,
        summarize,
        write_event_rows,
        write_summary_csv,
    )
    from tokeye.inference import infer
    from tokeye.result import Segmentation

    setup = _common.setup_or_report(args, "segmentation", unique_stems=args.png)
    if setup is None:
        return _common.EXIT_USAGE
    config, out_dir = setup.config, setup.out_dir

    rows = []
    summaries = []
    failures = 0
    for path in setup.paths:
        try:
            spec = batch.load_spectrogram(path, config, fs=args.fs)
            mask = infer(setup.model, spec.values, tile=args.tile)
            names = resolve_channels(setup.channels, mask.shape[0])
            seg = Segmentation(mask, spec, names, str(args.model))
            events = extract_elm_events(
                _transient(seg),
                threshold=args.threshold,
                activity_min=args.activity_min,
                min_gap_cols=args.min_gap_cols,
                min_duration_cols=args.min_duration_cols,
            )
        except Exception as exc:  # noqa: BLE001 - mirror `tokeye run`: keep batch going
            _common.report_failure(path, exc)
            failures += 1
            continue

        dt = args.dt
        if dt is None and spec.fs is not None:
            dt = seconds_per_col(config.hop, spec.fs)
            if spec.kind == "spectrogram":
                print(
                    f"note: {path} is a spectrogram; times assume its columns "
                    f"are --hop={config.hop} samples apart (set --dt to override)",
                    file=sys.stderr,
                )
        summary = summarize(events, n_cols=mask.shape[-1], dt=dt)
        rows += event_rows(str(path), events, dt=dt)
        summaries.append((str(path), summary))
        freq = summary["elm_freq_hz"]
        freq_text = f", {freq:.1f} Hz" if freq is not None else ""
        print(f"{path}: {summary['n_events']} ELM event(s){freq_text}")

        if args.png:
            preview = out_dir / f"{path.stem}_elm_preview.png"
            save_preview(seg, preview, threshold=args.threshold)

    events_csv = out_dir / "elm_events.csv"
    summary_csv = out_dir / "elm_summary.csv"
    write_event_rows(events_csv, rows)
    write_summary_csv(summary_csv, summaries)
    print(events_csv)
    print(summary_csv)
    return _common.EXIT_FAILED if failures else _common.EXIT_OK
