"""Pure-numpy ELM event extraction from a transient-activity mask.

Input is the transient channel of a TokEye mask (``mask[1]``, shape ``(H, W)``,
values in [0, 1]). An ELM shows up as a broadband vertical stripe: many
frequency bins active in the same time column. Detection is therefore
column-wise: threshold the mask, measure the active fraction per column,
mark columns above ``activity_min``, close small gaps, and report the
remaining contiguous runs as events.

Times are column times: column ``j`` is ``j * hop / fs`` seconds after the
first sample (``j * dt`` with ``dt``), the centre of its STFT frame (see
:func:`tokeye.transforms.compute_stft`). An event's ``t_start_s`` is the time
of its first active column, ``t_end_s`` the time of the column after its last
active one, and ``duration_s = t_end_s - t_start_s``.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ElmEvent:
    start_col: int
    end_col: int  # inclusive
    peak_activity: float

    @property
    def duration_cols(self) -> int:
        return self.end_col - self.start_col + 1


def column_activity(transient_mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Fraction of frequency bins at or above ``threshold``, per time column."""
    return (transient_mask >= threshold).mean(axis=0)


def _contiguous_runs(active: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of each True run."""
    padded = np.concatenate(([False], active, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    starts, ends = edges[::2], edges[1::2] - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _fill_gaps(active: np.ndarray, max_gap: int) -> np.ndarray:
    """Close False gaps of at most ``max_gap`` columns between True runs."""
    if max_gap <= 0:
        return active
    filled = active.copy()
    runs = _contiguous_runs(active)
    for (_, prev_end), (next_start, _) in zip(runs, runs[1:], strict=False):
        if next_start - prev_end - 1 <= max_gap:
            filled[prev_end : next_start + 1] = True
    return filled


def extract_elm_events(
    transient_mask: np.ndarray,
    *,
    threshold: float = 0.5,
    activity_min: float = 0.1,
    min_gap_cols: int = 3,
    min_duration_cols: int = 1,
) -> list[ElmEvent]:
    """Detect ELM events in a ``(H, W)`` transient-activity mask.

    ``threshold`` binarizes mask values; ``activity_min`` is the minimum
    active-bin fraction for a column to count as part of an event;
    runs separated by gaps of at most ``min_gap_cols`` columns are merged;
    events shorter than ``min_duration_cols`` are dropped.
    """
    activity = column_activity(transient_mask, threshold=threshold)
    active = _fill_gaps(activity >= activity_min, min_gap_cols)
    return [
        ElmEvent(start, end, float(activity[start : end + 1].max()))
        for start, end in _contiguous_runs(active)
        if end - start + 1 >= min_duration_cols
    ]


def seconds_per_col(
    hop: int | None = None, fs: float | None = None, *, dt: float | None = None
) -> float | None:
    """Seconds per spectrogram column: ``dt`` if given, else ``hop / fs``.

    Column ``j`` is ``j * hop / fs`` seconds after the first sample, or
    ``j * dt``. ``None`` when ``dt`` is not given and ``hop`` or ``fs`` is
    ``None`` (columns then have no absolute timebase).

    Raises
    ------
    ValueError
        ``dt``, ``hop`` or ``fs`` is given (not ``None``) but is not positive
        and finite, whether or not it is used.
    """
    for name, value in (("dt", dt), ("hop", hop), ("fs", fs)):
        if value is not None and not 0 < value < math.inf:
            raise ValueError(f"{name} must be positive and finite, got {value!r}")
    if dt is not None:
        return float(dt)
    if hop is None or fs is None:
        return None
    return hop / fs


def _cols_to_s(
    n: int, hop: int | None, fs: float | None, dt: float | None
) -> float | None:
    """``n`` columns in seconds: ``n * dt``, else ``n * hop / fs``, else ``None``.

    ``n * hop / fs`` is evaluated left to right, as before 1.0 (never
    ``n * (hop / fs)``, which differs in the last bit for some ``n``).
    """
    if dt is not None:
        return n * dt
    if hop is None or fs is None:
        return None
    return n * hop / fs


def summarize(
    events: list[ElmEvent],
    n_cols: int,
    hop: int | None = None,
    fs: float | None = None,
    *,
    dt: float | None = None,
    duration_s: float | None = None,
) -> dict[str, float | int | None]:
    """Per-input summary: event count, ELM frequency, duty cycle.

    ``elm_freq_hz`` is events per second of analyzed signal: ``n_events /
    duration_s`` when ``duration_s`` is given, else per ``n_cols`` columns
    (``n_cols * hop / fs`` or ``n_cols * dt`` seconds); ``None`` when the
    column spacing is unknown (see :func:`seconds_per_col`). For a 1D input,
    pass ``duration_s = n_samples / fs``: its ``1 + n_samples // hop``
    columns span only ``n_samples // hop`` hops.

    Raises
    ------
    ValueError
        A bad timebase (see :func:`seconds_per_col`), even without events,
        or a ``duration_s`` that is not positive and finite.
    """
    seconds_per_col(hop, fs, dt=dt)
    if duration_s is not None and not 0 < duration_s < math.inf:
        raise ValueError(f"duration_s must be positive and finite, got {duration_s!r}")
    active_cols = sum(event.duration_cols for event in events)
    total_s = duration_s if duration_s is not None else _cols_to_s(n_cols, hop, fs, dt)
    return {
        "n_events": len(events),
        "elm_freq_hz": len(events) / total_s if total_s else None,
        "duty_cycle": active_cols / n_cols if n_cols else 0.0,
    }


EVENT_FIELDS = (
    "input",
    "event",
    "start_col",
    "end_col",
    "duration_cols",
    "t_start_s",
    "t_end_s",
    "duration_s",
    "peak_activity",
)

SUMMARY_FIELDS = ("input", "n_events", "elm_freq_hz", "duty_cycle")


def event_rows(
    name: str,
    events: list[ElmEvent],
    *,
    hop: int | None = None,
    fs: float | None = None,
    dt: float | None = None,
) -> list[dict[str, object]]:
    """CSV rows (:data:`EVENT_FIELDS`) for one input's events.

    Column ``j`` is ``j * hop / fs`` seconds after the first sample (``j *
    dt`` with ``dt``). ``t_start_s`` is the time of the first active column,
    ``t_end_s`` that of the column after the last active one, and
    ``duration_s = t_end_s - t_start_s``; they are blank (``""``) when the
    column spacing is unknown. Build rows per input when inputs have
    different timebases, then write them all with :func:`write_event_rows`.

    Raises
    ------
    ValueError
        A bad timebase (see :func:`seconds_per_col`), even without events.
    """
    seconds_per_col(hop, fs, dt=dt)

    def seconds(n: int) -> float | str:
        value = _cols_to_s(n, hop, fs, dt)
        return "" if value is None else value

    return [
        {
            "input": name,
            "event": index,
            "start_col": event.start_col,
            "end_col": event.end_col,
            "duration_cols": event.duration_cols,
            "t_start_s": seconds(event.start_col),
            "t_end_s": seconds(event.end_col + 1),
            "duration_s": seconds(event.duration_cols),
            "peak_activity": event.peak_activity,
        }
        for index, event in enumerate(events)
    ]


def write_event_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write :func:`event_rows` output as a CSV with a header."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_events_csv(
    path: Path,
    per_input: list[tuple[str, list[ElmEvent]]],
    hop: int | None = None,
    fs: float | None = None,
    *,
    dt: float | None = None,
) -> None:
    """One row per detected event, all inputs sharing one timebase.

    Time columns are blank when the column spacing is unknown.
    """
    rows = [
        row
        for name, events in per_input
        for row in event_rows(name, events, hop=hop, fs=fs, dt=dt)
    ]
    write_event_rows(path, rows)


def write_summary_csv(
    path: Path, per_input: list[tuple[str, dict[str, float | int | None]]]
) -> None:
    """One row per input file with its :func:`summarize` result."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for name, summary in per_input:
            row = {"input": name, **summary}
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
