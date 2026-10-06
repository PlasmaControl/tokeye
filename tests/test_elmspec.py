from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from tokeye.elmspec import (
    ElmEvent,
    column_activity,
    event_rows,
    extract_elm_events,
    seconds_per_col,
    summarize,
    write_event_rows,
    write_events_csv,
)


def _mask_with_bursts(bursts: list[tuple[int, int]], width: int = 50, height: int = 10):
    """Transient-channel mask with full-column bursts over [start, end] cols."""
    mask = np.zeros((height, width), dtype=np.float32)
    for start, end in bursts:
        mask[:, start : end + 1] = 1.0
    return mask


def test_column_activity_is_fraction_of_active_bins():
    mask = np.zeros((10, 4), dtype=np.float32)
    mask[:5, 1] = 1.0  # half the bins in column 1
    mask[:, 2] = 1.0  # all bins in column 2
    activity = column_activity(mask, threshold=0.5)
    assert activity.shape == (4,)
    assert activity[0] == 0.0
    assert activity[1] == 0.5
    assert activity[2] == 1.0


def test_extract_merges_events_across_small_gaps():
    mask = _mask_with_bursts([(10, 12), (15, 17)])  # gap of 2 columns
    events = extract_elm_events(mask, min_gap_cols=3)
    assert len(events) == 1
    assert events[0].start_col == 10
    assert events[0].end_col == 17


def test_extract_keeps_events_across_large_gaps():
    mask = _mask_with_bursts([(10, 12), (30, 32)])
    events = extract_elm_events(mask, min_gap_cols=3)
    assert [(e.start_col, e.end_col) for e in events] == [(10, 12), (30, 32)]


def test_extract_drops_short_events():
    mask = _mask_with_bursts([(5, 5), (20, 24)])
    events = extract_elm_events(mask, min_gap_cols=1, min_duration_cols=2)
    assert [(e.start_col, e.end_col) for e in events] == [(20, 24)]


def test_extract_empty_mask_gives_no_events():
    mask = np.zeros((10, 50), dtype=np.float32)
    assert extract_elm_events(mask) == []


def test_extract_records_peak_activity():
    mask = np.zeros((10, 50), dtype=np.float32)
    mask[:5, 10:13] = 1.0  # activity 0.5
    events = extract_elm_events(mask, activity_min=0.3)
    assert len(events) == 1
    assert events[0].peak_activity == 0.5


def test_summarize_counts_and_frequency():
    events = [ElmEvent(10, 12, 1.0), ElmEvent(30, 32, 1.0)]
    # 100 columns at hop=256, fs=200_000 -> 0.128 s total
    summary = summarize(events, n_cols=100, hop=256, fs=200_000.0)
    assert summary["n_events"] == 2
    assert np.isclose(summary["elm_freq_hz"], 2 / (100 * 256 / 200_000.0))
    assert np.isclose(summary["duty_cycle"], 6 / 100)


def test_summarize_without_fs_leaves_frequency_unset():
    summary = summarize([ElmEvent(0, 1, 1.0)], n_cols=10, hop=256, fs=None)
    assert summary["n_events"] == 1
    assert summary["elm_freq_hz"] is None


def test_write_events_csv(tmp_path):
    out = tmp_path / "elm_events.csv"
    events = [ElmEvent(10, 12, 0.75)]
    write_events_csv(out, [("shot1.npy", events)], hop=256, fs=200_000.0)

    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["input"] == "shot1.npy"
    assert rows[0]["event"] == "0"
    assert rows[0]["start_col"] == "10"
    assert rows[0]["end_col"] == "12"
    assert float(rows[0]["t_start_s"]) == 10 * 256 / 200_000.0
    assert float(rows[0]["peak_activity"]) == 0.75


def test_write_events_csv_without_fs_leaves_times_blank(tmp_path):
    out = tmp_path / "elm_events.csv"
    write_events_csv(out, [("shot1.npy", [ElmEvent(0, 3, 1.0)])], hop=256, fs=None)
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["t_start_s"] == ""


def test_seconds_per_col_prefers_dt():
    assert seconds_per_col(256, 200_000.0) == 256 / 200_000.0
    assert seconds_per_col(256, 200_000.0, dt=0.01) == 0.01
    assert seconds_per_col(dt=0.01) == 0.01
    assert seconds_per_col(256, None) is None
    assert seconds_per_col() is None


@pytest.mark.parametrize(
    ("name", "value"),
    [("dt", v) for v in (0.0, -1.0, float("nan"), math.inf)]
    + [("hop", v) for v in (0, -128, float("nan"), math.inf)]
    + [("fs", v) for v in (0.0, -1.0, float("nan"), math.inf)],
)
def test_seconds_per_col_rejects_a_non_positive_or_non_finite_timebase(name, value):
    # The dt message keeps its prefix, "dt must be positive".
    with pytest.raises(ValueError, match=rf"^{name} must be positive and finite"):
        if name == "dt":
            seconds_per_col(None, None, dt=value)
        elif name == "hop":
            seconds_per_col(value, 1000.0)
        else:
            seconds_per_col(128, value)


@pytest.mark.parametrize(
    "call",
    [
        lambda: event_rows("x", [], dt=0.0),
        lambda: event_rows("x", [], hop=128, fs=float("nan")),
        lambda: summarize([], n_cols=10, hop=0, fs=1000.0),
        lambda: summarize([], n_cols=10, duration_s=math.inf),
    ],
    ids=["rows-dt", "rows-fs", "summary-hop", "summary-duration"],
)
def test_a_bad_timebase_raises_even_without_events(call):
    with pytest.raises(ValueError, match="must be positive and finite"):
        call()


def test_event_times_keep_the_pre_1_0_arithmetic():
    row = event_rows("x", [ElmEvent(3, 4, 1.0)], hop=128, fs=100_000.0)[0]
    assert row["t_start_s"] == 3 * 128 / 100_000.0
    assert repr(row["t_start_s"]) == "0.00384"


def test_summarize_divides_by_the_signal_duration():
    summary = summarize(
        [ElmEvent(0, 1, 1.0)], n_cols=313, hop=128, fs=10_000.0, duration_s=4.0
    )
    assert summary["elm_freq_hz"] == 0.25


def test_summarize_with_dt_only():
    summary = summarize([ElmEvent(0, 1, 1.0)], n_cols=100, dt=0.001)
    assert np.isclose(summary["elm_freq_hz"], 1 / 0.1)


def test_event_rows_per_input_timebases(tmp_path):
    rows = event_rows("a.npy", [ElmEvent(10, 12, 1.0)], dt=0.001)
    rows += event_rows("b.npy", [ElmEvent(10, 12, 1.0)], hop=128, fs=1000.0)
    rows += event_rows("c.npy", [ElmEvent(10, 12, 1.0)])
    out = tmp_path / "elm_events.csv"
    write_event_rows(out, rows)

    with out.open(encoding="utf-8") as fh:
        read = list(csv.DictReader(fh))
    assert [r["input"] for r in read] == ["a.npy", "b.npy", "c.npy"]
    assert float(read[0]["t_start_s"]) == pytest.approx(0.010)
    assert float(read[1]["t_start_s"]) == pytest.approx(10 * 128 / 1000.0)
    assert read[2]["t_start_s"] == ""
    assert float(read[0]["duration_s"]) == pytest.approx(0.003)
