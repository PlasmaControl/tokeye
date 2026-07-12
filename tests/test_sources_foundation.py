"""Tests for the Princeton foundation_model HDF5 source.

All offline: shot files are tiny synthetic ``{shot}_processed.h5`` built in
``tmp_path`` with the real archive's layout (``xdata`` float32 seconds ``(N,)``,
``ydata`` float32 ``(C, N)``, no attributes, absent signal = shape ``(1,)``).
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


def _write_shot(root, shot, groups):
    """Write ``{shot}_processed.h5`` with ``{name: (xdata, ydata)}`` groups."""
    path = root / f"{shot}_processed.h5"
    with h5py.File(path, "w") as f:
        for name, (x, y) in groups.items():
            g = f.create_group(name)
            g.create_dataset("xdata", data=np.asarray(x, dtype=np.float32))
            g.create_dataset("ydata", data=np.asarray(y, dtype=np.float32))
    return path


def _ramp_shot(root, shot, *, fs=500_000.0, t0=-4.336, n=2_500_000, n_ch=2):
    """A shot whose float32 time base reproduces the archive's quantization."""
    t = t0 + np.arange(n, dtype=np.float64) / fs
    y = np.tile(np.arange(n, dtype=np.float64) % 1000.0, (n_ch, 1))
    _write_shot(root, shot, {"mirnov": (t, y)})
    return t


def test_import_hygiene_no_h5py_or_mdsplus():
    # The sources package (and the foundation module itself) must import
    # without h5py: it is an optional extra, deferred to fetch-time.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import tokeye.sources, tokeye.sources.foundation, sys; "
            "assert 'h5py' not in sys.modules; "
            "assert 'MDSplus' not in sys.modules",
        ],
        check=True,
    )


# ── pointname scheme ──────────────────────────────────────────────────────────────
def test_parse_pointname_variants():
    from tokeye.sources.foundation import parse_pointname

    assert parse_pointname("mirnov/07") == ("mirnov", 7)
    assert parse_pointname("mirnov/7") == ("mirnov", 7)
    assert parse_pointname(" sxr/319 ") == ("sxr", 319)
    assert parse_pointname("mirnov") == ("mirnov", 0)  # bare group -> channel 0
    for bad in ("", "/7", "mirnov/", "mirnov/x", "mirnov/-1"):
        with pytest.raises(ValueError):
            parse_pointname(bad)


def test_pointname_slug_is_filename_safe():
    from tokeye.sources.foundation import pointname_slug

    assert pointname_slug("mirnov/07") == "mirnov-07"
    assert "/" not in pointname_slug("a/b/c")


# ── fs derivation (the float32 quantization trap) ─────────────────────────────────
def test_fetch_fs_survives_float32_timebase(tmp_path):
    # A float32 500 kHz ramp starting at -4.336 s: per-sample diffs quantize to
    # ULP multiples and a median-diff estimate returns 524288 Hz. The endpoint
    # derivation must stay within 1 Hz of the truth.
    from tokeye.sources.foundation import FoundationSource

    _ramp_shot(tmp_path, 190000)
    t_ms, x, fs = FoundationSource(data_dir=tmp_path).fetch(
        190000, "mirnov/01", (0.0, 500.0)
    )
    assert abs(fs - 500_000.0) < 1.0
    assert t_ms.dtype == np.float64
    assert x.size == t_ms.size > 0
    # ms axis, inside the requested window
    assert t_ms[0] >= -1e-6 and t_ms[-1] <= 500.0 + 1e-6
    # ~500 ms at 500 kHz -> ~250k samples
    assert abs(x.size - 250_000) < 5


def test_fetch_full_signal_and_negative_start(tmp_path):
    from tokeye.sources.foundation import FoundationSource

    _ramp_shot(tmp_path, 190001, n=50_000)
    t_ms, x, fs = FoundationSource(data_dir=tmp_path).fetch(190001, "mirnov/00")
    assert x.size == 50_000
    assert t_ms[0] == pytest.approx(-4336.0, abs=0.01)
    assert abs(fs - 500_000.0) < 1.0


# ── tlim crop semantics ───────────────────────────────────────────────────────────
def test_fetch_tlim_inclusive_endpoints(tmp_path):
    from tokeye.sources.foundation import FoundationSource

    # dt = 1/1024 s is exactly representable in float32 -> exact index math.
    n = 100
    t = np.arange(n, dtype=np.float64) / 1024.0
    y = np.arange(n, dtype=np.float64)[None, :]
    _write_shot(tmp_path, 5, {"sig": (t, y)})

    lo_ms, hi_ms = 10 * 1000.0 / 1024.0, 20 * 1000.0 / 1024.0
    t_ms, x, _ = FoundationSource(data_dir=tmp_path).fetch(5, "sig/0", (lo_ms, hi_ms))
    assert np.array_equal(x, np.arange(10, 21, dtype=float))  # both endpoints in
    assert t_ms[0] == pytest.approx(lo_ms) and t_ms[-1] == pytest.approx(hi_ms)


def test_fetch_tlim_outside_data_returns_empty(tmp_path):
    from tokeye.sources.foundation import FoundationSource

    _ramp_shot(tmp_path, 6, n=1000)
    t_ms, x, fs = FoundationSource(data_dir=tmp_path).fetch(
        6, "mirnov/00", (50_000.0, 60_000.0)
    )
    assert t_ms.size == 0 and x.size == 0 and fs == 0.0


# ── degenerate / error paths ──────────────────────────────────────────────────────
def test_fetch_empty_group_returns_empty(tmp_path):
    from tokeye.sources.foundation import FoundationSource, time_bounds

    _write_shot(tmp_path, 7, {"bes": (np.zeros(1), np.zeros((64, 1)))})
    t_ms, x, fs = FoundationSource(data_dir=tmp_path).fetch(7, "bes/00")
    assert t_ms.size == 0 and x.size == 0 and fs == 0.0
    assert time_bounds(7, "bes/00", data_dir=tmp_path) is None


def test_fetch_errors_name_the_problem(tmp_path):
    from tokeye.sources.foundation import FoundationSource

    _ramp_shot(tmp_path, 8, n=100, n_ch=3)
    src = FoundationSource(data_dir=tmp_path)

    with pytest.raises(ValueError, match="no foundation_model file for shot 999"):
        src.fetch(999, "mirnov/00")
    with pytest.raises(ValueError, match="available: mirnov"):
        src.fetch(8, "ece/00")
    with pytest.raises(ValueError, match="channels 0–2"):
        src.fetch(8, "mirnov/3")


def test_fetch_nonuniform_timebase_falls_back(tmp_path):
    from tokeye.sources.foundation import FoundationSource

    # Rate change halfway through: the midpoint spot-check must catch it and
    # fall back to reading the full time base (with a warning).
    t = np.concatenate(
        [np.arange(500) * 1e-3, 0.5 + np.arange(500) * 2e-3]
    )
    _write_shot(tmp_path, 9, {"sig": (t, t[None, :])})
    with pytest.warns(UserWarning, match="not uniform"):
        t_ms, x, fs = FoundationSource(data_dir=tmp_path).fetch(9, "sig/0")
    assert x.size == 1000
    assert t_ms[-1] == pytest.approx(t[-1] * 1e3, rel=1e-5)
    assert fs > 0


# ── directory scanning ────────────────────────────────────────────────────────────
def test_list_shots_ignores_non_shot_entries(tmp_path):
    from tokeye.sources.foundation import list_shots

    _ramp_shot(tmp_path, 185601, n=10)
    _ramp_shot(tmp_path, 204999, n=10)
    (tmp_path / "data").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "notes.h5").touch()
    (tmp_path / "123_processed.h5.bak").touch()

    assert list_shots(tmp_path) == [185601, 204999]
    assert list_shots(tmp_path / "does-not-exist") == []


def test_latest_shot_caches_scan(tmp_path):
    import tokeye.sources.foundation as fnd

    fnd._LATEST_CACHE.clear()
    _ramp_shot(tmp_path, 190000, n=10)
    assert fnd.latest_shot(tmp_path) == 190000

    # Within the TTL the scan is cached: a newer file is not seen yet.
    _ramp_shot(tmp_path, 190001, n=10)
    assert fnd.latest_shot(tmp_path) == 190000
    fnd._LATEST_CACHE.clear()
    assert fnd.latest_shot(tmp_path) == 190001


# ── time_bounds ───────────────────────────────────────────────────────────────────
def test_time_bounds_reads_endpoints_and_caches(tmp_path):
    import tokeye.sources.foundation as fnd

    fnd._BOUNDS_CACHE.clear()
    _ramp_shot(tmp_path, 10, n=50_000)
    b = fnd.time_bounds(10, "mirnov/05", data_dir=tmp_path)
    assert b is not None
    t0, t1 = b
    assert t0 == pytest.approx(-4336.0, abs=0.01)
    assert t1 > t0

    # Cached per (dir, shot, group): survives the file disappearing.
    (tmp_path / "10_processed.h5").unlink()
    assert fnd.time_bounds(10, "mirnov/05", data_dir=tmp_path) == b
    assert fnd.time_bounds(11, "mirnov/00", data_dir=tmp_path) is None


# ── env override ──────────────────────────────────────────────────────────────────
def test_foundation_dir_env_override(tmp_path, monkeypatch):
    from tokeye.sources.foundation import (
        DEFAULT_FOUNDATION_DIR,
        FoundationSource,
        foundation_dir,
    )

    monkeypatch.delenv("TOKEYE_FOUNDATION_DIR", raising=False)
    assert foundation_dir() == DEFAULT_FOUNDATION_DIR
    assert FoundationSource().data_dir == DEFAULT_FOUNDATION_DIR

    monkeypatch.setenv("TOKEYE_FOUNDATION_DIR", str(tmp_path))
    assert foundation_dir() == str(tmp_path)
    assert FoundationSource().data_dir == str(tmp_path)

    _ramp_shot(tmp_path, 12, n=100)
    t_ms, x, _ = FoundationSource().fetch(12, "mirnov/00")
    assert x.size == 100


# ── list_signals + per-shot presets ───────────────────────────────────────────────
def test_list_signals_skips_empty_groups(tmp_path):
    from tokeye.sources.foundation import list_signals

    n = 100
    t = np.arange(n) * 2e-6
    _write_shot(
        tmp_path,
        13,
        {
            "mirnov": (t, np.zeros((4, n))),
            "bes": (np.zeros(1), np.zeros((64, 1))),  # absent this shot
            "tangtv": (t, np.zeros((2, n, 3, 4))),  # camera video, not a signal
        },
    )
    assert list_signals(13, tmp_path) == {"mirnov": (4, n)}
    assert list_signals(999, tmp_path) is None


def test_fetch_rejects_non_signal_group(tmp_path):
    from tokeye.sources.foundation import FoundationSource

    n = 50
    t = np.arange(n) * 1e-3
    _write_shot(tmp_path, 15, {"tangtv": (t, np.zeros((2, n, 3, 4)))})
    with pytest.raises(ValueError, match="camera video"):
        FoundationSource(data_dir=tmp_path).fetch(15, "tangtv/0")


def test_foundation_presets_shape():
    from tokeye.sources.foundation_presets import (
        FOUNDATION_DIAGNOSTICS,
        foundation_dropdown_choices,
    )

    mirnov = FOUNDATION_DIAGNOSTICS["mirnov"]
    assert len(mirnov.pointnames) == 29
    assert mirnov.pointnames[0] == "mirnov/00"
    assert mirnov.default == "mirnov/00"
    assert mirnov.verified is True
    assert "identity" in mirnov.note

    assert FOUNDATION_DIAGNOSTICS["sxr"].pointnames[0] == "sxr/000"  # 3-digit pad
    assert len(FOUNDATION_DIAGNOSTICS["sxr"].pointnames) == 320
    assert foundation_dropdown_choices()[0] == (mirnov.label, "mirnov")
    for diag in FOUNDATION_DIAGNOSTICS.values():
        assert diag.default in diag.pointnames


def test_signals_for_shot_matches_file_and_falls_back(tmp_path):
    from tokeye.sources.foundation_presets import (
        FOUNDATION_DIAGNOSTICS,
        signals_for_shot,
    )

    n = 100
    t = np.arange(n) * 2e-6
    _write_shot(
        tmp_path,
        14,
        {"mirnov": (t, np.zeros((4, n))), "weird": (t, np.zeros((2, n)))},
    )
    diags = signals_for_shot(14, tmp_path)
    assert set(diags) == {"mirnov", "weird"}
    assert diags["mirnov"].pointnames == ("mirnov/00", "mirnov/01", "mirnov/02", "mirnov/03")
    assert diags["weird"].label == "weird"  # unknown group still selectable

    # Missing shot file -> static presets.
    assert signals_for_shot(999, tmp_path) is FOUNDATION_DIAGNOSTICS
