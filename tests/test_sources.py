"""Tests for the tokeye.sources layer.

Offline only — live MDSplus/atlas fetches are untestable off the GA cluster, so
these exercise the cache path (with a seeded pickle) and the preset registry.
"""

from __future__ import annotations

import pickle
import subprocess
import sys

import numpy as np


def test_import_does_not_require_mdsplus():
    # Importing the package must not import MDSplus: MDSSource defers the
    # (import-guarded) MDS fetchers to .fetch(). Checked in a fresh subprocess
    # so the result is independent of what other tests have already imported.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import tokeye.sources, sys; assert 'MDSplus' not in sys.modules",
        ],
        check=True,
    )


def test_presets_defaults_are_valid():
    from tokeye.sources import DIAGNOSTICS

    assert "mag" in DIAGNOSTICS
    assert DIAGNOSTICS["mag"].verified is True
    for diag in DIAGNOSTICS.values():
        assert diag.pointnames, f"{diag.key} has no pointnames"
        assert diag.default in diag.pointnames


def test_presets_cover_pyspecview_diagnostics():
    from tokeye.sources import DIAGNOSTICS
    from tokeye.sources.presets import MIRNOV_TOROIDAL, MIRNOV_TOROIDAL_ANGLES

    # b1-b8 + the other magnetics/diagnostics are wired in.
    assert set(DIAGNOSTICS) >= {"mag", "mag_pol", "mhr", "ece", "co2", "bes"}
    assert DIAGNOSTICS["mhr"].pointnames == tuple(f"B{i}" for i in range(1, 9))
    assert len(DIAGNOSTICS["ece"].pointnames) == 40
    assert len(DIAGNOSTICS["bes"].pointnames) == 40
    assert len(DIAGNOSTICS["mag_pol"].pointnames) == 31
    # Toroidal angles are index-aligned to the probe names (needed for modespec).
    assert len(MIRNOV_TOROIDAL) == len(MIRNOV_TOROIDAL_ANGLES) == 14


def test_latest_shot_uses_current_shot_tdi(monkeypatch):
    import sys
    import types

    from tokeye.sources import latest_shot

    calls = {}

    class _FakeConn:
        def __init__(self, server):
            calls["server"] = server

        def get(self, expr):
            calls["expr"] = expr
            return 190904

    fake = types.ModuleType("MDSplus")
    fake.Connection = _FakeConn
    monkeypatch.setitem(sys.modules, "MDSplus", fake)

    assert latest_shot("atlas.gat.com") == 190904
    assert calls["server"] == "atlas.gat.com"
    assert 'current_shot("d3d")' in calls["expr"]


def test_latest_shot_none_when_mdsplus_missing(monkeypatch):
    import builtins

    from tokeye.sources import latest_shot

    real_import = builtins.__import__

    def _no_mds(name, *a, **k):
        if name == "MDSplus":
            raise ImportError("no MDSplus")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_mds)
    assert latest_shot() is None


def _seed_cache(data_dir, shot, key, x, t_ms):
    """Write the {data_dir}/{shot}/{shot}_{key}.pkl layout data_utils expects."""
    shot_dir = data_dir / str(shot)
    shot_dir.mkdir(parents=True, exist_ok=True)
    with (shot_dir / f"{shot}_{key}.pkl").open("wb") as f:
        pickle.dump((x, t_ms), f)


def test_fetch_cache_hit_derives_fs_and_skips_mdsplus(tmp_path, monkeypatch):
    from tokeye.sources import MDSSource

    shot, key = 190904, "MPI66M067D"
    t_ms = np.arange(0.0, 10.0, 0.005)  # 0.005 ms step -> fs = 200 kHz
    x = np.sin(t_ms)
    _seed_cache(tmp_path, shot, key, x, t_ms)

    # A cache hit must not call the live fetcher.
    import tokeye.modespec.classic.data_utils as du

    def _boom(*_a, **_k):
        raise AssertionError("cache hit must not call fetch_ptdata")

    monkeypatch.setattr(du, "fetch_ptdata", _boom)

    t_out, x_out, fs = MDSSource(data_dir=tmp_path).fetch(shot, key)

    assert x_out.size == x.size
    assert t_out.size == t_ms.size
    assert abs(fs - 200_000.0) < 10.0


def test_fetch_tlim_crops(tmp_path):
    from tokeye.sources import MDSSource

    shot, key = 1, "P"
    t_ms = np.arange(0.0, 100.0, 1.0)
    _seed_cache(tmp_path, shot, key, t_ms.copy(), t_ms)

    t_out, x_out, _ = MDSSource(data_dir=tmp_path).fetch(shot, key, tlim=(10.0, 20.0))

    assert t_out.min() >= 10.0
    assert t_out.max() <= 20.0
    assert x_out.size == t_out.size


def test_fetch_mirnov_cached_assembles_from_cache(tmp_path, monkeypatch):
    from tokeye.sources.mirnov import fetch_mirnov_cached
    from tokeye.sources.presets import MIRNOV_TOROIDAL

    shot = 190904
    t_ms = np.arange(0.0, 5.0, 0.005)  # 200 kHz
    for name in MIRNOV_TOROIDAL:
        _seed_cache(tmp_path, shot, name, np.sin(t_ms), t_ms)

    import tokeye.modespec.classic.data_utils as du

    monkeypatch.setattr(
        du, "fetch_ptdata", lambda *a, **k: (_ for _ in ()).throw(AssertionError("live"))
    )

    signals, t_out, angles, names = fetch_mirnov_cached(
        shot, "toroidal", data_dir=str(tmp_path)
    )
    assert signals.shape == (14, t_ms.size)
    assert angles.shape == (14,)
    assert len(names) == 14
    assert t_out.size == t_ms.size


def test_gate_dominant_intersects_mask_and_coherence():
    from tokeye.sources.mirnov import gate_dominant

    rng = np.random.default_rng(1)
    n_win, n_freq = 40, 30
    result = {
        "t_win_ms": np.linspace(1000, 1020, n_win),
        "freq_khz": np.linspace(5, 150, n_freq),
        "n_dominant": rng.integers(-3, 4, size=(n_win, n_freq)),
        "coherence": rng.random((n_win, n_freq)),
        "n_range": (-3, 3),
        "c95": 0.3,
    }
    arr_extract = rng.random((2, 512, 400)).astype("float32")
    meta = {"fs": 2.0e6, "t0_ms": 1000.0, "n_fft": 1024, "hop": 256, "clip_dc": True}

    nd = gate_dominant(result, arr_extract, meta, mask_threshold=0.5, coh_thresh=0.3)
    assert nd.shape == (n_win, n_freq)
    # A random mask + coherence gate keeps some bins and suppresses others.
    assert np.isnan(nd).any()
    assert np.isfinite(nd).any()


def test_gate_dominant_requires_fs():
    import pytest

    from tokeye.sources.mirnov import gate_dominant

    result = {
        "t_win_ms": np.array([1.0]),
        "freq_khz": np.array([10.0]),
        "n_dominant": np.array([[1]]),
        "coherence": np.array([[0.9]]),
        "n_range": (-1, 1),
        "c95": 0.3,
    }
    with pytest.raises(ValueError):
        gate_dominant(result, np.zeros((2, 8, 8)), {"fs": 0.0})


def test_cache_root_env_override(monkeypatch):
    from tokeye.sources import DEFAULT_CACHE_ROOT, MDSSource, cache_root

    monkeypatch.delenv("TOKEYE_CACHE", raising=False)
    assert cache_root() == DEFAULT_CACHE_ROOT
    assert MDSSource().data_dir == DEFAULT_CACHE_ROOT

    monkeypatch.setenv("TOKEYE_CACHE", "/tmp/tokeye-cache-test")
    assert cache_root() == "/tmp/tokeye-cache-test"
    assert MDSSource().data_dir == "/tmp/tokeye-cache-test"
