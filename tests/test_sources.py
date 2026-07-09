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


class _Wrap:
    """Minimal stand-in for an MDSplus data object (``.data()``)."""

    def __init__(self, v):
        self._v = v

    def data(self):
        return self._v


def _install_fake_mdsplus(monkeypatch, conn_cls):
    import types

    fake = types.ModuleType("MDSplus")
    fake.Connection = conn_cls
    monkeypatch.setitem(sys.modules, "MDSplus", fake)


# ── CO2 fetch (all-zeros PTDATA trap → real BCI.DPD / segmented BCI source) ──────
def test_co2_preset_uses_real_chord_names():
    from tokeye.sources import DIAGNOSTICS
    from tokeye.sources.co2 import CO2_CHORDS, is_co2_chord

    assert DIAGNOSTICS["co2"].pointnames == ("DENV1_UF", "DENV2_UF", "DENV3_UF", "DENR0_UF")
    assert DIAGNOSTICS["co2"].default == "DENV2_UF"
    assert set(DIAGNOSTICS["co2"].pointnames) == set(CO2_CHORDS)
    assert is_co2_chord("denv2_uf") and not is_co2_chord("MPI66M067D")


def test_fetch_co2_chord_prefers_nonzero_dpd(monkeypatch):
    from tokeye.sources.co2 import fetch_co2_chord

    t = np.linspace(1000.0, 2000.0, 500)
    z = np.ones(500)

    class _Conn:
        def __init__(self, server):
            pass

        def openTree(self, *a):
            pass

        def closeAllTrees(self):
            pass

        def get(self, expr):
            return _Wrap(t if expr.startswith("dim_of(") else z)

    _install_fake_mdsplus(monkeypatch, _Conn)
    data, t_ms = fetch_co2_chord(190904, "DENV2_UF")
    assert np.count_nonzero(data) == 500
    assert t_ms[0] == 1000.0 and t_ms[-1] == 2000.0


def test_fetch_co2_chord_falls_back_to_bci_segments(monkeypatch):
    from tokeye.sources.co2 import fetch_co2_chord

    dpd_z = np.zeros(10)  # DPD present but all-zero -> must fall back to BCI
    segments = {
        0: (np.linspace(1000.0, 1500.0, 300), np.ones(300)),
        1: (np.linspace(1500.0, 2000.0, 200), 2 * np.ones(200)),
    }

    class _Conn:
        def __init__(self, server):
            self._cur = None

        def openTree(self, *a):
            pass

        def closeAllTrees(self):
            pass

        def get(self, expr):
            e = expr.strip()
            if e.startswith("findsig("):
                n = int(e.split('"')[1].rsplit("_", 1)[1])
                if n not in segments:
                    raise RuntimeError("%TREE-W-NNF")
                return _Wrap(f"\\TAG::seg_{n}")
            if e == "_fstree":
                return _Wrap("bci")
            if e.startswith("_s ="):
                self._cur = int(e.split("=", 1)[1].strip().rsplit("_", 1)[1])
                return _Wrap(segments[self._cur][1])
            if e == "dim_of(_s)":
                return _Wrap(segments[self._cur][0])
            if e.startswith("dim_of("):
                return _Wrap(np.arange(dpd_z.size, dtype=float))
            return _Wrap(dpd_z)

    _install_fake_mdsplus(monkeypatch, _Conn)
    data, t_ms = fetch_co2_chord(170008, "DENV2_UF")
    # segments concatenated in time order (seg0 then seg1)
    assert data.size == 500
    assert t_ms[0] == 1000.0 and t_ms[-1] == 2000.0
    assert data[0] == 1.0 and data[-1] == 2.0


def test_fetch_co2_chord_raises_when_all_zero(monkeypatch):
    import pytest

    from tokeye.sources.co2 import fetch_co2_chord

    class _Conn:
        def __init__(self, server):
            pass

        def openTree(self, *a):
            pass

        def closeAllTrees(self):
            pass

        def get(self, expr):
            if expr.strip().startswith("findsig("):
                raise RuntimeError("%TREE-W-NNF")  # no BCI segments
            if expr.startswith("dim_of("):
                return _Wrap(np.arange(10, dtype=float))
            return _Wrap(np.zeros(10))  # DPD all-zero

    _install_fake_mdsplus(monkeypatch, _Conn)
    with pytest.raises(RuntimeError):
        fetch_co2_chord(999999, "DENV2_UF")


def test_mds_fetch_routes_co2_pointnames(tmp_path, monkeypatch):
    import tokeye.modespec.classic.data_utils as du
    import tokeye.sources.co2 as co2
    from tokeye.sources import MDSSource

    t = np.linspace(1000.0, 2000.0, 500)
    z = np.ones(500)
    seen = {}

    def _fake_co2(shot, chord, atlas="atlas.gat.com"):
        seen["chord"] = chord
        return z, t

    monkeypatch.setattr(co2, "fetch_co2_chord", _fake_co2)
    monkeypatch.setattr(
        du, "fetch_ptdata",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("PTDATA used for CO2")),
    )
    t_out, x_out, fs = MDSSource(data_dir=tmp_path).fetch(190904, "DENV2_UF")
    assert seen["chord"] == "DENV2_UF"
    assert x_out.size == 500
    assert fs > 0  # derived from the returned time axis


# ── time_bounds (cheap scalar-TDI window for shot/probe autofill) ─────────────────
def test_time_bounds_single_round_trip(monkeypatch):
    import tokeye.sources.mds as mds
    from tokeye.sources import time_bounds

    mds._BOUNDS_CACHE.clear()

    class _Conn:
        def __init__(self, server):
            pass

        def openTree(self, *a):
            pass

        def closeAllTrees(self):
            pass

        def get(self, expr):
            # one call returns both endpoints of the (assigned-once) time base
            assert "DIM_OF" in expr and expr.strip().startswith("[")
            return _Wrap(np.array([1000.0, 2000.0]))

    _install_fake_mdsplus(monkeypatch, _Conn)
    assert time_bounds(190904, "MPI66M067D") == (1000.0, 2000.0)
    # second call is served from the in-process cache (no MDSplus needed)
    monkeypatch.setitem(sys.modules, "MDSplus", None)
    assert time_bounds(190904, "MPI66M067D") == (1000.0, 2000.0)


def test_time_bounds_none_on_failure(monkeypatch):
    import tokeye.sources.mds as mds
    from tokeye.sources import time_bounds

    mds._BOUNDS_CACHE.clear()

    class _Conn:
        def __init__(self, server):
            raise RuntimeError("atlas unreachable")

    _install_fake_mdsplus(monkeypatch, _Conn)
    assert time_bounds(1, "P") is None


# ── modespec decimation speedup ──────────────────────────────────────────────────
def test_decimation_factor_respects_band_and_user():
    from tokeye.sources.mirnov import decimation_factor

    assert decimation_factor(2.0e6, 200, None) == 4  # 2 MHz, 200 kHz band -> /4
    assert decimation_factor(5.0e5, 200, None) == 1  # already near Nyquist
    assert decimation_factor(2.0e6, 200, 8) == 8  # honor a larger user request
    assert decimation_factor(0.0, 200, None) == 1  # unknown fs -> no decimation


def test_maybe_decimate_reduces_samples():
    from tokeye.sources.mirnov import _fs_hz, _maybe_decimate

    fs, n = 2.0e6, 200_000
    t = np.arange(n) / fs * 1e3 + 1000.0
    sig = np.random.default_rng(0).standard_normal((14, n))
    sd, td = _maybe_decimate(sig, t, None, 200.0)
    assert sd.shape[1] == td.size
    assert sd.shape[1] < n
    assert _fs_hz(td) < _fs_hz(t)

    # Already at ~Nyquist for the band: no decimation.
    fs2, n2 = 5.0e5, 50_000
    t2 = np.arange(n2) / fs2 * 1e3
    sig2 = np.random.default_rng(0).standard_normal((14, n2))
    sd2, td2 = _maybe_decimate(sig2, t2, None, 200.0)
    assert sd2.shape[1] == n2


def test_cache_root_env_override(monkeypatch):
    from tokeye.sources import DEFAULT_CACHE_ROOT, MDSSource, cache_root

    monkeypatch.delenv("TOKEYE_CACHE", raising=False)
    assert cache_root() == DEFAULT_CACHE_ROOT
    assert MDSSource().data_dir == DEFAULT_CACHE_ROOT

    monkeypatch.setenv("TOKEYE_CACHE", "/tmp/tokeye-cache-test")
    assert cache_root() == "/tmp/tokeye-cache-test"
    assert MDSSource().data_dir == "/tmp/tokeye-cache-test"
