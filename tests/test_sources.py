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


def test_cache_root_env_override(monkeypatch):
    from tokeye.sources import DEFAULT_CACHE_ROOT, MDSSource, cache_root

    monkeypatch.delenv("TOKEYE_CACHE", raising=False)
    assert cache_root() == DEFAULT_CACHE_ROOT
    assert MDSSource().data_dir == DEFAULT_CACHE_ROOT

    monkeypatch.setenv("TOKEYE_CACHE", "/tmp/tokeye-cache-test")
    assert cache_root() == "/tmp/tokeye-cache-test"
    assert MDSSource().data_dir == "/tmp/tokeye-cache-test"
