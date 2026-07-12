"""Tests for the Princeton app tab (requires the `app` extra / gradio + plotly).

Handlers are exercised with in-memory arrays and tiny synthetic shot files in
``tmp_path`` (pointed at via ``TOKEYE_FOUNDATION_DIR``) — no GPFS, no model.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest


def test_princeton_tab_import_is_h5py_free():
    # Importing the tab must not import h5py (deferred to Load) or MDSplus.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import tokeye.app.tabs.princeton, sys; "
            "assert 'h5py' not in sys.modules; "
            "assert 'MDSplus' not in sys.modules",
        ],
        check=True,
    )


def test_create_app_registers_princeton_tab():
    """Building the app wires the Princeton tab with no file I/O / model load."""
    from tokeye.app.__main__ import create_app

    app = create_app()

    assert app is not None
    labels = {getattr(b, "label", None) for b in getattr(app, "blocks", {}).values()}
    assert "Princeton" in labels
    assert "Analyze" in labels and "Annotate" in labels and "Utilities" in labels


def _write_shot(root, shot, groups):
    h5py = pytest.importorskip("h5py")
    path = root / f"{shot}_processed.h5"
    with h5py.File(path, "w") as f:
        for name, (x, y) in groups.items():
            g = f.create_group(name)
            g.create_dataset("xdata", data=np.asarray(x, dtype=np.float32))
            g.create_dataset("ydata", data=np.asarray(y, dtype=np.float32))
    return path


def test_pointname_update_follows_shot_file(tmp_path, monkeypatch):
    """The channel dropdown lists exactly the selected shot's channels."""
    from tokeye.app.tabs.princeton import _pointname_update

    monkeypatch.setenv("TOKEYE_FOUNDATION_DIR", str(tmp_path))
    n = 100
    t = np.arange(n) * 2e-6
    _write_shot(tmp_path, 42, {"mirnov": (t, np.zeros((3, n)))})

    dd = _pointname_update("mirnov", 42)
    assert dd.constructor_args["choices"] == ["mirnov/00", "mirnov/01", "mirnov/02"]
    assert dd.constructor_args["value"] == "mirnov/00"

    # No shot / missing file -> static presets (29 mirnov channels).
    dd_static = _pointname_update("mirnov", None)
    assert len(dd_static.constructor_args["choices"]) == 29
    dd_missing = _pointname_update("mirnov", 999)
    assert len(dd_missing.constructor_args["choices"]) == 29

    # Unknown diagnostic -> empty dropdown, no crash.
    dd_unknown = _pointname_update("nope", 42)
    assert dd_unknown.constructor_args["choices"] == []


def test_fill_window_autofills_from_archive(tmp_path, monkeypatch):
    import tokeye.sources.foundation as fnd
    from tokeye.app.tabs.princeton import fill_window

    monkeypatch.setenv("TOKEYE_FOUNDATION_DIR", str(tmp_path))
    fnd._BOUNDS_CACHE.clear()
    n = 1000
    t = -0.5 + np.arange(n, dtype=np.float64) * 1e-3
    _write_shot(tmp_path, 43, {"mirnov": (t, np.zeros((2, n)))})

    t0, t1 = fill_window(43, "mirnov/01")
    assert t0 == pytest.approx(-500.0, abs=0.01)
    assert t1 == pytest.approx(499.0, abs=0.01)

    # Missing shot / no selection: leave the fields untouched (gr.update()).
    upd = fill_window(None, "mirnov/01")
    assert all(not u.get("value") for u in upd)


def test_load_shot_without_shot_warns_and_returns_none():
    """Guard path: no shot -> a warning + (None, None), never touching a file."""
    from tokeye.app.tabs.princeton import load_shot

    with pytest.warns(UserWarning):
        spec, meta = load_shot(
            None, "mirnov", None, None, None, 256, 64, True, 1.0, 99.0, 1
        )

    assert spec is None
    assert meta is None


def test_load_shot_reads_archive_and_records_meta(tmp_path, monkeypatch):
    from tokeye.app.tabs.princeton import load_shot

    monkeypatch.setenv("TOKEYE_FOUNDATION_DIR", str(tmp_path))
    n = 20_000
    fs = 100_000.0
    t = np.arange(n, dtype=np.float64) / fs
    rng = np.random.default_rng(0)
    _write_shot(tmp_path, 44, {"mirnov": (t, rng.standard_normal((2, n)))})

    spec, meta = load_shot(
        44, "mirnov", "mirnov/01", None, None, 256, 64, True, 1.0, 99.0, 1
    )
    assert spec is not None
    assert meta["fs"] == pytest.approx(fs, abs=1.0)
    assert meta["t0_ms"] == pytest.approx(0.0, abs=1e-6)
    assert meta["n_fft"] == 256 and meta["hop"] == 64

    # A missing group surfaces as a warning + (None, None), not an exception.
    with pytest.warns(UserWarning):
        spec_bad, meta_bad = load_shot(
            44, "ece", "ece/00", None, None, 256, 64, True, 1.0, 99.0, 1
        )
    assert spec_bad is None and meta_bad is None


def test_export_princeton_analysis_writes_npz_with_slugged_name():
    """Happy path: source='princeton' bundle; the group/index '/' never reaches
    the filename."""
    from pathlib import Path

    from tokeye.app.tabs.princeton import export_princeton_analysis

    rng = np.random.default_rng(0)
    spectrogram = rng.random((128, 60)).astype("float32")
    mask = rng.random((2, 128, 60)).astype("float32")
    stft_meta = {
        "fs": 5.0e5, "t0_ms": -4336.0, "n_fft": 256, "hop": 64, "clip_dc": True,
    }

    path = export_princeton_analysis(
        190000,          # shot
        "mirnov/07",     # pointname (contains '/')
        "big_tf_unet",   # model_file
        spectrogram,     # signal_transform
        stft_meta,       # stft_meta
        mask,            # inference_output
        256,             # n_fft
        64,              # hop_length
        True,            # clip_dc
        1.0,             # clip_low
        99.0,            # clip_high
        1,               # decimation
        5.0,             # stft_fmin
        250.0,           # stft_fmax
        0.5,             # threshold
        "Enhanced",      # view_mode
    )

    assert path is not None
    assert Path(path).name == "190000_mirnov-07_analysis.npz"

    data = np.load(path, allow_pickle=False)
    assert str(data["schema"]) == "tokeye-analysis/v1"
    assert str(data["source"]) == "princeton"
    for key in ("spectrogram", "mask", "time_ms", "freq_khz", "params_json"):
        assert key in data


def test_export_princeton_analysis_no_data_warns_and_returns_none():
    """No-data path: a warning + None (not gr.update()) so the download slot clears."""
    from tokeye.app.tabs.princeton import export_princeton_analysis

    with pytest.warns(UserWarning):
        result = export_princeton_analysis(
            None, None, "big_tf_unet", None, None, None,
            256, 64, True, 1.0, 99.0, 1, 5.0, 250.0, 0.5, "Enhanced",
        )

    assert result is None
