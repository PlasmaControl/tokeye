"""Tests for the DIII-D app tab (requires the `app` extra / gradio)."""

from __future__ import annotations

import subprocess
import sys


def test_diiid_tab_import_is_mdsplus_free():
    # Importing the tab (hence tokeye.sources) must not import MDSplus — MDS
    # access is deferred to the "Load shot" click. Subprocess = order-independent.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import tokeye.app.tabs.diiid, sys; assert 'MDSplus' not in sys.modules",
        ],
        check=True,
    )


def test_create_app_registers_diiid_tabs():
    """Building the app wires both DIII-D tabs with no model load / no network."""
    from tokeye.app.__main__ import create_app

    app = create_app()

    assert app is not None
    labels = {getattr(b, "label", None) for b in getattr(app, "blocks", {}).values()}
    assert "DIII-D" in labels
    assert "DIII-D Offline" in labels


def test_load_shot_without_shot_warns_and_returns_none():
    """Guard path: no shot -> a warning + (None, None), never touching MDSplus."""
    import pytest

    from tokeye.app.tabs.diiid import load_shot

    transform_args = {
        "n_fft": 256,
        "hop_length": 64,
        "clip_dc": True,
        "percentile_low": 1.0,
        "percentile_high": 99.0,
    }
    with pytest.warns(UserWarning):
        spec, meta = load_shot(None, "mag", None, None, None, transform_args)

    assert spec is None
    assert meta is None


def test_render_view_and_modespec_produce_images():
    """Renderers work on synthetic arrays with no network (offline / CI-safe)."""
    import numpy as np

    from tokeye.sources.viz import render_modespec, render_view

    rng = np.random.default_rng(0)
    arr = rng.random((256, 300)).astype("float32")
    ext = rng.random((2, 256, 300)).astype("float32")
    meta = {"fs": 2.0e6, "t0_ms": 1000.0, "n_fft": 512, "hop": 256, "clip_dc": True}

    for view in ("Original", "Enhanced", "Mask", "Amplitude"):
        img = render_view(view, arr, ext, True, True, 0, 100, 0.5, meta)
        assert img is not None
        assert img.size[0] > 100

    result = {
        "t_win_ms": np.linspace(1000, 1020, 30),
        "freq_khz": np.linspace(5, 150, 25),
        "n_dominant": rng.integers(-3, 4, size=(30, 25)),
        "coherence": rng.random((30, 25)),
        "n_range": (-3, 3),
        "c95": 0.3,
    }
    assert render_modespec(result, shot=190000) is not None
