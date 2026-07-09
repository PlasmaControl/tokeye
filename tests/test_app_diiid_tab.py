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


def test_create_app_registers_diiid_tab():
    """Building the app wires the DIII-D tab with no model load / no network."""
    from tokeye.app.__main__ import create_app

    app = create_app()

    assert app is not None
    labels = {getattr(b, "label", None) for b in getattr(app, "blocks", {}).values()}
    assert "DIII-D" in labels


def test_load_shot_without_shot_warns_and_returns_none():
    """Guard path: no shot -> a warning + None, and it never touches MDSplus."""
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
        result = load_shot(None, "mag", None, None, None, transform_args)

    assert result is None
