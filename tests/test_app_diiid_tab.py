"""Tests for the DIII-D app tabs (requires the `app` extra / gradio + plotly)."""

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


def test_diiid_modespec_tab_import_is_mdsplus_free():
    # Same guarantee for the self-contained DIII-D Modespec tab.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import tokeye.app.tabs.diiid_modespec, sys; assert 'MDSplus' not in sys.modules",
        ],
        check=True,
    )


def test_create_app_registers_diiid_tabs():
    """Building the app wires all three DIII-D tabs with no model load / no network."""
    from tokeye.app.__main__ import create_app

    app = create_app()

    assert app is not None
    labels = {getattr(b, "label", None) for b in getattr(app, "blocks", {}).values()}
    assert "DIII-D" in labels
    assert "DIII-D Modespec" in labels
    assert "DIII-D Offline" in labels


def test_diiid_tab_has_no_modespec_view():
    """Modespec moved to its own tab — the DIII-D tab is a pure spectrogram viewer."""
    from tokeye.app.tabs import diiid

    assert diiid._VIEW_MODES == ["Original", "Enhanced", "Mask", "Amplitude"]
    assert "Modespec" not in diiid._VIEW_MODES


def test_load_shot_without_shot_warns_and_returns_none():
    """Guard path: no shot -> a warning + (None, None), never touching MDSplus."""
    import pytest

    from tokeye.app.tabs.diiid import load_shot

    # New raw-components signature: the STFT knobs are passed straight through
    # (no transform_args dict), so a Load always reflects the live controls.
    with pytest.warns(UserWarning):
        spec, meta = load_shot(
            None, "mag", None, None, None, 256, 64, True, 1.0, 99.0, 1
        )

    assert spec is None
    assert meta is None


def test_export_diiid_analysis_writes_npz_with_real_axes():
    """Happy path: an analysis bundle with a mask and real kHz/ms axes lands on disk."""
    from pathlib import Path

    import numpy as np

    from tokeye.app.tabs.diiid import export_diiid_analysis

    rng = np.random.default_rng(0)
    spectrogram = rng.random((128, 60)).astype("float32")
    mask = rng.random((2, 128, 60)).astype("float32")
    stft_meta = {
        "fs": 2.0e6, "t0_ms": 1000.0, "n_fft": 256, "hop": 64, "clip_dc": True,
    }

    path = export_diiid_analysis(
        190000,          # shot
        "mpi66m307d",    # pointname
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
    assert Path(path).name.startswith("190000_mpi66m307d")

    data = np.load(path, allow_pickle=False)
    assert str(data["schema"]) == "tokeye-analysis/v1"
    assert str(data["source"]) == "diiid"
    for key in ("spectrogram", "mask", "time_ms", "freq_khz", "params_json"):
        assert key in data


def test_export_diiid_analysis_no_data_warns_and_returns_none():
    """No-data path: a warning + None (not gr.update()) so the download slot clears."""
    import pytest

    from tokeye.app.tabs.diiid import export_diiid_analysis

    with pytest.warns(UserWarning):
        result = export_diiid_analysis(
            None, None, "big_tf_unet", None, None, None,
            256, 64, True, 1.0, 99.0, 1, 5.0, 250.0, 0.5, "Enhanced",
        )

    assert result is None


def test_plotly_view_builds_interactive_figures():
    """plotly_view returns a serialisable Plotly figure with real kHz/ms axes."""
    import numpy as np

    from tokeye.sources.viz import plotly_view

    rng = np.random.default_rng(0)
    arr = rng.random((256, 300)).astype("float32")
    ext = rng.random((2, 256, 300)).astype("float32")
    # fmin/fmax present -> the image is frequency-cropped for display.
    meta = {
        "fs": 2.0e6, "t0_ms": 1000.0, "n_fft": 512, "hop": 256, "clip_dc": True,
        "fmin_khz": 20, "fmax_khz": 200,
    }

    for view in ("Original", "Enhanced", "Mask", "Amplitude"):
        fig = plotly_view(view, arr, ext, True, True, 0, 100, 0.5, meta)
        assert fig.to_json()  # serialisable for gr.Plot
        # spectrogram drawn as one stretched layout image (not an aspect-locked trace)
        assert len(fig.layout.images) == 1
        assert fig.layout.xaxis.title.text == "time (ms)"
        assert fig.layout.yaxis.title.text == "frequency (kHz)"
        # y-axis ascending -> low frequency sits at the bottom
        assert fig.layout.yaxis.range[0] < fig.layout.yaxis.range[1]

    # Missing inputs -> a valid (empty) figure, never a crash.
    assert plotly_view("Enhanced", None, None, True, True, 0, 100, 0.5, meta).to_json()


def test_plotly_modespec_is_discrete_heatmap_with_integer_colorbar():
    import numpy as np

    from tokeye.sources.viz import plotly_modespec, render_modespec_png

    rng = np.random.default_rng(0)
    result = {
        "t_win_ms": np.linspace(1000, 1020, 30),
        "freq_khz": np.linspace(5, 150, 25),
        "n_dominant": rng.integers(-3, 4, size=(30, 25)),
        "coherence": rng.random((30, 25)),
        "n_range": (-3, 3),
        "c95": 0.3,
    }
    fig = plotly_modespec(result)
    assert fig.data[0].type == "heatmap"
    assert fig.to_json()
    # discrete integer colorbar: one tick per mode number in [-3, 3]
    assert list(fig.data[0].colorbar.ticktext) == ["-3", "-2", "-1", "+0", "+1", "+2", "+3"]
    assert fig.data[0].zmin == -3.5 and fig.data[0].zmax == 3.5

    # Offline path stays matplotlib (Kaleido/Chromium absent on compute nodes).
    png = render_modespec_png(result, shot=190000)
    assert png is not None
    assert png.size[0] > 100
