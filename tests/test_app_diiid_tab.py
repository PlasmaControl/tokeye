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


def test_offline_default_outdir_honors_runs_dir_env(monkeypatch):
    """TOKEYE_RUNS_DIR overrides the default output folder; unset -> per-user path."""
    from tokeye.app.tabs import diiid_offline

    monkeypatch.setenv("TOKEYE_RUNS_DIR", "/custom/runs")
    assert diiid_offline._default_outdir() == "/custom/runs"

    monkeypatch.delenv("TOKEYE_RUNS_DIR", raising=False)
    assert diiid_offline._default_outdir().endswith("/tokeye/data/runs")


def test_offline_slurm_defaults_honor_env(monkeypatch):
    """The Slurm field defaults read TOKEYE_SLURM_* at call time; unset -> today's values."""
    from tokeye.app.tabs import diiid_offline

    monkeypatch.setenv("TOKEYE_SLURM_PARTITION", "preemptable")
    assert diiid_offline._default_partition() == "preemptable"

    monkeypatch.delenv("TOKEYE_SLURM_PARTITION", raising=False)
    monkeypatch.delenv("TOKEYE_SLURM_GRES", raising=False)
    monkeypatch.delenv("TOKEYE_SLURM_TIME", raising=False)
    assert diiid_offline._default_partition() == "gpus"
    assert diiid_offline._default_gres() == "gpu:v100:1"
    assert diiid_offline._default_time() == "0-02:00:00"


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
        # shared control-room dark palette (not stock plotly_dark blue-grey)
        assert fig.layout.paper_bgcolor == "#13151a"
        assert fig.layout.plot_bgcolor == "#0c0d11"
        assert fig.layout.modebar.activecolor == "#45b8cb"

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
    # shared control-room dark palette (suppressed bins read as the plot canvas)
    assert fig.layout.paper_bgcolor == "#13151a"
    assert fig.layout.plot_bgcolor == "#0c0d11"
    assert fig.layout.modebar.activecolor == "#45b8cb"

    # Offline path stays matplotlib (Kaleido/Chromium absent on compute nodes).
    png = render_modespec_png(result, shot=190000)
    assert png is not None
    assert png.size[0] > 100


def _fake_modespec_result_with_mode_amp():
    """Live-style mode-spectrogram result including ``mode_amp`` (required by the
    CSV path's vendored ``detect_modes``). Shape cribbed from
    ``tests/test_export.py``."""
    import numpy as np

    rng = np.random.default_rng(0)
    result = {
        "t_win_ms": np.linspace(1000, 1020, 30),
        "freq_khz": np.linspace(5, 150, 25),
        "n_dominant": rng.integers(-3, 4, size=(30, 25)),
        "coherence": rng.random((30, 25)),
        "n_range": (-3, 3),
        "c95": 0.3,
    }
    n_lo, n_hi = result["n_range"]
    n_win, n_freq = result["n_dominant"].shape
    result["mode_amp"] = {
        n: np.ones((n_win, n_freq)) for n in range(int(n_lo), int(n_hi) + 1)
    }
    return result


def test_export_modespec_writes_npz_and_csv():
    """Happy path (gate off -> nd None): a 2-file list — a modespec .npz + a modes
    .csv — with diiid-batch filenames and the vendored CSV header."""
    from pathlib import Path

    import numpy as np

    from tokeye.app.tabs.diiid_modespec import export_modespec

    result = _fake_modespec_result_with_mode_amp()
    paths = export_modespec(
        190000,           # shot
        "MPI66M067D",     # ref_probe
        1000.0,           # t_min
        1020.0,           # t_max
        result,           # result_state
        None,             # tok_mask_state
        None,             # gate_meta_state
        5.0,              # f_min
        150.0,            # f_max
        -3,               # n_min
        3,                # n_max
        1,                # decimation
        0.5,              # coh_thresh
        False,            # gate
        "Array average",  # gate_source
        "big_tf_unet",    # model_file
    )

    assert isinstance(paths, list)
    assert len(paths) == 2
    npz_path, csv_path = paths
    assert Path(npz_path).name == "190000_modespec.npz"
    assert Path(csv_path).name == "190000_modes.csv"

    data = np.load(npz_path, allow_pickle=False)
    assert str(data["schema"]) == "tokeye-modespec/v1"
    assert str(data["source"]) == "diiid-modespec"
    for key in (
        "n_dominant", "coherence", "t_win_ms", "freq_khz", "n_range", "c95",
        "coh_thresh", "params_json",
    ):
        assert key in data

    header = Path(csv_path).read_text().splitlines()[0]
    assert "array" in header
    assert "mode_label" in header


def test_export_modespec_no_data_warns_and_returns_none():
    """No-data path: a warning + None (not gr.update()) so the download slot clears."""
    import pytest

    from tokeye.app.tabs.diiid_modespec import export_modespec

    with pytest.warns(UserWarning):
        result = export_modespec(
            190000, "MPI66M067D", None, None,
            None, None, None,
            5.0, 150.0, -3, 3, 1, 0.5, False, "Array average", "big_tf_unet",
        )

    assert result is None


def test_rerender_coh_surfaces_gate_failure(monkeypatch):
    """Gate-failure path: when ``gate_dominant_mask`` raises, ``rerender_coh`` warns
    (no longer silent) and still returns the ungated figure (not None)."""
    import numpy as np
    import pytest

    import tokeye.sources.mirnov as mirnov
    from tokeye.app.tabs.diiid_modespec import rerender_coh

    def _boom(*args, **kwargs):
        raise RuntimeError("gate exploded")

    # rerender_coh does `from tokeye.sources.mirnov import gate_dominant_mask`
    # inside the function, so patch the attribute on the module it re-imports from.
    monkeypatch.setattr(mirnov, "gate_dominant_mask", _boom)

    rng = np.random.default_rng(0)
    result = {
        "t_win_ms": np.linspace(1000, 1020, 30),
        "freq_khz": np.linspace(5, 150, 25),
        "n_dominant": rng.integers(-3, 4, size=(30, 25)),
        "coherence": rng.random((30, 25)),
        "n_range": (-3, 3),
        "c95": 0.3,
    }
    tok_mask = np.ones((128, 60), dtype=bool)
    gate_meta = {"fs": 2.0e6, "t0_ms": 1000.0, "n_fft": 256, "hop": 64, "clip_dc": True}

    with pytest.warns(UserWarning):
        fig = rerender_coh(result, tok_mask, gate_meta, 0.5, True)

    assert fig is not None
    assert fig.to_json()  # serialisable Plotly figure, not None
