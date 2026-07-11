"""Qt/pyqtgraph tests for the spectrogram view (offscreen)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from tokeye.gui.render import spectrogram_rect

_META = {"fs": 500_000.0, "t0_ms": 2000.0, "n_fft": 512, "hop": 128, "clip_dc": True}


def test_view_sets_image_rect_and_orientation(qapp):
    from PySide6 import QtCore

    from tokeye.gui.widgets.spectrogram_view import SpectrogramView

    view = SpectrogramView(window=None)
    # disable the default 5–250 kHz band so the full image maps to the rect
    view.stft._fmin.setValue(0.0)
    view.stft._fmax.setValue(0.0)
    h, w = 200, 60
    arr = np.random.default_rng(0).random((h, w))
    view.set_spectrogram(arr, _META)

    img = view.canvas.image_item()
    expected = spectrogram_rect(_META, h, w, r_lo=0)  # no band crop
    vr = img.mapRectToView(img.boundingRect())
    assert (vr.x(), vr.y(), vr.width(), vr.height()) == pytest.approx(expected, rel=1e-6)

    # Orientation: setRect maps col->x and row->y with positive scales, so
    # low-freq row 0 sits at the bottom (no flipud). m11 = ms/col, m22 = kHz/row.
    transform = img.transform()
    assert transform.m11() > 0
    assert transform.m22() > 0
    # bottom-left corner (row 0, col 0) maps to (x_left, y_bottom)
    bottom_left = img.mapToView(QtCore.QPointF(0.0, 0.0))
    assert (bottom_left.x(), bottom_left.y()) == pytest.approx(expected[:2])


def test_axis_labels_have_no_si_prefix(qapp):
    from tokeye.gui.widgets.plot_items import SpectrogramCanvas

    canvas = SpectrogramCanvas()
    plot = canvas.spectrogram_plot()
    assert "Time (ms)" in plot.getAxis("bottom").labelString()
    assert "Frequency (kHz)" in plot.getAxis("left").labelString()
    assert plot.getAxis("bottom").autoSIPrefix is False
    assert plot.getAxis("left").autoSIPrefix is False


def test_cursor_readout_forwarded_to_window(qapp):
    from tokeye.gui.widgets.spectrogram_view import SpectrogramView

    seen = []

    class FakeWindow:
        def set_readout(self, text):
            seen.append(text)

    view = SpectrogramView(window=FakeWindow())
    view.set_spectrogram(np.random.default_rng(1).random((32, 16)), _META)
    # simulate a cursor sample landing inside the image
    view._on_cursor((2000.5, 40.0, 0.123))
    view._on_cursor(None)
    assert any("ms" in s and "kHz" in s for s in seen)
    assert "" in seen  # cleared on leave


def test_main_window_registers_both_views(qapp):
    from tokeye.gui.main_window import MainWindow

    win = MainWindow()
    assert set(win._views) == {"spectrogram", "modespec"}
    # switching views must not raise
    win.show_view("modespec")
    win.show_view("spectrogram")


def test_plot_toolbar_toggles_mouse_mode(qapp):
    import pyqtgraph as pg

    from tokeye.gui.widgets.spectrogram_view import SpectrogramView

    view = SpectrogramView(window=None)
    vb = view.canvas.spectrogram_plot().getViewBox()
    assert vb.state["mouseMode"] == pg.ViewBox.PanMode  # default: pan
    view._zoom_btn.setChecked(True)  # exclusive group unchecks pan -> rect mode
    assert vb.state["mouseMode"] == pg.ViewBox.RectMode
    view._pan_btn.setChecked(True)
    assert vb.state["mouseMode"] == pg.ViewBox.PanMode


def test_raw_strip_enables_on_data(qapp):
    from tokeye.gui.widgets.spectrogram_view import SpectrogramView

    view = SpectrogramView(window=None)
    assert not view._raw_btn.isEnabled()
    assert not view.canvas.has_raw()
    t = np.linspace(0.0, 10.0, 500)
    view.set_raw_signal(t, np.sin(t))
    assert view._raw_btn.isEnabled()
    assert view.canvas.has_raw()


def test_fetch_then_overlay_renders_rgb(qapp):
    from tokeye.gui.widgets.spectrogram_view import SpectrogramView

    view = SpectrogramView(window=None)
    h, w = 64, 40
    spec = np.random.default_rng(0).random((h, w))
    view._fetch_id = 7  # mark active so _on_result accepts it
    view._on_result(7, {"spec": spec, "meta": _META, "t": None, "x": None})
    assert view._spec is not None
    assert view.analyze_btn.isEnabled()  # spec present -> Analyze enabled

    # overlay mode before inference falls back to a scalar image
    view.view_controls._mode.setCurrentText("Mask")
    assert view.canvas.image_item().image.ndim == 2

    # supply a stub inference and re-render: Mask view is now a 3-channel overlay
    view._analyze_id = 8
    view._on_result(8, np.random.default_rng(1).random((2, h, w)).astype(float))
    assert view.canvas.image_item().image.ndim == 3


def test_stale_result_is_dropped(qapp):
    from tokeye.gui.widgets.spectrogram_view import SpectrogramView

    view = SpectrogramView(window=None)
    view._fetch_id = 5  # current request
    view._on_result(3, {"spec": np.zeros((4, 4)), "meta": None, "t": None, "x": None})
    assert view._spec is None  # stale id 3 ignored


def test_save_button_state_tracks_data(qapp):
    from tokeye.gui.widgets.spectrogram_view import SpectrogramView

    view = SpectrogramView(window=None)
    assert not view._save_btn.isEnabled()  # nothing loaded yet
    view.set_spectrogram(np.random.default_rng(0).random((16, 8)), _META)
    assert view._save_btn.isEnabled()  # spectrogram present


def test_export_png_and_npz(qapp, tmp_path):
    from PIL import Image

    from tokeye.gui.widgets.spectrogram_view import SpectrogramView

    view = SpectrogramView(window=None)
    h, w = 48, 32
    spec = np.random.default_rng(3).random((h, w))
    view.set_spectrogram(spec, _META)
    # Late-window absolute time base (t≈4000 ms, 2 kHz cadence): these exact
    # values do NOT survive float32, so the npz must keep raw_t_ms as float64.
    t = 4000.0 + np.arange(400, dtype=np.float64) * 0.0005
    x = np.sin(t)
    view.set_raw_signal(t, x)

    png = view.export_png(tmp_path / "v.png")
    assert png.exists() and png.stat().st_size > 0
    with Image.open(png) as im:
        assert im.width > 0 and im.height > 0

    npz = view.export_npz(tmp_path / "v")
    assert npz.exists()
    data = np.load(npz, allow_pickle=False)
    assert str(data["schema"]) == "tokeye-analysis/v1"
    assert str(data["source"]) == "gui-spectrogram"
    assert np.allclose(data["spectrogram"], spec)
    assert data["raw_t_ms"].dtype == np.float64
    np.testing.assert_array_equal(data["raw_t_ms"], t)  # exact, not float32
    assert not np.array_equal(t.astype(np.float32).astype(np.float64), t)
    assert np.allclose(data["raw_x"], x)
    assert "time_ms" in data and "freq_khz" in data


def test_reload_without_raw_clears_stale_raw(qapp, tmp_path):
    """A load WITHOUT raw data must drop the previous shot's cached raw trace so
    it can't be embedded in the new shot's npz (cross-shot contamination)."""
    from tokeye.gui.widgets.spectrogram_view import SpectrogramView

    view = SpectrogramView(window=None)
    h, w = 48, 32

    # First load: carries a raw trace -> cached + button enabled.
    spec1 = np.random.default_rng(0).random((h, w))
    t = 4000.0 + np.arange(300, dtype=np.float64) * 0.0005
    view._fetch_id = 1
    view._on_result(1, {"spec": spec1, "meta": _META, "t": t, "x": np.sin(t)})
    assert view._raw_t is not None
    assert view._raw_btn.isEnabled()

    # Second load: NO raw data -> the stale trace must be cleared.
    spec2 = np.random.default_rng(1).random((h, w))
    view._fetch_id = 2
    view._on_result(2, {"spec": spec2, "meta": _META, "t": None, "x": None})
    assert view._raw_t is None
    assert view._raw_x is None
    assert not view._raw_btn.isEnabled()

    npz = view.export_npz(tmp_path / "second")
    data = np.load(npz, allow_pickle=False)
    assert "raw_t_ms" not in data.files
    assert "raw_x" not in data.files


def test_export_npz_without_data_raises(qapp, tmp_path):
    from tokeye.gui.widgets.spectrogram_view import SpectrogramView

    view = SpectrogramView(window=None)
    with pytest.raises(ValueError):
        view.export_npz(tmp_path / "empty")


def test_gui_view_import_is_torch_free():
    """Opening the window must not import torch (deferred to Analyze)."""
    import os
    import subprocess
    import sys

    code = (
        "import tokeye.gui.widgets.spectrogram_view, sys; "
        "assert 'torch' not in sys.modules, 'torch imported at GUI build time'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
