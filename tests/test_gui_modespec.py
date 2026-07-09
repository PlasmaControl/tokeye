"""Modespec view: discrete-n RGBA map, integer colour bar, coherence gating."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from tokeye.gui.render import axis_rect, mode_ticks


def _result():
    rng = np.random.default_rng(0)
    return {
        "t_win_ms": np.linspace(1000.0, 1020.0, 30),
        "freq_khz": np.linspace(5.0, 150.0, 25),
        "n_dominant": rng.integers(-3, 4, size=(30, 25)),
        "coherence": rng.random((30, 25)),
        "n_range": (-3, 3),
        "c95": 0.3,
    }


def test_modespec_view_renders_rgba_with_real_axes(qapp):
    from tokeye.gui.widgets.modespec_view import ModespecView

    view = ModespecView(window=None)
    view._result = _result()
    view.gate_controls._gate.setChecked(False)  # ungated (no worker needed)
    view._render_modes(reset_view=True)

    img = view.canvas.image_item()
    assert img.image.ndim == 3 and img.image.shape[2] == 4  # RGBA

    expected = axis_rect(view._result["t_win_ms"], view._result["freq_khz"])
    vr = img.mapRectToView(img.boundingRect())
    assert (vr.x(), vr.y(), vr.width(), vr.height()) == pytest.approx(expected, rel=1e-6)


def test_modespec_colorbar_has_integer_ticks(qapp):
    from tokeye.gui.widgets.modespec_view import ModespecView

    view = ModespecView(window=None)
    view._result = _result()
    view.gate_controls._gate.setChecked(False)
    view._render_modes(reset_view=True)

    cbar = view.canvas.colorbar()
    assert cbar is not None
    assert cbar.axis._tickLevels == [mode_ticks(-3, 3)]


def test_coherence_threshold_suppresses_more(qapp):
    from tokeye.gui.widgets.modespec_view import ModespecView

    view = ModespecView(window=None)
    view._result = _result()
    view.gate_controls._gate.setChecked(False)

    view.gate_controls._coh._slider.setValue(10)  # coh_thresh 0.10
    view._render_modes()
    opaque_low = int((view.canvas.image_item().image[..., 3] > 0).sum())

    view.gate_controls._coh._slider.setValue(90)  # coh_thresh 0.90
    view._render_modes()
    opaque_high = int((view.canvas.image_item().image[..., 3] > 0).sum())

    assert opaque_high < opaque_low  # a higher coherence bar keeps fewer modes


def test_modespec_stale_result_dropped(qapp):
    from tokeye.gui.widgets.modespec_view import ModespecView

    view = ModespecView(window=None)
    view._analyze_id = 9
    view._on_result(4, {"result": _result(), "tok_mask": None, "gate_meta": None,
                        "nd": None})
    assert view._result is None  # stale id ignored


def test_modespec_view_import_is_torch_free():
    import os
    import subprocess
    import sys

    code = (
        "import tokeye.gui.widgets.modespec_view, sys; "
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
