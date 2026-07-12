"""Control-rail widget behaviour (offscreen Qt)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def test_model_selector_default_and_choices(qapp):
    from tokeye.gui.widgets.controls import DEFAULT_MODEL, ModelSelector

    sel = ModelSelector()
    assert sel.name() == DEFAULT_MODEL == "big_tf_unet"
    items = [sel._combo.itemText(i) for i in range(sel._combo.count())]
    assert "big_tf_unet" in items
    assert sel._combo.isEditable()  # custom names / local paths allowed


def test_diagnostic_probe_repopulates_on_change(qapp):
    # Source-agnostic: the probe follows the factory's active presets, so this
    # holds on every site branch regardless of the default source kind.
    from tokeye.gui.widgets.controls import DiagnosticProbe
    from tokeye.sources.factory import active_diagnostics, default_diag_key

    diags = active_diagnostics()
    default_key = default_diag_key()
    dp = DiagnosticProbe()
    assert dp.diag_key() == default_key
    assert dp.pointname() == diags[default_key].default

    seen = []
    dp.probeChanged.connect(lambda k, p: seen.append((k, p)))
    other = next(k for k in diags if k != default_key)
    idx = dp._diag.findData(other)
    dp._diag.setCurrentIndex(idx)
    assert dp.diag_key() == other
    assert dp.pointname() == diags[other].default
    assert seen and seen[-1][0] == other


def test_shot_field_shot_and_bounds(qapp):
    from tokeye.gui.widgets.controls import ShotField

    sf = ShotField()
    assert sf.shot() is None  # 0 -> None (nothing entered)
    sf.set_shot(190904)
    assert sf.shot() == 190904
    assert sf.tlim() is None  # min==max==0
    sf.set_time_bounds(1000.0, 3000.0)
    assert sf.tlim() == (1000.0, 3000.0)


def test_stft_controls_params_and_band(qapp):
    from tokeye.gui.widgets.controls import StftControls

    s = StftControls()
    p = s.params()
    assert p["n_fft"] == 1024
    assert p["hop"] == 128  # training-recipe default
    assert p["clip_dc"] is True
    assert p["clip_low"] == 1.0 and p["clip_high"] == 99.0
    assert s.decimation() == 1
    assert s.band() == (5.0, 250.0)


def test_view_controls_state(qapp):
    from tokeye.gui.widgets.controls import ViewControls

    v = ViewControls()
    assert v.mode() == "Enhanced"
    assert v.ch0() and v.ch1()
    assert v.vmin() == 0.0 and v.vmax() == 100.0
    assert v.threshold() == pytest.approx(0.5)

    fired = []
    v.changed.connect(lambda: fired.append(1))
    v._mode.setCurrentText("Mask")
    assert v.mode() == "Mask"
    assert fired  # switching mode emits `changed`
