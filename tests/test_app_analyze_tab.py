"""Tests for the Analyze tab (src/tokeye/app/analyze/analyze.py).

Covers three defects fixed together, plus a new feature:
1. The model dropdown being silently ignored once a model was loaded once
   (``ensure_model`` used to short-circuit on ``model is not None`` alone).
2. Transform settings only taking effect via a separate "Apply Transform
   Settings" button (now folded inline into every Load).
3. No "Analyze" progress feedback / a redundant separate "Visualize" button.
4. No way to export the loaded spectrogram + inferred mask as a ``.npz``.

These run offline (no browser, no Gradio server, no torch/model download) by
calling the module-level handlers directly.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch.nn as nn

from tokeye.app.analyze import analyze

# ============================================================================
# ensure_model
# ============================================================================


def test_ensure_model_skips_load_when_no_signal(monkeypatch):
    """No signal loaded -> warn and return state unchanged, no load.

    gr.Warning does not halt a gradio .then() chain, so ensure_model itself
    must gate the expensive download/warmup on signal presence.
    """

    def fail_load(model_file):
        raise AssertionError("model load must not run without a signal")

    monkeypatch.setattr(analyze, "wrapper_model_load", fail_load)

    with pytest.warns(UserWarning, match="Load a signal first"):
        model, loaded_name = analyze.ensure_model(None, None, "big_tf_unet", None)

    assert model is None
    assert loaded_name is None


def test_ensure_model_passes_loaded_model_through_when_name_matches(monkeypatch):
    """A model already loaded under the current dropdown name -> no reload."""

    def fail_load(model_file):
        raise AssertionError("model load must not run when already loaded")

    monkeypatch.setattr(analyze, "wrapper_model_load", fail_load)
    loaded = nn.Conv2d(1, 2, 1)
    spectrogram = np.zeros((4, 4))

    model, loaded_name = analyze.ensure_model(
        loaded, "big_tf_unet", "big_tf_unet", spectrogram
    )

    assert model is loaded
    assert loaded_name == "big_tf_unet"


def test_ensure_model_loads_when_signal_present_and_model_cold(monkeypatch):
    stub_model = nn.Conv2d(1, 2, 1)
    monkeypatch.setattr(analyze, "wrapper_model_load", lambda model_file: stub_model)
    spectrogram = np.zeros((4, 4))

    model, loaded_name = analyze.ensure_model(None, None, "big_tf_unet", spectrogram)

    assert model is stub_model
    assert loaded_name == "big_tf_unet"


def test_ensure_model_reloads_when_dropdown_name_changes(monkeypatch):
    """The staleness bug: switching the dropdown must reload on next Analyze,
    even though a model is already loaded under the old name."""
    calls = []

    def fake_load(model_file):
        calls.append(model_file)
        return f"model:{model_file}"

    monkeypatch.setattr(analyze, "wrapper_model_load", fake_load)
    spectrogram = np.zeros((4, 4))

    model, loaded_name = analyze.ensure_model(
        "model:big_tf_unet", "big_tf_unet", "other_model", spectrogram
    )

    assert calls == ["other_model"]
    assert model == "model:other_model"
    assert loaded_name == "other_model"


def test_ensure_model_reloads_after_a_previous_load_failure(monkeypatch):
    """model is None (a previous load failed) -> retry even if loaded_name
    already equals model_file (set by the failed attempt)."""
    calls = []

    def fake_load(model_file):
        calls.append(model_file)
        return "recovered-model"

    monkeypatch.setattr(analyze, "wrapper_model_load", fake_load)
    spectrogram = np.zeros((4, 4))

    model, loaded_name = analyze.ensure_model(
        None, "big_tf_unet", "big_tf_unet", spectrogram
    )

    assert calls == ["big_tf_unet"]
    assert model == "recovered-model"
    assert loaded_name == "big_tf_unet"


def test_ensure_model_progress_kwarg_is_keyword_only_with_default():
    """Direct/unit calls (as above) never pass progress - it must default."""
    with pytest.warns(UserWarning):
        analyze.ensure_model(None, None, "big_tf_unet", None)  # no TypeError


# ============================================================================
# wrapper_model_load_pair (feeds load_model_btn's [model, model_name] outputs)
# ============================================================================


def test_wrapper_model_load_pair_success_tags_name(monkeypatch):
    stub_model = nn.Conv2d(1, 2, 1)
    monkeypatch.setattr(analyze, "wrapper_model_load", lambda model_file: stub_model)

    model, loaded_name = analyze.wrapper_model_load_pair("big_tf_unet")

    assert model is stub_model
    assert loaded_name == "big_tf_unet"


def test_wrapper_model_load_pair_failure_clears_name(monkeypatch):
    monkeypatch.setattr(analyze, "wrapper_model_load", lambda model_file: None)

    model, loaded_name = analyze.wrapper_model_load_pair("bad_model")

    assert model is None
    assert loaded_name is None


# ============================================================================
# export_analysis
# ============================================================================


def test_export_analysis_round_trips_to_npz():
    rng = np.random.default_rng(0)
    spectrogram = rng.random((6, 8)).astype(np.float32)
    mask = rng.random((2, 6, 8)).astype(np.float32)

    path = analyze.export_analysis(
        spectrogram,
        mask,
        "big_tf_unet",
        1024,
        128,
        True,
        1.0,
        99.0,
        0.5,
        "Enhanced",
    )

    loaded = np.load(path, allow_pickle=False)
    assert str(loaded["schema"]) == "tokeye-analysis/v1"
    assert str(loaded["source"]) == "analyze"

    params = json.loads(str(loaded["params_json"]))
    assert params == {
        "model": "big_tf_unet",
        "n_fft": 1024,
        "hop": 128,
        "clip_dc": True,
        "clip_low": 1.0,
        "clip_high": 99.0,
        "threshold": 0.5,
        "view_mode": "Enhanced",
    }

    assert np.allclose(loaded["spectrogram"], spectrogram)
    assert np.allclose(loaded["mask"], mask)


def test_export_analysis_without_mask_omits_mask_key():
    spectrogram = np.zeros((4, 4), dtype=np.float32)

    path = analyze.export_analysis(
        spectrogram, None, "big_tf_unet", 1024, 128, True, 1.0, 99.0, 0.5, "Original"
    )

    loaded = np.load(path, allow_pickle=False)
    assert "mask" not in loaded.files


def test_export_analysis_no_signal_warns_and_returns_update():
    import gradio as gr

    with pytest.warns(UserWarning, match="Load a signal first"):
        result = analyze.export_analysis(
            None, None, "big_tf_unet", 1024, 128, True, 1.0, 99.0, 0.5, "Enhanced"
        )

    assert result == gr.update()


# ============================================================================
# App wiring smoke test
# ============================================================================


def _collect_labels(block) -> list[str]:
    """Recursively collect every component 'label' plus gr.Button text from a
    gr.Blocks render tree, so tests can assert on what's actually wired up
    without needing a browser."""
    labels = []
    label = getattr(block, "label", None)
    if label:
        labels.append(label)
    value = getattr(block, "value", None)
    if block.__class__.__name__ == "Button" and isinstance(value, str):
        labels.append(value)
    for child in getattr(block, "children", []) or []:
        labels.extend(_collect_labels(child))
    return labels


def test_create_app_constructs_without_error():
    """Blocks construction should be pure UI wiring: no model load, no network."""
    from tokeye.app.__main__ import create_app

    app = create_app()

    assert app is not None


def test_analyze_tab_wiring_labels():
    """Stale-flow controls are gone; the new one-click flow and export are
    present."""
    from tokeye.app.__main__ import create_app

    app = create_app()
    labels = _collect_labels(app)

    assert "Apply Transform Settings" not in labels
    assert "Visualize" not in labels
    assert "Analyze" in labels
    assert "Load Model" in labels
    assert "Save results (.npz)" in labels
