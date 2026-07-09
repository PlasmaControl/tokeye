"""ModelService: lazy caching + serialized inference (no real weights)."""

from __future__ import annotations

import numpy as np


def test_model_service_caches_load_and_delegates_infer(monkeypatch):
    import tokeye.app.analyze.load as loadmod
    import tokeye.inference as infmod
    from tokeye.gui.model_service import ModelService

    calls = {"load": 0}
    sentinel_model = object()

    def fake_model_load(name, device="auto"):
        calls["load"] += 1
        assert name == "big_tf_unet"
        return sentinel_model

    def fake_model_infer(spec, model):
        assert model is sentinel_model
        return np.zeros((2, *spec.shape), dtype=np.float32)

    monkeypatch.setattr(loadmod, "model_load", fake_model_load)
    monkeypatch.setattr(infmod, "model_infer", fake_model_infer)

    svc = ModelService()
    out = svc.infer("big_tf_unet", np.zeros((8, 6)))
    assert out.shape == (2, 8, 6)

    svc.infer("big_tf_unet", np.zeros((8, 6)))
    assert calls["load"] == 1  # model cached across calls
