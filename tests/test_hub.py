from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tokeye import hub
from tokeye.hub import (
    DEFAULT_MODEL,
    DEFAULT_REPO_ID,
    MODEL_REGISTRY,
    cached_path,
    channels_for,
    download_model,
    is_cached,
    load_model,
    model_names,
    repo_for,
    require_task,
    resolve_device,
)


def test_default_model_is_registered():
    assert DEFAULT_MODEL in MODEL_REGISTRY


def test_default_model_is_first_in_registry():
    # _build_from_state_dict tries specs in insertion order; the U-Net must
    # come first so its checkpoints never construct the R-CNN builder.
    assert next(iter(MODEL_REGISTRY)) == DEFAULT_MODEL


def test_repo_for_resolves_per_model_repo():
    assert repo_for("big_tf_unet") == DEFAULT_REPO_ID
    assert repo_for("ae_tf_maskrcnn") == "nc1/ae_tf_maskrcnn"
    # Unknown names (e.g. local paths) fall back to the default repo.
    assert repo_for("/some/local/model.pt") == DEFAULT_REPO_ID


def test_download_model_uses_per_model_repo(monkeypatch):
    seen = {}

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        seen["repo_id"] = repo_id
        seen["filename"] = filename
        return "/fake/path.pt"

    monkeypatch.setattr("tokeye.hub.hf_hub_download", fake_hf_hub_download)

    download_model("ae_tf_maskrcnn")
    assert seen == {
        "repo_id": "nc1/ae_tf_maskrcnn",
        "filename": "ae_tf_maskrcnn_251223.pt",
    }

    download_model("ae_tf_maskrcnn", repo_id="someone/else")
    assert seen["repo_id"] == "someone/else"

    download_model("big_tf_unet")
    assert seen["repo_id"] == DEFAULT_REPO_ID


def test_load_model_from_registry_downloads_and_loads(tmp_path, monkeypatch):
    spec = MODEL_REGISTRY["big_tf_unet"]
    weights_path = tmp_path / spec.filename
    torch.save(spec.builder().state_dict(), weights_path)

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        assert filename == spec.filename
        return str(weights_path)

    monkeypatch.setattr("tokeye.hub.hf_hub_download", fake_hf_hub_download)

    model = load_model("big_tf_unet", device="cpu")

    assert not model.training  # eval mode
    with torch.no_grad():
        out = model(torch.randn(1, 1, 64, 64))
    assert out[0].shape == (1, 2, 64, 64)


def test_load_model_from_local_path_with_matching_state_dict(tmp_path):
    spec = MODEL_REGISTRY["big_tf_unet"]
    weights_path = tmp_path / "checkpoint.pt"
    torch.save(spec.builder().state_dict(), weights_path)

    model = load_model(weights_path, device="cpu")

    assert not model.training
    with torch.no_grad():
        out = model(torch.randn(1, 1, 64, 64))
    assert out[0].shape == (1, 2, 64, 64)


def test_load_model_legacy_pickled_module_falls_back(tmp_path, caplog):
    legacy_path = tmp_path / "legacy.pt"
    torch.save(nn.Linear(2, 2), legacy_path)

    with caplog.at_level(logging.WARNING, logger="tokeye.hub"):
        model = load_model(legacy_path, device="cpu")

    assert isinstance(model, nn.Linear)
    assert not model.training
    assert "trust" in caplog.text.lower()


def test_load_model_unknown_name_raises_value_error_listing_registry():
    with pytest.raises(ValueError, match="big_tf_unet"):
        load_model("not_a_real_model_name")


def test_load_model_nonexistent_path_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_model("/nonexistent/directory/does_not_exist.pt")


def test_missing_local_path_error_keeps_the_given_spelling():
    # Path() would normalize the "./" away.
    with pytest.raises(FileNotFoundError, match=re.escape("./nope/missing.pt")):
        load_model("./nope/missing.pt")


def test_specs_carry_task_channels_and_size():
    unet, ae = MODEL_REGISTRY["big_tf_unet"], MODEL_REGISTRY["ae_tf_maskrcnn"]
    assert (unet.task, unet.channels, unet.size_mb) == (
        "segmentation",
        ("coherent", "transient"),
        31,
    )
    assert (ae.task, ae.channels, ae.size_mb) == ("instance", (), 184)


def test_model_names_by_task():
    assert model_names() == ["big_tf_unet", "ae_tf_maskrcnn"]
    assert model_names("segmentation") == ["big_tf_unet"]
    assert model_names("instance") == ["ae_tf_maskrcnn"]


def test_require_task():
    require_task("big_tf_unet", "segmentation")
    require_task("some/local/model.pt", "segmentation")  # checked after loading
    with pytest.raises(ValueError, match="tokeye alfvenspec"):
        require_task("ae_tf_maskrcnn", "segmentation")
    with pytest.raises(ValueError, match="tokeye run"):
        require_task("big_tf_unet", "instance")


def test_channels_for():
    assert channels_for("big_tf_unet") == ("coherent", "transient")
    assert channels_for("ae_tf_maskrcnn") == ()
    assert channels_for("local.pt") == ("coherent", "transient")


@pytest.mark.parametrize("cached", [None, "/cache/big_tf_unet_251210.pt"])
def test_cached_path_and_is_cached(monkeypatch, cached):
    monkeypatch.setattr("tokeye.hub.try_to_load_from_cache", lambda r, f: cached)
    assert cached_path("big_tf_unet") == (None if cached is None else Path(cached))
    assert is_cached("big_tf_unet") is (cached is not None)


def test_cached_path_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        cached_path("nope")


def test_resolve_device_prefers_cuda_then_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hub, "_mps_available", lambda: True)
    assert resolve_device("auto") == "mps"
    monkeypatch.setattr(hub, "_mps_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
    assert resolve_device("cpu") == "cpu"


def test_hub_import_does_not_pull_in_torchvision():
    code = "import sys, tokeye.hub; assert 'torchvision' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_ae_builder_without_torchvision_hints_at_the_extra(monkeypatch):
    # None in sys.modules makes an import fail; cover already-imported
    # submodules too, or `from torchvision.models...` would still resolve.
    for name in list(sys.modules):
        if name == "torchvision" or name.startswith("torchvision."):
            monkeypatch.setitem(sys.modules, name, None)
        elif name.startswith("tokeye.models.ae_tf_maskrcnn"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "torchvision", None)

    with pytest.raises(ImportError, match=r"tokeye\[ae\]"):
        MODEL_REGISTRY["ae_tf_maskrcnn"].builder()


def test_state_dict_sniffing_treats_missing_extra_as_mismatch(tmp_path, monkeypatch):
    def no_torchvision():
        raise ImportError("ae_tf_maskrcnn needs torchvision")

    spec = MODEL_REGISTRY["ae_tf_maskrcnn"]
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "ae_tf_maskrcnn",
        hub.ModelSpec(
            spec.name, spec.filename, no_torchvision, spec.repo_id, "instance"
        ),
    )
    path = tmp_path / "other.pt"
    torch.save(nn.Linear(2, 2).state_dict(), path)

    with pytest.raises(ValueError, match="needs torchvision"):
        load_model(path, device="cpu")
