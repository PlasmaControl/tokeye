"""Checkpoint loading: TorchScript archives, the torchvision hint, repo_for."""

from __future__ import annotations

import logging
import re
import struct
import sys
import types
import warnings
import zipfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from tokeye import hub
from tokeye.api import TokEye
from tokeye.cli import main


class _TwoChannel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture
def torchscript_pt(tmp_path):
    traced = torch.jit.trace(_TwoChannel().eval(), torch.zeros(1, 1, 16, 16))
    path = tmp_path / "ts.pt"
    torch.jit.save(traced, str(path))
    return path


def _tokeye_warnings(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name.startswith("tokeye") and r.levelno == logging.WARNING
    ]


def _zip(path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def _block_torchvision(monkeypatch) -> None:
    """Make every torchvision import fail, as when it is not installed."""
    monkeypatch.setitem(sys.modules, "torchvision", None)
    for key in list(sys.modules):
        if key.startswith("torchvision."):
            monkeypatch.setitem(sys.modules, key, None)
    _forget_ae_modules(monkeypatch)


def _forget_ae_modules(monkeypatch) -> None:
    for key in list(sys.modules):
        if key.startswith("tokeye.models.ae_tf_maskrcnn"):
            monkeypatch.delitem(sys.modules, key)


def _is_torchvision(name: str | None) -> bool:
    return name is not None and (
        name == "torchvision" or name.startswith("torchvision.")
    )


class TestTorchScript:
    def test_loads_in_eval_mode_with_one_warning(self, torchscript_pt, caplog):
        with (
            caplog.at_level(logging.WARNING, logger="tokeye"),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            model = hub.load_model(str(torchscript_pt), "cpu")

        assert isinstance(model, nn.Module)
        assert not model.training
        records = _tokeye_warnings(caplog)
        assert len(records) == 1
        assert "TorchScript" in records[0].getMessage()
        assert records[0].getMessage().startswith(f"{torchscript_pt}: ")
        assert not [w for w in caught if issubclass(w.category, UserWarning)]

    def test_tokeye_predicts_with_it(self, torchscript_pt):
        eye = TokEye(str(torchscript_pt), device="cpu")

        mask = eye.predict(np.random.default_rng(0).random((32, 32)))

        assert mask.shape == (2, 32, 32)

    def test_tokeye_run_uses_it(self, torchscript_pt, tmp_path):
        x = tmp_path / "x.npy"
        np.save(x, np.random.default_rng(0).random((64, 64)).astype(np.float32))
        out = tmp_path / "out"

        argv = ["run", str(x), "--model", str(torchscript_pt), "--device", "cpu"]
        exit_code = main([*argv, "--output-dir", str(out)])

        assert exit_code == 0
        assert np.load(out / "x_mask.npy").shape == (2, 64, 64)

    def test_torch_save_files_are_not_torchscript(self, tmp_path):
        state_dict = tmp_path / "sd.pt"
        torch.save(nn.Conv2d(1, 2, 3).state_dict(), state_dict)
        module = tmp_path / "module.pt"
        torch.save(nn.Conv2d(1, 2, 3), module)

        assert hub._is_torchscript(state_dict) is False
        assert hub._is_torchscript(module) is False

    def test_a_saved_archive_is_torchscript(self, torchscript_pt):
        assert hub._is_torchscript(torchscript_pt) is True

    @pytest.mark.parametrize(
        ("member", "expected"),
        [
            (".data/ts_code/0/constants.pkl", False),
            ("constants.pkl", False),
            ("final.torchscript/constants.pkl", True),
            ("anything/constants.pkl", True),
        ],
    )
    def test_constants_pkl_must_sit_one_level_down(self, tmp_path, member, expected):
        path = tmp_path / "z.pt"
        _zip(path, {member: b"x", "other/data.pkl": b"y"})

        assert hub._is_torchscript(path) is expected

    def test_a_corrupt_central_directory_is_not_torchscript(self, tmp_path):
        path = tmp_path / "corrupt.pt"
        _zip(path, {"archive/constants.pkl": b"x"})
        data = bytearray(path.read_bytes())
        eocd = data.rindex(b"PK\x05\x06")
        (cd_offset,) = struct.unpack_from("<I", data, eocd + 16)
        data[cd_offset : cd_offset + 4] = b"XXXX"
        path.write_bytes(bytes(data))
        assert zipfile.is_zipfile(path)
        with pytest.raises(zipfile.BadZipFile):
            zipfile.ZipFile(path)

        assert hub._is_torchscript(path) is False

    def test_an_unreadable_archive_is_one_value_error(self, tmp_path, monkeypatch):
        path = tmp_path / "junk.pt"
        _zip(path, {"archive/constants.pkl": b"junk", "archive/data.pkl": b"junk"})
        real_jit_load = torch.jit.load
        calls = []

        def spy(*args, **kwargs):
            calls.append(args)
            return real_jit_load(*args, **kwargs)

        monkeypatch.setattr(torch.jit, "load", spy)

        with pytest.raises(ValueError, match="not a readable checkpoint"):
            hub.load_model(str(path), "cpu")
        assert len(calls) == 1


class TestTorchvisionHint:
    def test_missing_torchvision_gets_the_hint(self, monkeypatch):
        _block_torchvision(monkeypatch)

        with pytest.raises(ImportError, match="needs torchvision") as info:
            hub.MODEL_REGISTRY["ae_tf_maskrcnn"].builder()

        assert "tokeye[ae]" in str(info.value)
        cause = info.value.__cause__
        assert isinstance(cause, ModuleNotFoundError)
        assert _is_torchvision(cause.name)

    def test_a_broken_import_inside_tokeye_propagates(self, monkeypatch):
        _forget_ae_modules(monkeypatch)
        config = types.ModuleType("tokeye.models.ae_tf_maskrcnn.config_ae_tf_maskrcnn")
        config.AETFMaskConfig = object
        model = types.ModuleType("tokeye.models.ae_tf_maskrcnn.model_ae_tf_maskrcnn")
        monkeypatch.setitem(sys.modules, config.__name__, config)
        monkeypatch.setitem(sys.modules, model.__name__, model)

        with pytest.raises(
            ImportError, match="cannot import name 'AETFMaskModel'"
        ) as info:
            hub.MODEL_REGISTRY["ae_tf_maskrcnn"].builder()

        assert info.value.__cause__ is None
        assert "tokeye[ae]" not in str(info.value)

    def test_another_missing_module_propagates(self, monkeypatch):
        _forget_ae_modules(monkeypatch)
        name = "tokeye.models.ae_tf_maskrcnn.config_ae_tf_maskrcnn"
        monkeypatch.setitem(sys.modules, name, None)

        with pytest.raises(ModuleNotFoundError) as info:
            hub.MODEL_REGISTRY["ae_tf_maskrcnn"].builder()

        assert info.value.name == name
        assert "tokeye[ae]" not in str(info.value)

    def test_state_dict_sniffing_keeps_the_whole_hint(self, tmp_path, monkeypatch):
        path = tmp_path / "conv.pt"
        torch.save(nn.Conv2d(1, 2, 3).state_dict(), path)
        _block_torchvision(monkeypatch)

        with pytest.raises(ValueError) as info:
            hub.load_model(str(path), "cpu")

        message = str(info.value)
        assert re.search(r"needs torchvision.*\(.*torchvision.*\)", message), message
        assert "tokeye[ae]" in message
        assert len(message) < 500

    def test_any_builder_failure_is_reported_whole(self, tmp_path, monkeypatch):
        def broken():
            raise RuntimeError("operator torchvision::nms does not exist")

        spec = hub.MODEL_REGISTRY["ae_tf_maskrcnn"]
        monkeypatch.setitem(
            hub.MODEL_REGISTRY,
            "ae_tf_maskrcnn",
            hub.ModelSpec(spec.name, spec.filename, broken, spec.repo_id, "instance"),
        )
        path = tmp_path / "conv.pt"
        torch.save(nn.Conv2d(1, 2, 3).state_dict(), path)

        with pytest.raises(ValueError) as info:
            hub.load_model(str(path), "cpu")

        message = str(info.value)
        assert "ae_tf_maskrcnn: operator torchvision::nms does not exist" in message
        assert "big_tf_unet: " in message


def test_download_model_asks_repo_for(monkeypatch):
    seen = []
    monkeypatch.setattr("tokeye.hub.repo_for", lambda name: "sentinel/repo")
    monkeypatch.setattr(
        "tokeye.hub.hf_hub_download",
        lambda repo_id, filename, **kwargs: seen.append(repo_id) or "/fake/w.pt",
    )

    hub.download_model("big_tf_unet")
    hub.download_model("big_tf_unet", repo_id="x/y")

    assert seen == ["sentinel/repo", "x/y"]
