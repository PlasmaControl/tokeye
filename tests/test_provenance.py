"""Weights provenance: model labels and weights records, without local paths."""

from __future__ import annotations

import hashlib
import json
import os
import types

import numpy as np
import pytest
import torch.nn as nn

from tokeye import batch, export, hub
from tokeye.api import TokEye
from tokeye.cli import main
from tokeye.preprocess import prepare
from tokeye.result import Segmentation

_REVISION = "a" * 40
_FILENAME = hub.MODEL_REGISTRY["big_tf_unet"].filename


def strings(obj):
    """Every string in parsed JSON: keys and values, recursing into containers."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from strings(key)
            yield from strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from strings(value)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def x_npy(tmp_path):
    path = tmp_path / "x.npy"
    np.save(path, np.random.default_rng(0).random((64, 64)).astype(np.float32))
    return path


@pytest.fixture
def local_checkpoint(tmp_path):
    path = tmp_path / "deep" / "m.pt"
    path.parent.mkdir()
    path.write_bytes(b"local checkpoint bytes")
    return path


@pytest.fixture
def stub_model(monkeypatch):
    model = nn.Conv2d(1, 2, kernel_size=1)
    monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: model)
    return model


@pytest.fixture
def cached_snapshot(tmp_path, monkeypatch):
    """A fake HF cache entry: snapshots/<rev>/<file> -> blobs/<name>."""
    repo = tmp_path / "hub" / "models--x--y"
    blob = repo / "blobs" / "0123abcd"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"registry weights bytes")
    snapshot = repo / "snapshots" / _REVISION / _FILENAME
    snapshot.parent.mkdir(parents=True)
    try:
        snapshot.symlink_to(blob)
    except OSError:  # Windows without developer mode
        snapshot.write_bytes(blob.read_bytes())
    asked = []

    def fake_lookup(repo_id, filename, *args, **kwargs):
        asked.append((repo_id, filename))
        return str(snapshot)

    monkeypatch.setattr("tokeye.hub.try_to_load_from_cache", fake_lookup)
    return types.SimpleNamespace(
        path=snapshot, digest=_digest(blob.read_bytes()), asked=asked
    )


def _registry_record(digest: str | None, revision: str | None = _REVISION) -> dict:
    return {
        "repo": hub.repo_for("big_tf_unet"),
        "filename": _FILENAME,
        "revision": revision,
        "sha256": digest,
    }


def _params(out_dir, stem="x") -> tuple[dict, str]:
    text = (out_dir / f"{stem}_params.json").read_text(encoding="utf-8")
    return json.loads(text), text


class TestModelLabel:
    def test_registry_names_are_kept(self):
        assert hub.model_label("big_tf_unet") == "big_tf_unet"

    def test_paths_become_file_names(self, tmp_path):
        assert hub.model_label(tmp_path / "sub" / "m.pt") == "m.pt"
        assert hub.model_label(str(tmp_path / "sub" / "m.pt")) == "m.pt"


class TestWeightsInfo:
    def test_a_cached_registry_model(self, cached_snapshot):
        info = hub.weights_info("big_tf_unet")

        assert info == _registry_record(cached_snapshot.digest)
        assert cached_snapshot.asked == [(hub.repo_for("big_tf_unet"), _FILENAME)]

    def test_an_uncached_registry_model(self, monkeypatch):
        monkeypatch.setattr(
            "tokeye.hub.try_to_load_from_cache", lambda *args, **kwargs: None
        )

        assert hub.weights_info("big_tf_unet") == _registry_record(None, None)

    def test_a_local_file_is_hashed_once(self, tmp_path):
        path = tmp_path / "m.pt"
        path.write_bytes(b"first")
        hub._sha256_file.cache_clear()

        first = hub.weights_info(path)
        second = hub.weights_info(str(path))

        assert first == second == {"name": "m.pt", "sha256": _digest(b"first")}
        info = hub._sha256_file.cache_info()
        assert (info.misses, info.hits) == (1, 1)

        path.write_bytes(b"a different, longer content")
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))

        third = hub.weights_info(path)

        assert third == {
            "name": "m.pt",
            "sha256": _digest(b"a different, longer content"),
        }
        assert hub._sha256_file.cache_info().misses == 2

    def test_anything_else_has_no_record(self, tmp_path):
        assert hub.weights_info("not a model and not a file") is None
        assert hub.weights_info(tmp_path) is None
        assert hub.weights_info(tmp_path / "missing.pt") is None

    def test_a_failed_hash_records_no_digest(self, tmp_path, monkeypatch):
        path = tmp_path / "m.pt"
        path.write_bytes(b"x")

        def unreadable(*args):
            raise PermissionError("no read permission")

        monkeypatch.setattr(hub, "_sha256_file", unreadable)

        assert hub.weights_info(path) == {"name": "m.pt", "sha256": None}


class TestSegmentationWeights:
    def _spectrogram(self):
        return prepare(np.random.default_rng(0).random((32, 32)))

    def test_not_a_mapping_is_rejected(self):
        spec = self._spectrogram()

        with pytest.raises(
            TypeError, match="weights must be a mapping or None, got list"
        ):
            Segmentation(np.zeros((2, 32, 32)), spec, weights=["x"])

    def test_a_mapping_is_stored_as_a_dict(self):
        spec = self._spectrogram()
        weights = types.MappingProxyType({"name": "m.pt", "sha256": None})

        seg = Segmentation(np.zeros((2, 32, 32)), spec, weights=weights)

        assert type(seg.weights) is dict
        assert seg.weights == {"name": "m.pt", "sha256": None}
        assert Segmentation(np.zeros((2, 32, 32)), spec).weights is None


class TestWhereTheyLand:
    def test_tokeye_records_the_file_name(self, tmp_path, local_checkpoint, stub_model):
        eye = TokEye(str(local_checkpoint), device="cpu")

        assert eye.model_name == "m.pt"
        assert eye.weights == {
            "name": "m.pt",
            "sha256": _digest(local_checkpoint.read_bytes()),
        }

        x = np.random.default_rng(0).random((64, 64)).astype(np.float32)
        saved = eye.segment(x).save(tmp_path / "seg.npz")

        with np.load(saved, allow_pickle=False) as data:
            params = json.loads(str(data["params_json"]))
        assert params["weights"] == eye.weights
        assert not any(str(tmp_path) in s for s in strings(params))
        loaded = Segmentation.load(saved)
        assert loaded.model == "m.pt"
        assert loaded.weights == eye.weights

    def test_run_with_a_local_checkpoint(
        self, tmp_path, x_npy, local_checkpoint, stub_model
    ):
        out = tmp_path / "out"

        argv = ["run", str(x_npy), "--model", str(local_checkpoint), "--no-png"]
        assert main([*argv, "--output-dir", str(out)]) == 0

        params, text = _params(out)
        assert params["model"] == "m.pt"
        assert params["weights"] == {
            "name": "m.pt",
            "sha256": _digest(local_checkpoint.read_bytes()),
        }
        assert params["input"] == str(x_npy)
        assert not any(str(tmp_path / "deep") in s for s in strings(json.loads(text)))

    def test_run_with_the_default_model(
        self, tmp_path, x_npy, cached_snapshot, stub_model
    ):
        out = tmp_path / "out"

        assert main(["run", str(x_npy), "--no-png", "--output-dir", str(out)]) == 0

        params, _ = _params(out)
        assert params["model"] == "big_tf_unet"
        assert params["weights"] == _registry_record(cached_snapshot.digest)

    def test_process_file_without_model_name(self, tmp_path, x_npy):
        out = tmp_path / "out"
        out.mkdir()

        batch.process_file(x_npy, nn.Conv2d(1, 2, 1), None, out, save_png=False)

        params, text = _params(out)
        assert params["model"] == "unknown"
        assert params["weights"] is None
        assert '"weights": null' in text

    def test_npz_bundle_carries_the_same_record(
        self, tmp_path, x_npy, local_checkpoint, stub_model
    ):
        out = tmp_path / "out"

        argv = ["run", str(x_npy), "--model", str(local_checkpoint), "--no-png"]
        assert main([*argv, "--format", "npz", "--output-dir", str(out)]) == 0

        params, _ = _params(out)
        seg = Segmentation.load(out / "x_tokeye.npz")
        assert seg.model == params["model"] == "m.pt"
        assert seg.weights == params["weights"]
        assert seg.weights["name"] == "m.pt"
        with np.load(out / "x_tokeye.npz", allow_pickle=False) as data:
            bundle_params = json.loads(str(data["params_json"]))
        assert not any(str(tmp_path) in s for s in strings(bundle_params))

    @pytest.mark.parametrize("weights", [["x"], {"name": 3}, "m.pt"])
    def test_invalid_recorded_weights_warn_once(self, tmp_path, weights):
        spectrogram = np.zeros((32, 32), dtype=np.float32)
        bundle = export.analysis_bundle(
            spectrogram=spectrogram,
            mask=np.zeros((2, 32, 32), dtype=np.float32),
            params={"model": "m.pt", "weights": weights},
        )
        path = export.save_npz(tmp_path / "b.npz", bundle)

        with pytest.warns(UserWarning) as record:
            seg = Segmentation.load(path)

        assert len(record) == 1
        assert "weights" in str(record[0].message)
        assert seg.weights is None

    @pytest.mark.parametrize("params", [{"model": "m.pt"}, {"weights": None}])
    def test_absent_weights_load_silently(self, tmp_path, params, recwarn):
        bundle = export.analysis_bundle(
            spectrogram=np.zeros((32, 32), dtype=np.float32),
            mask=np.zeros((2, 32, 32), dtype=np.float32),
            params=params,
        )
        path = export.save_npz(tmp_path / "b.npz", bundle)

        seg = Segmentation.load(path)

        assert seg.weights is None
        assert not recwarn.list
