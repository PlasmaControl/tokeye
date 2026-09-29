"""Pre-1.0 contracts: what a 0.12.0 caller got, a 1.0 caller still gets.

``TokEye.spectrogram`` and ``batch.load_input`` return float64, ``predict``
on a one-channel model returns ``(H, W)``, an ``fs`` key in a pre-1.0 STFT
settings dict moves to ``fs=``, and ``0``/``1`` still work (with a warning)
for the boolean settings.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn

from tokeye import batch
from tokeye.api import TokEye
from tokeye.config import SpectrogramConfig
from tokeye.transforms import compute_stft

CFG = SpectrogramConfig(n_fft=256, hop=64)


def _conv(channels: int) -> nn.Module:
    return nn.Conv2d(1, channels, 1)


@pytest.fixture
def stub(monkeypatch):
    model = _conv(2)
    monkeypatch.setattr("tokeye.hub.load_model", lambda source, device="auto": model)
    return model


@pytest.fixture
def eye(stub):
    return TokEye(config=CFG)


def _signal(n: int = 8192) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(n)


def _must_not_load(*args, **kwargs):
    raise AssertionError("the model was loaded")


class TestFloat64Wrappers:
    def test_spectrogram_of_2d_float64_is_the_input_values_not_the_input(self, eye):
        arr = np.random.default_rng(0).random((32, 24))

        result = eye.spectrogram(arr)

        assert result.dtype == np.float64
        assert np.array_equal(result, arr)
        assert not np.shares_memory(result, arr)

    def test_spectrogram_with_log_is_float64_log1p(self, eye):
        arr = np.random.default_rng(0).random((32, 24))

        result = eye.spectrogram(arr, log=True)

        assert result.dtype == np.float64
        assert np.array_equal(result, np.log1p(arr))

    def test_spectrogram_of_1d_is_the_float64_stft(self, eye):
        x = _signal()

        result = eye.spectrogram(x)

        assert result.dtype == np.float64
        assert np.array_equal(
            result.astype(np.float32), eye.segment(x).spectrogram.values
        )
        assert np.array_equal(
            result, compute_stft(np.asarray(x, dtype=np.float64), **CFG.stft_kwargs())
        )

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_load_input_of_1d_is_the_float64_stft(self, tmp_path, dtype):
        x = _signal().astype(dtype)
        path = tmp_path / "x.npy"
        np.save(path, x)

        result = batch.load_input(path, {"n_fft": 256, "hop": 64})

        assert result.dtype == np.float64
        assert np.array_equal(
            result, compute_stft(np.asarray(x, dtype=np.float64), **CFG.stft_kwargs())
        )

    def test_load_input_of_2d_float64_is_the_file(self, tmp_path):
        path = tmp_path / "spec.npy"
        np.save(path, np.random.default_rng(1).random((32, 24)))

        result = batch.load_input(path, {})

        assert result.dtype == np.float64
        assert np.array_equal(result, np.load(path))


class TestSingleChannelPredict:
    def test_one_channel_model_gives_h_w_from_predict_and_call(self, monkeypatch):
        monkeypatch.setattr("tokeye.hub.load_model", lambda s, d="auto": _conv(1))
        eye = TokEye("custom.pt")
        spec = np.random.default_rng(0).random((32, 40))

        assert eye.predict(spec).shape == (32, 40)
        assert eye(spec).shape == (32, 40)
        assert eye.segment(spec).mask.shape == (1, 32, 40)

    def test_two_channel_model_keeps_the_channel_axis(self, eye):
        spec = np.random.default_rng(0).random((32, 40))

        assert eye.predict(spec).shape == (2, 32, 40)
        assert eye(spec).shape == (2, 32, 40)


class TestFsInSettingsDicts:
    @pytest.fixture
    def signal_npy(self, tmp_path):
        path = tmp_path / "signal.npy"
        np.save(path, _signal())
        return path

    def _params(self, out_dir: Path) -> dict:
        return json.loads((out_dir / "signal_params.json").read_text("utf-8"))

    def test_run_batch_moves_fs_out_of_stft_kwargs(self, stub, signal_npy, tmp_path):
        settings = {"n_fft": 256, "hop": 64, "fs": 1000.0}
        out_dir = tmp_path / "out"

        with pytest.warns(DeprecationWarning):
            failures = batch.run_batch(
                [str(signal_npy)], out_dir=out_dir, stft_kwargs=settings
            )

        assert failures == 0
        params = self._params(out_dir)
        assert params["fs"] == 1000.0
        assert params["config"]["n_fft"] == 256
        assert settings == {"n_fft": 256, "hop": 64, "fs": 1000.0}

    def test_process_file_moves_fs_out_of_a_dict_config(
        self, stub, signal_npy, tmp_path
    ):
        settings = {"n_fft": 256, "fs": 1000.0}

        with pytest.warns(DeprecationWarning):
            batch.process_file(signal_npy, stub, settings, tmp_path)

        params = self._params(tmp_path)
        assert params["fs"] == 1000.0
        assert params["config"]["n_fft"] == 256
        assert settings == {"n_fft": 256, "fs": 1000.0}

    def test_process_files_moves_fs_out_of_a_dict_config(
        self, stub, signal_npy, tmp_path
    ):
        with pytest.warns(DeprecationWarning):
            failures = batch.process_files(
                [signal_npy], stub, {"n_fft": 256, "fs": 1000.0}, tmp_path
            )

        assert failures == 0
        assert self._params(tmp_path)["fs"] == 1000.0

    def test_fs_given_twice_raises_before_loading(
        self, monkeypatch, signal_npy, tmp_path
    ):
        monkeypatch.setattr("tokeye.hub.load_model", _must_not_load)

        with (
            pytest.raises(ValueError, match="fs given twice"),
            pytest.warns(DeprecationWarning),
        ):
            batch.run_batch(
                [str(signal_npy)],
                out_dir=tmp_path / "out",
                stft_kwargs={"fs": 1000.0},
                fs=2000.0,
            )
        assert not (tmp_path / "out").exists()

    def test_an_fs_of_none_in_the_dict_counts_as_absent(
        self, stub, signal_npy, tmp_path
    ):
        with pytest.warns(DeprecationWarning):
            batch.process_file(signal_npy, stub, {"fs": None}, tmp_path, fs=2000.0)

        assert self._params(tmp_path)["fs"] == 2000.0

    def test_a_bad_fs_in_the_dict_fails_before_loading(
        self, monkeypatch, signal_npy, tmp_path
    ):
        monkeypatch.setattr("tokeye.hub.load_model", _must_not_load)

        with (
            pytest.raises(ValueError, match="fs must be a finite positive"),
            pytest.warns(DeprecationWarning),
        ):
            batch.run_batch(
                [str(signal_npy)], out_dir=tmp_path / "out", stft_kwargs={"fs": -1.0}
            )

    def test_load_input_accepts_fs_in_the_dict(self, signal_npy):
        result = batch.load_input(signal_npy, {"n_fft": 256, "hop": 64, "fs": 1000.0})

        assert result.dtype == np.float64
        assert np.array_equal(
            result, compute_stft(np.load(signal_npy), **CFG.stft_kwargs())
        )

    def test_from_dict_points_fs_to_its_own_argument(self):
        with pytest.raises(ValueError, match="pass fs= separately"):
            SpectrogramConfig.from_dict({"fs": 1})

    def test_from_dict_names_unknown_keys_of_mixed_types(self):
        with pytest.raises(ValueError, match="unknown SpectrogramConfig") as info:
            SpectrogramConfig.from_dict({1: 2, "bogus": 3})
        assert "['bogus', 1]" in str(info.value)


class TestBooleanSettings:
    def test_numpy_bools_are_stored_as_python_bools(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg = SpectrogramConfig(clip_dc=np.bool_(False), log=np.bool_(True))

        assert cfg.clip_dc is False
        assert cfg.log is True

    def test_integer_one_warns_and_becomes_true(self):
        with pytest.warns(DeprecationWarning, match=r"^clip_dc=1: pass True or False"):
            cfg = SpectrogramConfig(clip_dc=1)

        assert cfg.clip_dc is True

    def test_numpy_integer_zero_warns_and_becomes_false(self):
        with pytest.warns(DeprecationWarning, match=r"^log=.*: pass True or False"):
            cfg = SpectrogramConfig(log=np.int64(0))

        assert cfg.log is False

    @pytest.mark.parametrize(
        "changes", [{"clip_dc": 1.0}, {"log": np.float64(0.0)}, {"clip_dc": 2}]
    )
    def test_floats_and_other_integers_still_raise(self, changes):
        with pytest.raises(TypeError, match="must be True or False"):
            SpectrogramConfig(**changes)

    def test_tokeye_accepts_an_integer_flag_with_a_warning(self, stub):
        with pytest.warns(DeprecationWarning, match="clip_dc=1"):
            eye = TokEye(clip_dc=1)

        assert eye.config.clip_dc is True

    @pytest.mark.parametrize(
        "build",
        [
            lambda: SpectrogramConfig(clip_dc=1),
            lambda: SpectrogramConfig().replace(log=1),
            lambda: TokEye(clip_dc=1),
        ],
        ids=["constructor", "replace", "tokeye"],
    )
    def test_the_warning_names_the_callers_line(self, stub, build):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build()

        assert [w.category for w in caught] == [DeprecationWarning]
        assert Path(caught[0].filename).name == Path(__file__).name
