from __future__ import annotations

import numpy as np
import pytest
import torch.nn as nn

import tokeye
from tokeye.api import TokEye
from tokeye.config import SpectrogramConfig
from tokeye.result import Segmentation


@pytest.fixture
def eye(monkeypatch):
    monkeypatch.setattr(
        "tokeye.hub.load_model", lambda source, device="auto": nn.Conv2d(1, 2, 1)
    )
    return TokEye(n_fft=64, hop=16)


class TestLazyExport:
    def test_package_getattr_resolves_class(self):
        assert tokeye.TokEye is TokEye

    def test_lazy_segmentation_and_config(self):
        from tokeye.config import SpectrogramConfig
        from tokeye.result import Segmentation

        assert tokeye.Segmentation is Segmentation
        assert tokeye.SpectrogramConfig is SpectrogramConfig
        assert {"TokEye", "Segmentation", "SpectrogramConfig"} <= set(dir(tokeye))

    def test_version_is_a_string(self):
        assert isinstance(tokeye.__version__, str)
        assert tokeye.__version__

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError, match="no_such_thing"):
            _ = tokeye.no_such_thing


class TestPredict:
    def test_1d_signal_returns_two_channel_mask(self, eye):
        rng = np.random.default_rng(0)
        mask = eye(rng.standard_normal(2048))

        assert mask.ndim == 3
        assert mask.shape[0] == 2
        assert np.all((mask >= 0) & (mask <= 1))  # sigmoid scores

    def test_2d_spectrogram_preserves_shape(self, eye):
        mask = eye(np.random.default_rng(0).random((48, 40)))

        assert mask.shape == (2, 48, 40)

    def test_3d_input_raises(self, eye):
        with pytest.raises(ValueError, match="ndim=3"):
            eye(np.zeros((2, 8, 8)))

    def test_call_matches_predict(self, eye):
        arr = np.random.default_rng(1).random((16, 16))

        np.testing.assert_allclose(eye(arr), eye.predict(arr))


class TestLogOption:
    def test_off_by_default_2d_passthrough(self, eye):
        arr = np.random.default_rng(0).random((16, 16))

        np.testing.assert_allclose(eye.spectrogram(arr), arr)

    def test_instance_level_log_applies_log1p(self, monkeypatch):
        monkeypatch.setattr(
            "tokeye.hub.load_model", lambda source, device="auto": nn.Conv2d(1, 2, 1)
        )
        arr = np.random.default_rng(0).random((16, 16))

        eye = TokEye(log=True)

        np.testing.assert_allclose(eye.spectrogram(arr), np.log1p(arr))

    def test_per_call_override_wins(self, eye):
        arr = np.random.default_rng(0).random((16, 16))

        np.testing.assert_allclose(eye.spectrogram(arr, log=True), np.log1p(arr))
        assert eye.log is False  # instance setting untouched

    def test_negative_values_with_log_raise(self, eye):
        with pytest.raises(ValueError, match="negative"):
            eye.spectrogram(np.full((8, 8), -3.0), log=True)

    def test_log_ignored_for_1d_signal(self, eye):
        signal = np.random.default_rng(0).standard_normal(2048)

        # STFT already log-scales; a signed signal must not trip the
        # linear-scale guard.
        spec = eye.spectrogram(signal, log=True)

        assert spec.ndim == 2


class _OneChannel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv(x)


class TestConfig:
    def test_explicit_kwargs_override_the_config(self, monkeypatch):
        monkeypatch.setattr(
            "tokeye.hub.load_model", lambda s, d="auto": nn.Conv2d(1, 2, 1)
        )
        eye = TokEye(config=SpectrogramConfig(hop=64, n_fft=256), hop=32)
        assert (eye.config.n_fft, eye.config.hop) == (256, 32)
        assert TokEye(config={"hop": 64}).config.hop == 64
        assert TokEye().config == SpectrogramConfig()

    def test_positional_order_is_unchanged(self, monkeypatch):
        monkeypatch.setattr(
            "tokeye.hub.load_model", lambda s, d="auto": nn.Conv2d(1, 2, 1)
        )
        eye = TokEye("big_tf_unet", "cpu", 512, 64, False, 2.0, 98.0, True)
        assert eye.config == SpectrogramConfig(
            n_fft=512, hop=64, clip_dc=False, clip_low=2.0, clip_high=98.0, log=True
        )

    def test_log_property_round_trips_through_config(self, eye):
        eye.log = True
        assert eye.config.log is True
        assert eye.log is True

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"fs": -1.0}, ValueError),
            ({"tile": 10}, ValueError),
            ({"hop": 0}, ValueError),
        ],
    )
    def test_bad_settings_fail_before_loading(self, monkeypatch, kwargs, error):
        def boom(*args, **kwargs):
            raise AssertionError("load_model should not be called")

        monkeypatch.setattr("tokeye.hub.load_model", boom)
        with pytest.raises(error):
            TokEye(**kwargs)

    def test_instance_models_are_rejected_before_loading(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("load_model should not be called")

        monkeypatch.setattr("tokeye.hub.load_model", boom)
        with pytest.raises(ValueError, match="tokeye alfvenspec"):
            TokEye("ae_tf_maskrcnn")


class TestSegment:
    def test_segment_returns_a_segmentation_with_axes(self, eye):
        signal = np.random.default_rng(0).standard_normal(2048)
        seg = eye.segment(signal, fs=1000.0)

        assert isinstance(seg, Segmentation)
        assert seg.channels == ("coherent", "transient")
        assert seg.mask.shape == (2, *seg.spectrogram.shape)
        assert seg.fs == 1000.0
        assert seg.freqs[0] == pytest.approx(1000.0 / 64)
        np.testing.assert_array_equal(seg.mask, eye.predict(signal))

    def test_instance_fs_is_the_default(self, monkeypatch):
        monkeypatch.setattr(
            "tokeye.hub.load_model", lambda s, d="auto": nn.Conv2d(1, 2, 1)
        )
        eye = TokEye(n_fft=64, hop=16, fs=500.0)
        signal = np.random.default_rng(0).standard_normal(2048)
        assert eye.segment(signal).fs == 500.0
        assert eye.segment(signal, fs=250.0).fs == 250.0

    def test_reference_gives_cross_power(self, eye):
        rng = np.random.default_rng(0)
        seg = eye.segment(
            rng.standard_normal(2048), reference=rng.standard_normal(2048)
        )
        assert seg.spectrogram.kind == "cross"

    def test_spectrogram_is_float32(self, eye):
        assert (
            eye.spectrogram(np.random.default_rng(0).random((16, 16))).dtype
            == np.float32
        )

    def test_nan_input_is_rejected(self, eye):
        with pytest.raises(ValueError, match="non-finite"):
            eye(np.full((16, 16), np.nan))

    def test_single_channel_custom_model(self, monkeypatch):
        monkeypatch.setattr("tokeye.hub.load_model", lambda s, d="auto": _OneChannel())
        eye = TokEye("custom.pt")
        spec = np.random.default_rng(0).random((32, 32))
        assert eye.predict(spec).shape == (1, 32, 32)
        assert eye.segment(spec).channels == ("channel_0",)
