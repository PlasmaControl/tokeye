from __future__ import annotations

import numpy as np
import pytest

from tokeye.config import SpectrogramConfig
from tokeye.preprocess import Spectrogram, prepare
from tokeye.transforms import compute_stft

CFG = SpectrogramConfig(n_fft=256, hop=64)
FS = 10_000.0


@pytest.fixture
def signal() -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(4096) / FS
    return np.sin(2 * np.pi * 1000 * t) + 0.1 * rng.standard_normal(t.size)


class TestSignalInput:
    def test_1d_is_the_stft_as_float32(self, signal):
        spec = prepare(signal, CFG)

        assert isinstance(spec, Spectrogram)
        assert spec.kind == "stft"
        assert spec.values.dtype == np.float32
        assert spec.values.flags.c_contiguous
        expected = compute_stft(signal, **CFG.stft_kwargs()).astype(np.float32)
        np.testing.assert_array_equal(spec.values, expected)
        assert spec.shape == expected.shape

    def test_axes_in_hz_and_seconds_when_fs_known(self, signal):
        spec = prepare(signal, CFG, fs=FS)

        assert spec.fs == FS
        assert spec.freqs.shape == (spec.shape[0],)
        assert spec.times.shape == (spec.shape[1],)
        # clip_dc drops row 0, so the first row is one bin above DC
        assert spec.freqs[0] == pytest.approx(FS / 256)
        # Centred frames: 1 + N // hop columns, column j at j * hop / fs.
        assert spec.shape[1] == 1 + 4096 // 64
        np.testing.assert_array_equal(spec.times, np.arange(spec.shape[1]) * 64 / FS)
        assert spec.times[0] == 0.0

    def test_axes_are_indices_without_fs(self, signal):
        spec = prepare(signal, CFG)

        assert spec.fs is None
        assert spec.freqs[0] == 1.0  # bin 1 (DC dropped)
        np.testing.assert_array_equal(spec.times, np.arange(spec.shape[1]))

    def test_keep_dc_axis_starts_at_zero(self, signal):
        spec = prepare(signal, CFG.replace(clip_dc=False), fs=FS)
        assert spec.freqs[0] == 0.0

    def test_config_may_be_a_mapping(self, signal):
        assert (
            prepare(signal, {"n_fft": 256, "hop": 64}).shape
            == prepare(signal, CFG).shape
        )


class TestCrossPower:
    def test_reference_gives_the_cross_spectrum(self, signal):
        other = np.roll(signal, 3)
        spec = prepare(signal, CFG, reference=other)

        assert spec.kind == "cross"
        expected = compute_stft(np.stack([signal, other]), **CFG.stft_kwargs())
        np.testing.assert_array_equal(spec.values, expected.astype(np.float32))

    def test_reference_needs_same_length(self, signal):
        with pytest.raises(ValueError, match="same length"):
            prepare(signal, CFG, reference=signal[:-1])

    def test_reference_needs_1d(self, signal):
        with pytest.raises(ValueError, match="two 1D signals"):
            prepare(np.zeros((8, 8)), CFG, reference=np.zeros(64))


class TestSpectrogramInput:
    def test_2d_passthrough(self):
        arr = np.random.default_rng(1).random((48, 40))
        spec = prepare(arr, CFG)

        assert spec.kind == "spectrogram"
        np.testing.assert_allclose(spec.values, arr, rtol=1e-6)
        np.testing.assert_array_equal(spec.freqs, np.arange(48) + 1)

    def test_2d_log(self):
        arr = np.random.default_rng(1).random((8, 8))
        spec = prepare(arr, CFG.replace(log=True))
        np.testing.assert_allclose(spec.values, np.log1p(arr), rtol=1e-6)

    def test_2d_log_rejects_negative(self):
        with pytest.raises(ValueError, match="negative"):
            prepare(np.full((8, 8), -1.0), CFG.replace(log=True))

    def test_2d_axes_assume_config_hop(self):
        spec = prepare(np.zeros((10, 5)), CFG, fs=FS)
        np.testing.assert_allclose(spec.times, np.arange(5) * 64 / FS)
        assert spec.freqs[0] == pytest.approx(FS / 256)


class TestRejectsBadInput:
    @pytest.mark.parametrize(
        ("data", "match"),
        [
            (np.zeros(64, dtype=complex), "complex"),
            (np.zeros(64, dtype=bool), "numeric"),
            (np.array(["a", "b"]), "numeric"),
            (np.zeros(0), "empty"),
            (np.array([1.0, np.nan, 2.0]), "1 non-finite"),
            (np.array([np.inf, -np.inf]), "2 non-finite"),
            (np.zeros((2, 3, 4)), "ndim=3"),
        ],
    )
    def test_bad_data(self, data, match):
        with pytest.raises(ValueError, match=match):
            prepare(data, CFG)

    @pytest.mark.parametrize("fs", [0, -1.0, float("nan"), float("inf")])
    def test_bad_fs_value(self, signal, fs):
        with pytest.raises(ValueError, match="fs must be"):
            prepare(signal, CFG, fs=fs)

    def test_bad_fs_type(self, signal):
        with pytest.raises(TypeError, match="fs must be a number"):
            prepare(signal, CFG, fs="fast")

    def test_lists_are_accepted(self):
        assert prepare([[1.0, 2.0], [3.0, 4.0]], CFG).shape == (2, 2)

    def test_a_signal_shorter_than_half_a_window_raises(self):
        with pytest.raises(ValueError, match=r"n_fft=256 needs at least 129"):
            prepare(np.zeros(64), CFG)

    @pytest.mark.parametrize("fs", [True, np.bool_(True), "1000", b"1000"], ids=repr)
    def test_bool_and_string_fs_raise(self, signal, fs):
        with pytest.raises(TypeError, match="fs must be a number"):
            prepare(signal, CFG, fs=fs)

    @pytest.mark.parametrize("fs", [np.float32(1000), 1000], ids=repr)
    def test_numeric_fs_types_work(self, signal, fs):
        assert prepare(signal, CFG, fs=fs).fs == 1000.0

    @pytest.mark.parametrize("shape", [(1, 4096), (4096, 1), (1, 1)])
    def test_vector_shaped_2d_input_raises(self, shape):
        with pytest.raises(ValueError, match=r"np\.ravel"):
            prepare(np.ones(shape), CFG)

    def test_pre_1_0_spectrogram_keeps_a_vector_shaped_input(self, monkeypatch):
        import torch.nn as nn

        from tokeye.api import TokEye

        monkeypatch.setattr(
            "tokeye.hub.load_model", lambda source, device="auto": nn.Conv2d(1, 2, 1)
        )
        arr = np.random.default_rng(2).random((1, 64))

        out = TokEye(n_fft=64, hop=16).spectrogram(arr)

        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, arr)


def _inputs(kind: str, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """``(data, reference)`` for one input kind."""
    if kind == "stft":
        return signal, None
    if kind == "cross":
        return signal, np.roll(signal, 3)
    return np.random.default_rng(3).random((48, 40)), None


class TestOneAxisRule:
    @pytest.mark.parametrize("kind", ["stft", "cross", "spectrogram"])
    def test_axes_with_fs_are_the_index_axes_scaled(self, signal, kind):
        data, reference = _inputs(kind, signal)

        idx = prepare(data, CFG, reference=reference)
        hz = prepare(data, CFG, fs=FS, reference=reference)

        assert idx.kind == hz.kind == kind
        assert idx.times[0] == hz.times[0] == 0.0
        np.testing.assert_array_equal(idx.freqs, np.arange(idx.shape[0]) + 1)
        np.testing.assert_array_equal(idx.times, np.arange(idx.shape[1]))
        np.testing.assert_array_equal(hz.freqs, idx.freqs * FS / 256)
        np.testing.assert_array_equal(hz.times, idx.times * 64 / FS)

    def test_2d_rows_without_fs_start_at_bin_0_without_clip_dc(self):
        spec = prepare(np.ones((8, 5)), CFG.replace(clip_dc=False))
        np.testing.assert_array_equal(spec.freqs, np.arange(8))

    @pytest.mark.parametrize(
        ("kind", "expected"), [("stft", 4096), ("cross", 4096), ("spectrogram", None)]
    )
    def test_n_samples_is_the_signal_length(self, signal, kind, expected):
        data, reference = _inputs(kind, signal)
        assert prepare(data, CFG, reference=reference).n_samples == expected
