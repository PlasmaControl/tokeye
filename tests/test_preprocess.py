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
        np.testing.assert_allclose(np.diff(spec.times), 64 / FS)

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
        np.testing.assert_array_equal(spec.freqs, np.arange(48))

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
