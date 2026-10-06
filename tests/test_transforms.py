from __future__ import annotations

import numpy as np
import pytest

from tokeye.transforms import compute_stft


def _sine_signal(n_samples: int = 4096, freq: float = 0.05) -> np.ndarray:
    t = np.arange(n_samples)
    return np.sin(2 * np.pi * freq * t).astype(np.float64)[np.newaxis, :]


def test_compute_stft_output_is_2d():
    arr = _sine_signal()
    out = compute_stft(arr, n_fft=256, hop=64)
    assert out.ndim == 2


def test_clip_dc_removes_one_frequency_row():
    arr = _sine_signal()
    with_dc_clip = compute_stft(arr, n_fft=256, hop=64, clip_dc=True)
    without_dc_clip = compute_stft(arr, n_fft=256, hop=64, clip_dc=False)
    assert with_dc_clip.shape[0] == without_dc_clip.shape[0] - 1
    assert with_dc_clip.shape[1] == without_dc_clip.shape[1]


def test_expected_frame_count():
    # Frames are centred on samples 0, hop, 2*hop, ... (torch.stft's
    # center=True, the training framing), so 4096 samples at hop=256 give
    # 1 + 4096 // 256 = 17 frames; pin that number so a change to the
    # padding/framing convention is caught as a regression.
    n_samples = 4096
    n_fft = 1024
    hop = 256
    arr = _sine_signal(n_samples=n_samples)
    out = compute_stft(arr, n_fft=n_fft, hop=hop)
    assert out.shape[1] == 17


_UNCLIPPED = {"clip_dc": False, "clip_low": 0.0, "clip_high": 100.0}


def _torch_stft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """The training framing: ``torch.stft`` with centred, reflect-padded frames."""
    import torch

    return torch.stft(
        torch.from_numpy(x),
        n_fft,
        hop_length=hop,
        window=torch.hann_window(n_fft, dtype=torch.float64),
        center=True,
        pad_mode="reflect",
        return_complex=True,
    ).numpy()


@pytest.mark.parametrize(
    ("n_fft", "hop", "n_samples"),
    [
        (1024, 128, 4096),
        (256, 64, 1000),  # N not a multiple of hop
        (255, 64, 1000),  # odd n_fft
        (256, 64, 129),  # the minimum length
        (64, 64, 33),
    ],
)
def test_compute_stft_equals_torch_stft(n_fft, hop, n_samples):
    x = np.random.default_rng(7).standard_normal(n_samples)

    out = compute_stft(x, n_fft=n_fft, hop=hop, **_UNCLIPPED)

    expected = np.log1p(np.abs(_torch_stft(x, n_fft, hop)))
    assert out.shape == expected.shape
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)


def test_cross_power_equals_torch_stft():
    rng = np.random.default_rng(8)
    x, y = rng.standard_normal(1000), rng.standard_normal(1000)

    out = compute_stft(np.stack([x, y]), n_fft=256, hop=64, **_UNCLIPPED)

    cross = _torch_stft(x, 256, 64) * np.conj(_torch_stft(y, 256, 64))
    expected = np.log1p(np.abs(cross))
    assert out.shape == expected.shape
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)


def test_a_click_at_sample_k_hop_peaks_in_column_k():
    x = np.zeros(4096)
    x[8 * 64] = 1.0

    spec = compute_stft(x, n_fft=512, hop=64, **_UNCLIPPED)

    assert int(spec.max(axis=0).argmax()) == 8
    for d in (1, 2, 3):
        assert spec[:, 8 - d].max() > 0
        np.testing.assert_allclose(spec[:, 8 - d], spec[:, 8 + d], atol=1e-12)


def test_a_signal_shorter_than_half_a_window_raises():
    with pytest.raises(ValueError, match=r"n_fft=256 needs at least 129"):
        compute_stft(np.zeros(128), n_fft=256, hop=64)


def test_the_minimum_length_gives_one_frame_per_hop_plus_one():
    assert compute_stft(np.zeros(129), n_fft=256, hop=64).shape[1] == 3


def test_values_within_percentile_clip_bounds():
    # A signal with an extreme outlier burst has a much wider raw dynamic
    # range than its 1st-99th percentile band; percentile clipping should
    # compress the output range to (approximately) that narrower band.
    rng = np.random.default_rng(0)
    arr = rng.normal(size=4096)
    arr[2000] = 1000.0  # single extreme outlier
    arr = arr[np.newaxis, :].astype(np.float64)

    clip_low, clip_high = 1.0, 99.0
    clipped = compute_stft(
        arr, n_fft=256, hop=64, clip_low=clip_low, clip_high=clip_high
    )
    unclipped = compute_stft(arr, n_fft=256, hop=64, clip_low=0.0, clip_high=100.0)

    assert np.all(np.isfinite(clipped))
    assert clipped.min() >= unclipped.min()
    assert clipped.max() <= unclipped.max()
    assert clipped.max() < unclipped.max()  # outlier should be visibly clipped


def test_1d_and_single_row_inputs_agree():
    arr = _sine_signal()
    np.testing.assert_array_equal(
        compute_stft(arr[0], n_fft=256, hop=64), compute_stft(arr, n_fft=256, hop=64)
    )


@pytest.mark.parametrize("shape", [(3, 4096), (1, 2, 4096), (0, 4096)])
def test_unsupported_shapes_raise(shape):
    # 3+ rows used to silently drop the extra rows (audit bug A7).
    with pytest.raises(ValueError, match="compute_stft expects"):
        compute_stft(np.zeros(shape), n_fft=256, hop=64)


def test_defaults_come_from_the_config():
    from tokeye import transforms
    from tokeye.config import DEFAULT_CONFIG

    assert transforms.DEFAULT_HOP == DEFAULT_CONFIG.hop == 128
    assert transforms.DEFAULT_N_FFT == 1024
    assert transforms.DEFAULT_WINDOW == "hann"
