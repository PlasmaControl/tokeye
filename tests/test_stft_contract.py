"""Pins ``compute_stft`` for a 1D and a 2-row (cross-power) input."""

from __future__ import annotations

import numpy as np
import pytest
from golden_utils import STFT_KW, STFT_PATH, stft_inputs

from tokeye.transforms import compute_stft


@pytest.fixture(scope="module")
def contract() -> dict[str, np.ndarray]:
    with np.load(STFT_PATH) as data:
        return dict(data)


def _check(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_1d_input(contract):
    x, _ = stft_inputs()
    _check(compute_stft(x, **STFT_KW), contract["one"])


def test_single_row_input(contract):
    x, _ = stft_inputs()
    _check(compute_stft(x[np.newaxis], **STFT_KW), contract["one"])


def test_two_row_cross_power(contract):
    x, y = stft_inputs()
    _check(compute_stft(np.stack([x, y]), **STFT_KW), contract["cross"])
