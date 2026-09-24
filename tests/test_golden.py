"""Numbers guard: the default path must reproduce the pre-1.0 output.

Needs the real ``big_tf_unet`` weights in the HF cache (``tokeye download``);
skipped otherwise. See ``golden_utils.py`` for how the golden was made.
"""

from __future__ import annotations

import platform
import sys

import numpy as np
import pytest
import torch
from golden_utils import (
    GOLDEN_PATH,
    centre_crop,
    channel_stats,
    golden_inputs,
    pool,
    sha256,
    spec_stats,
)

pytestmark = pytest.mark.weights

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if torch.backends.mps.is_available():
    DEVICES.append("mps")


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    with np.load(GOLDEN_PATH) as data:
        return dict(data)


def _atol(device: str, golden: dict[str, np.ndarray]) -> float:
    """1e-5 on the platform the golden was made on, 1e-4 elsewhere."""
    base = torch.__version__.split("+")[0]
    golden_base = str(golden["torch_version"]).split("+")[0]
    exact = (
        device == "cpu"
        and sys.platform.startswith("linux")
        and platform.machine() == "x86_64"
        and base == golden_base
    )
    return 1e-5 if exact else 1e-4


@pytest.fixture(scope="module", params=DEVICES)
def result(request, real_weights):
    from tokeye import TokEye

    eye = TokEye(device=request.param)
    signal = golden_inputs()
    return request.param, eye.spectrogram(signal), eye.predict(signal)


def test_cached_weights_are_the_golden_weights(real_weights, golden):
    assert sha256(real_weights) == str(golden["weights_sha256"]), (
        "default weights changed; regenerate the golden on purpose "
        "(python tests/golden_utils.py --write) and add a CHANGELOG entry"
    )


def test_shape(result, golden):
    _, _, mask = result
    assert mask.shape == tuple(golden["shape"])


def test_spectrogram_stats(result, golden):
    _, spec, _ = result
    np.testing.assert_allclose(spec_stats(spec), golden["spec_stats"], rtol=1e-6)


def test_channel_stats(result, golden):
    device, _, mask = result
    stats = channel_stats(mask)
    atol = _atol(device, golden)
    np.testing.assert_allclose(stats[:, :4], golden["channel_stats"][:, :4], atol=atol)
    np.testing.assert_allclose(stats[:, 4], golden["channel_stats"][:, 4], atol=1e-4)


def test_pooled_mask(result, golden):
    device, _, mask = result
    np.testing.assert_allclose(pool(mask), golden["pool"], atol=_atol(device, golden))


def test_centre_crop(result, golden):
    device, _, mask = result
    np.testing.assert_allclose(
        centre_crop(mask), golden["crop"], atol=_atol(device, golden)
    )
