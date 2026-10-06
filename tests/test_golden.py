"""Numbers guard: the default path must reproduce the pre-1.0 output.

Needs the real ``big_tf_unet`` weights in the HF cache (``tokeye download``);
skipped otherwise. See ``golden_utils.py`` for how the golden was made.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from golden_utils import (
    GOLDEN_PATH,
    _atol,
    centre_crop,
    channel_stats,
    crop_origin,
    fp32_cuda,
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


def _compute(device: str) -> tuple[np.ndarray, np.ndarray]:
    """The spectrogram stats and the mask of the golden input on ``device``."""
    from tokeye import TokEye

    eye = TokEye(device=device)
    signal = golden_inputs()
    stats = spec_stats(eye.spectrogram(signal))
    if device == "cuda":
        with fp32_cuda():  # no TF32 on Ampere and newer
            mask = eye.predict(signal)
    else:
        mask = eye.predict(signal)
    return stats, mask


@pytest.fixture(scope="module", params=DEVICES)
def result(request, real_weights, tmp_path_factory):
    """``(device, spectrogram stats, mask)``, computed once per device.

    Under pytest-xdist the first worker to get here computes it and saves it
    beside the workers' temporary directories; the others load that file
    (the pytest-xdist recipe for fixtures that execute only once).
    """
    device = request.param
    if "PYTEST_XDIST_WORKER" not in os.environ:
        return (device, *_compute(device))

    from filelock import FileLock

    run = os.environ.get("PYTEST_XDIST_TESTRUNUID", "run")
    path = tmp_path_factory.getbasetemp().parent / f"golden_{run}_{device}.npz"
    with FileLock(str(path) + ".lock"):
        if path.exists():
            with np.load(path) as data:
                stats, mask = data["spec_stats"], data["mask"]
        else:
            stats, mask = _compute(device)
            tmp = path.with_suffix(".part")
            with tmp.open("wb") as fh:  # np.savez appends .npz to a path
                np.savez(fh, spec_stats=stats, mask=mask)
            tmp.replace(path)  # os.replace: atomic, no truncated file
    return device, stats, mask


def test_cached_weights_are_the_golden_weights(real_weights, golden):
    assert sha256(real_weights) == str(golden["weights_sha256"]), (
        "default weights changed; regenerate the golden on purpose "
        "(python tests/golden_utils.py --write golden) and add a CHANGELOG entry"
    )


def test_shape(result, golden):
    _, _, mask = result
    assert mask.shape == tuple(golden["shape"])


def test_crop_origin(result, golden):
    _, _, mask = result
    assert crop_origin(mask.shape[1:]) == tuple(int(v) for v in golden["crop_origin"])


def test_spectrogram_stats(result, golden):
    _, stats, _ = result
    np.testing.assert_allclose(stats, golden["spec_stats"], rtol=1e-6)


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
