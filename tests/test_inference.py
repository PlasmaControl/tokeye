from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from tokeye.inference import (
    ALIGN,
    MARGIN,
    STD_EPS,
    _axis_windows,
    _plan_tiles,
    infer,
    model_infer,
    signal_to_spectrogram,
    warmup,
)


class _IdentityLikeModel(nn.Module):
    """Returns a 1-tuple of (B, 2, H, W), mimicking BigTFUNetModel's output."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.conv(x),)


def test_model_infer_output_shape_and_range():
    model = _IdentityLikeModel()
    model.eval()
    inp = np.random.default_rng(0).normal(size=(64, 64)).astype(np.float32)

    out = model_infer(inp, model)

    assert out.shape == (2, 64, 64)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)  # sigmoid range


def test_model_infer_none_input_returns_none():
    model = _IdentityLikeModel()
    assert model_infer(None, model) is None


def test_model_infer_none_model_returns_none():
    inp = np.zeros((32, 32), dtype=np.float32)
    assert model_infer(inp, None) is None


def test_signal_to_spectrogram_returns_2d():
    signal = np.random.default_rng(0).normal(size=4096).astype(np.float64)
    spec = signal_to_spectrogram(signal, n_fft=256, hop=64)
    assert spec.ndim == 2


def test_warmup_runs_without_error():
    model = _IdentityLikeModel()
    model.eval()
    warmup(model, iterations=2)  # should not raise


# ---------------------------------------------------------------------------
# infer (1.0)
# ---------------------------------------------------------------------------


class _Echo(nn.Module):
    """One-channel model that returns its (standardized) input."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _DictModel(_Echo):
    def forward(self, x: torch.Tensor):
        return [{"boxes": torch.zeros(0, 4)}]


def _conv_stub(kernel: int) -> nn.Module:
    torch.manual_seed(0)
    return nn.Conv2d(1, 2, kernel_size=kernel, padding=kernel // 2).eval()


def test_infer_returns_float32_chw_in_unit_range():
    x = np.random.default_rng(0).normal(size=(64, 80))
    out = infer(_IdentityLikeModel(), x)
    assert out.shape == (2, 64, 80)
    assert out.dtype == np.float32
    assert np.all((out >= 0) & (out <= 1))


def test_infer_uses_the_v1_standardization():
    x = np.random.default_rng(1).normal(3.0, 2.0, size=(32, 48))
    out = infer(_Echo(), x)
    expected = 1 / (1 + np.exp(-(x - x.mean()) / (x.std() + STD_EPS)))
    np.testing.assert_allclose(out[0], expected, atol=1e-6)


def test_infer_untiled_matches_model_infer():
    x = np.random.default_rng(2).normal(size=(64, 64))
    model = _IdentityLikeModel()
    np.testing.assert_array_equal(infer(model, x, tile=None), model_infer(x, model))


@pytest.mark.parametrize("kernel", [1, 3])
@pytest.mark.parametrize(("shape", "tile"), [((40, 3000), 300), ((700, 900), 400)])
def test_tiled_equals_untiled_for_local_models(kernel, shape, tile):
    x = np.random.default_rng(3).normal(size=shape)
    model = _conv_stub(kernel)
    np.testing.assert_allclose(
        infer(model, x, tile=tile), infer(model, x, tile=None), atol=1e-6
    )


@pytest.mark.parametrize(
    ("n", "size"), [(100, 300), (3000, 300), (1000, 272), (4097, 4096)]
)
def test_axis_windows_partition_the_axis(n, size):
    windows = _axis_windows(n, size)
    cores = [(c_lo, c_hi) for _, _, c_lo, c_hi in windows]
    assert cores[0][0] == 0
    assert cores[-1][1] == n
    assert all(a[1] == b[0] for a, b in zip(cores, cores[1:], strict=False))
    for lo, hi, c_lo, c_hi in windows:
        assert hi - lo <= size
        assert lo % ALIGN == 0
        assert c_lo - lo == min(MARGIN, c_lo)
        assert hi - c_hi == min(MARGIN, n - c_hi)


def test_plan_tiles():
    assert _plan_tiles((512, 3132), "auto") is None  # the example stays untiled
    assert _plan_tiles((512, 8192), "auto") == (512, 4096)
    assert _plan_tiles((2048, 4096), "auto") == (1024, 2048)
    assert _plan_tiles((512, 8192), None) is None
    assert _plan_tiles((512, 8192), 1024) == (512, 1024)
    with pytest.raises(ValueError, match=">= 272"):
        _plan_tiles((512, 8192), 100)
    with pytest.raises(ValueError, match="'auto'"):
        _plan_tiles((512, 8192), "big")
    with pytest.raises(TypeError):
        _plan_tiles((512, 8192), True)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.zeros(64), "2D"),
        (np.zeros((8, 100)), "at least 16x16"),
        (np.full((32, 32), np.nan), "NaN"),
    ],
)
def test_infer_rejects_bad_values(values, match):
    with pytest.raises(ValueError, match=match):
        infer(_IdentityLikeModel(), values)


def test_infer_rejects_detection_models_with_a_hint():
    with pytest.raises(ValueError, match="alfvenspec"):
        infer(_DictModel(), np.zeros((32, 32)))


def test_single_channel_models():
    x = np.random.default_rng(4).normal(size=(32, 32))
    assert infer(_Echo(), x).shape == (1, 32, 32)
    assert model_infer(x, _Echo()).shape == (32, 32)  # pre-1.0 squeeze kept


@pytest.mark.skipif(torch.backends.mps.is_available(), reason="MPS is real here")
def test_mps_failure_falls_back_to_cpu_once():
    # Without MPS, moving a model to "mps" raises RuntimeError -- the same
    # path an unsupported MPS op takes.
    model = _IdentityLikeModel()
    x = np.random.default_rng(5).normal(size=(32, 32))
    expected = infer(model, x, device="cpu")

    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        out = infer(model, x, device="mps")

    np.testing.assert_allclose(out, expected)
    assert next(model.parameters()).device.type == "cpu"


@pytest.mark.weights
def test_tiling_error_is_bounded_on_the_example(real_weights):
    from tokeye import hub
    from tokeye.examples import make_example_signal
    from tokeye.preprocess import prepare

    model = hub.load_model(hub.DEFAULT_MODEL, "cpu")
    values = prepare(make_example_signal()).values
    untiled = infer(model, values, tile=None)
    tiled = infer(model, values, tile=1024)

    diff = np.abs(tiled - untiled)
    assert diff.mean() < 2e-3
    assert np.quantile(diff, 0.999) < 0.08
    assert ((tiled >= 0.5) != (untiled >= 0.5)).mean() < 1e-3
