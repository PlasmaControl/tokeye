from __future__ import annotations

import re
import warnings

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
@pytest.mark.parametrize(("shape", "tile"), [((40, 3000), 512), ((700, 900), 512)])
def test_tiled_equals_untiled_for_local_models(kernel, shape, tile):
    x = np.random.default_rng(3).normal(size=shape)
    model = _conv_stub(kernel)
    np.testing.assert_allclose(
        infer(model, x, tile=tile), infer(model, x, tile=None), atol=1e-6
    )


@pytest.mark.parametrize(
    ("n", "size"), [(100, 300), (3000, 300), (1000, 512), (4097, 4096)]
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
    with pytest.raises(ValueError, match=">= 512"):
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
        (np.zeros((32, 32), dtype=complex), "real-valued"),
    ],
)
def test_infer_rejects_bad_values(values, match):
    with pytest.raises(ValueError, match=match):
        infer(_IdentityLikeModel(), values)


def test_infer_rejects_detection_models_with_a_hint():
    with pytest.raises(ValueError, match="alfvenspec"):
        infer(_DictModel(), np.zeros((32, 32)))


class _Returns(_Echo):
    """A model that returns a fixed ``output``."""

    def __init__(self, output: object) -> None:
        super().__init__()
        self.output = output

    def forward(self, x: torch.Tensor):
        return self.output


@pytest.mark.parametrize(
    ("output", "error", "match"),
    [
        ((), ValueError, "the model returned an empty output"),
        ([], ValueError, "the model returned an empty output"),
        ({"boxes": torch.zeros(0, 4)}, ValueError, "alfvenspec"),
        ([{"labels": torch.zeros(0)}], ValueError, "alfvenspec"),
        (
            {"out": torch.zeros(1, 1, 32, 32), "aux": None},
            TypeError,
            re.escape("unexpected model output: dict with keys ['aux', 'out']"),
        ),
        ([{"masks": None}], TypeError, re.escape("dict with keys ['masks']")),
    ],
)
def test_infer_rejects_unusable_outputs(output, error, match):
    with pytest.raises(error, match=match) as info:
        infer(_Returns(output), np.zeros((32, 32)))
    if error is TypeError:
        assert "alfvenspec" not in str(info.value)


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

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = infer(model, x, device="mps")
    assert [w.category for w in caught] == [RuntimeWarning]
    assert re.fullmatch(
        r"MPS inference failed \(.+\); falling back to CPU", str(caught[0].message)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        again = infer(model, x)
    assert caught == []

    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(again, expected)
    assert next(model.parameters()).device.type == "cpu"


@pytest.mark.skipif(torch.backends.mps.is_available(), reason="MPS is real here")
def test_model_infer_falls_back_to_cpu(monkeypatch):
    model = _IdentityLikeModel()
    x = np.random.default_rng(6).normal(size=(32, 32)).astype(np.float32)
    expected = model_infer(x, model)
    monkeypatch.setattr(
        "tokeye.inference._device_of", lambda model: torch.device("mps")
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = model_infer(x, model)

    assert [w.category for w in caught] == [RuntimeWarning]
    assert "falling back to CPU" in str(caught[0].message)
    np.testing.assert_allclose(out, expected)
    assert next(model.parameters()).device.type == "cpu"


def _assert_matches_untiled(tiled: np.ndarray, untiled: np.ndarray) -> None:
    assert np.abs(tiled - untiled).max() <= 1e-5
    decided = np.abs(untiled - 0.5) > 1e-5
    assert not ((tiled >= 0.5) != (untiled >= 0.5))[decided].any()


@pytest.mark.weights
def test_tiled_matches_untiled_on_the_example(real_weights):
    from tokeye import hub
    from tokeye.examples import make_example_signal
    from tokeye.preprocess import prepare

    model = hub.load_model(hub.DEFAULT_MODEL, "cpu")
    values = prepare(make_example_signal()).values
    tiled = infer(model, values, tile=1024)
    _assert_matches_untiled(tiled, infer(model, values, tile=None))


@pytest.mark.weights
def test_auto_tiles_a_long_input_exactly(real_weights):
    from tokeye import hub
    from tokeye.examples import make_example_signal
    from tokeye.preprocess import prepare

    model = hub.load_model(hub.DEFAULT_MODEL, "cpu")
    values = prepare(make_example_signal(duration_s=2.8)).values
    assert _plan_tiles(values.shape, "auto") == (512, 4096)
    _assert_matches_untiled(infer(model, values), infer(model, values, tile=None))
