"""Tiled inference reproduces an untiled run (global-grid upsampling)."""

from __future__ import annotations

import io
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokeye import _tiling
from tokeye.inference import _plan_tiles, infer
from tokeye.models.big_tf_unet.config_big_tf_unet import BigTFUNetConfig
from tokeye.models.big_tf_unet.model_big_tf_unet import BigTFUNetModel

pytestmark = pytest.mark.usefixtures("few_threads")

SHAPES = [
    (300, 1300),  # columns only
    (700, 900),  # both axes
    (531, 1037),  # odd sizes on both axes: the decoder's F.pad path
]
CONFIG = BigTFUNetConfig(first_layer_size=4, dropout_rate=0.0)


def _narrow_unet() -> BigTFUNetModel:
    """The shipped architecture, 8x narrower (about 64x fewer FLOPs).

    He-initialized: under PyTorch's default init the activations' variance
    drops about six-fold per conv, so an untrained net's output is nearly
    flat and even the old core-crop tiler stays within 1e-4 of an untiled
    run. He init keeps the variance, so a tiling error shows.
    """
    torch.manual_seed(0)
    model = BigTFUNetModel(CONFIG)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity="leaky_relu")
    return model.eval()


@pytest.fixture(scope="module")
def unet() -> BigTFUNetModel:
    return _narrow_unet()


def _input(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    for row in rng.integers(0, shape[0], 6):
        x[row] += 4.0  # a few bright rows, so the output is not flat
    return x


def _max_diff(model: nn.Module, x: np.ndarray) -> float:
    return float(np.abs(infer(model, x, tile=512) - infer(model, x, tile=None)).max())


@pytest.mark.parametrize("shape", SHAPES)
def test_tiled_equals_untiled(unet, shape):
    assert _plan_tiles(shape, 512) is not None
    assert _max_diff(unet, _input(shape)) <= 1e-5


@pytest.mark.parametrize("shape", SHAPES)
def test_the_comparison_sees_the_old_tiling_error(unet, shape, monkeypatch):
    monkeypatch.setattr(_tiling, "tiling_view", lambda model, grid: model)
    assert _max_diff(unet, _input(shape)) > 1e-4


def _grid(full, offset=(0, 0), window=None) -> _tiling.TileGrid:
    grid = _tiling.TileGrid(full)
    grid.offset = offset
    grid.window = full if window is None else window
    return grid


@pytest.mark.parametrize("shape", [(1, 3, 17, 33), (2, 2, 1, 6), (1, 1, 1, 1)])
def test_upsample_matches_interpolate(shape):
    x = torch.randn(*shape, generator=torch.Generator().manual_seed(0))
    expected = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
    up = _tiling.GlobalGridUpsample(_grid(shape[2:]))
    torch.testing.assert_close(up(x), expected, rtol=0, atol=1e-6)


def test_upsample_of_a_window_matches_the_whole_image():
    # A window at pooling level 1: full image 80x144, window rows 16..64 and
    # columns 32..144 (it reaches the right edge).
    whole = torch.randn(2, 5, 40, 72, generator=torch.Generator().manual_seed(1))
    expected = F.interpolate(whole, scale_factor=2, mode="bilinear", align_corners=True)
    grid = _grid((80, 144), offset=(16, 32), window=(48, 112))
    out = _tiling.global_grid_upsample(whole[:, :, 8:32, 16:72], grid)
    assert out.shape == (2, 5, 48, 112)
    # Only the first and last row and the first column read taps outside the
    # window (clamped); the right edge is the image's own.
    torch.testing.assert_close(
        out[:, :, 1:-1, 1:], expected[:, :, 17:63, 33:144], rtol=0, atol=1e-6
    )


def test_upsample_rejects_inconsistent_geometry():
    x = torch.zeros(1, 1, 12, 20)  # pooling level 2 of a 48x80 window
    with pytest.raises(RuntimeError, match="not a pooling level"):
        _tiling.global_grid_upsample(x, _grid((80, 144), window=(56, 80)))
    with pytest.raises(RuntimeError, match="not a multiple of 2"):
        _tiling.global_grid_upsample(x, _grid((80, 144), (6, 0), (48, 80)))


def test_view_shares_weights_and_swaps_the_upsamples(unet):
    view = _tiling.tiling_view(unet, _grid((64, 64)))
    assert view is not unet
    assert all(
        a is b for a, b in zip(view.parameters(), unet.parameters(), strict=True)
    )
    assert all(a is b for a, b in zip(view.buffers(), unet.buffers(), strict=True))
    swapped = [m for m in view.modules() if isinstance(m, _tiling.GlobalGridUpsample)]
    assert len(swapped) == CONFIG.num_layers - 1
    assert not any(isinstance(m, nn.Upsample) for m in view.modules())


def test_the_callers_model_is_untouched(unet):
    ids = [id(m) for m in unet.modules()]
    ups = [m for m in unet.modules() if isinstance(m, nn.Upsample)]
    x = _input((300, 1300))
    before = infer(unet, x, tile=None)

    infer(unet, x, tile=512)

    assert [id(m) for m in unet.modules()] == ids
    assert [m for m in unet.modules() if isinstance(m, nn.Upsample)] == ups
    for module in unet.modules():
        assert not module._forward_hooks
        assert not module._forward_pre_hooks
    buffer = io.BytesIO()
    torch.save(unet.state_dict(), buffer)
    buffer.seek(0)
    fresh = BigTFUNetModel(CONFIG)
    fresh.load_state_dict(torch.load(buffer, weights_only=True))
    np.testing.assert_array_equal(infer(fresh.eval(), x, tile=None), before)


def test_concurrent_tiled_runs_match_serial_runs(unet):
    inputs = [_input((700, 900), seed=1), _input((531, 1037), seed=2)]
    serial = [infer(unet, x, tile=512) for x in inputs]
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(lambda x: infer(unet, x, tile=512), inputs))
    for got, expected in zip(parallel, serial, strict=True):
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# Models that cannot be made tile-exact still tile, with one warning
# ---------------------------------------------------------------------------


class _Interpolates(nn.Module):
    """Conv, 2x average pool, then ``F.interpolate`` back to the input size."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.avg_pool2d(self.conv(x), 2)
        size = [x.shape[2], x.shape[3]]
        return F.interpolate(y, size=size, mode="bilinear", align_corners=True)


class _PoolUp(nn.Module):
    """Conv, 2x average pool, then ``up`` (for even input sides)."""

    def __init__(self, up: nn.Module) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.up = up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.avg_pool2d(self.conv(x), 2))


def _bilinear_x2() -> nn.Upsample:
    return nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)


def _uncopyable() -> nn.Module:
    model = _PoolUp(_bilinear_x2())
    model.lock = threading.Lock()  # copy.deepcopy cannot copy a lock
    return model


def _tile_warnings(model: nn.Module, x: np.ndarray, tile) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        infer(model, x, tile=tile)
    return [str(w.message) for w in caught if w.category is UserWarning]


@pytest.mark.parametrize(
    "build",
    [
        lambda: torch.jit.script(_Interpolates()),
        lambda: torch.fx.symbolic_trace(_PoolUp(_bilinear_x2())),
        lambda: _PoolUp(
            nn.Upsample(scale_factor=2, mode="bicubic", align_corners=True)
        ),
        _uncopyable,
    ],
    ids=["torchscript", "fx", "bicubic", "deepcopy-fails"],
)
def test_models_that_cannot_be_made_exact_warn_once(build):
    torch.manual_seed(0)
    model = build().eval()
    x = _input((40, 1024))
    assert _tile_warnings(model, x, 512) == [
        "tiled output of this model may differ slightly from an untiled run; "
        "pass tile=None (CLI: --tile none) for untiled output"
    ]
    assert _tile_warnings(model, x, None) == []
    assert _tile_warnings(model, np.zeros((40, 400)), 512) == []  # fits one tile


def test_models_without_align_corners_upsampling_tile_as_is():
    torch.manual_seed(0)
    stub = nn.Conv2d(1, 2, kernel_size=3, padding=1).eval()
    assert _tiling.tiling_view(stub, _grid((64, 64))) is stub
    assert _tile_warnings(stub, _input((40, 1024)), 512) == []


def test_small_eager_models_with_the_x2_upsample_are_exact():
    torch.manual_seed(0)
    model = _PoolUp(_bilinear_x2()).eval()
    x = _input((40, 1024))
    assert _tile_warnings(model, x, 512) == []
    assert _max_diff(model, x) <= 1e-6


def test_a_tile_covering_the_input_runs_untiled():
    assert _plan_tiles((300, 400), 512) is None
    assert _plan_tiles((300, 1300), 512) == (300, 512)
