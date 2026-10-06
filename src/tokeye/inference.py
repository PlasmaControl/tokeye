"""Model inference: standardize, tile when large, forward, sigmoid.

:func:`infer` is the 1.0 entry point. :func:`model_infer`,
:func:`signal_to_spectrogram` and :func:`warmup` keep their pre-1.0
behaviour for existing callers (the app, dev scripts).
"""

from __future__ import annotations

import logging
import numbers
import warnings
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import torch
from tqdm.auto import tqdm

from .transforms import compute_stft

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch.nn as nn

    from ._tiling import TileGrid

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

WARMUP_INPUT_SHAPE = (1, 1, 512, 512)  # (batch_size, channels, height, width)

STD_EPS = 1e-6  # the v1 standardization: (x - mean) / (std + STD_EPS)
MIN_SIZE = 16  # smallest H or W the U-Net accepts
# Tile offsets are multiples of this: 2^4, big_tf_unet's total stride, so
# every pooling level's offset is exact for it and for any BigTFUNetModel
# with num_layers <= 5. A model with a deeper level tiles by plain core-crop,
# with TILE_WARNING.
ALIGN = 16
# Context kept on each side of a tile core: at least the model's
# receptive-field radius (<= 118 px for big_tf_unet). With global-grid
# upsampling, each core then matches an untiled run.
MARGIN = 128
MIN_TILE = 4 * MARGIN  # every core is at least 2 * MARGIN
AUTO_TILE_PIXELS = 2**21  # "auto" runs untiled up to this many pixels
AUTO_TILE_HEIGHT = 1024
TILE_WARNING = (
    "tiled output of this model may differ slightly from an untiled run; "
    "pass tile=None (CLI: --tile none) for untiled output"
)


def infer(
    model: nn.Module,
    values: np.ndarray,
    *,
    device: str | torch.device | None = None,
    tile: int | str | None = "auto",
) -> np.ndarray:
    """Segment one spectrogram.

    Parameters
    ----------
    model
        A segmentation model mapping ``(1, 1, H, W)`` to ``(1, C, H, W)``
        logits (a tuple/list whose first item is that tensor also works).
    values
        ``(H, W)`` spectrogram, log-scaled as the model expects (see
        :func:`tokeye.preprocess.prepare`). Both sides must be >= 16.
    device
        Where to run. ``None`` uses the device the model is on; otherwise
        the model is moved there. If an op is unsupported on MPS, the
        model moves to the CPU with one ``RuntimeWarning``.
    tile
        ``"auto"`` (default) runs untiled up to ``2**21`` pixels and tiles
        larger inputs; an int ``>= 512`` sets the tile side; ``None``
        never tiles. Standardization always uses the whole image. Tiled
        output matches an untiled run to float32 rounding for
        ``big_tf_unet`` and any ``BigTFUNetModel`` of the default depth or
        shallower (``num_layers <= 5``). Other models still tile, and
        TokEye warns (one ``UserWarning``) when it can tell that tiles may
        differ: TorchScript and ``torch.fx`` (``.pt2``) models, other
        ``align_corners=True`` upsampling, deeper models, feature maps that
        do not halve by flooring, and models ``copy.deepcopy`` cannot copy.
        It cannot tell for a model that upsamples through ``F.interpolate``
        inside ``forward``, or whose receptive field is wider than the
        128-px margin.

    Returns
    -------
    numpy.ndarray
        ``(C, H, W)`` float32 sigmoid scores in ``[0, 1]``.
    """
    arr = _check_values(values)
    dev = _device_of(model) if device is None else torch.device(device)
    plan = _plan_tiles(arr.shape, tile)

    def run(target: torch.device) -> tuple[np.ndarray, bool]:
        if device is not None:
            model.to(target)
        return _run(model, arr, plan, target)

    out, exact = _with_cpu_fallback(model, dev, run)
    if not exact:
        warnings.warn(TILE_WARNING, UserWarning, stacklevel=2)
    return out


def _with_cpu_fallback(
    model: nn.Module, device: torch.device, run: Callable[[torch.device], _T]
) -> _T:
    """``run(device)``; if it raises ``RuntimeError`` on MPS, move ``model``
    to the CPU, warn once and rerun there."""
    try:
        return run(device)
    except RuntimeError as exc:
        if device.type != "mps":
            raise
        warnings.warn(
            f"MPS inference failed ({exc}); falling back to CPU",
            RuntimeWarning,
            stacklevel=3,
        )
        model.to("cpu")
        return run(torch.device("cpu"))


def _check_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D (H, W) spectrogram, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
        raise ValueError(f"spectrogram must be real-valued, got dtype {arr.dtype}")
    height, width = arr.shape
    if min(height, width) < MIN_SIZE:
        raise ValueError(
            f"spectrogram is {height}x{width}; TokEye needs at least "
            f"{MIN_SIZE}x{MIN_SIZE} (for a 1D signal: use more samples or a "
            "smaller n_fft/hop)"
        )
    if not np.isfinite(arr).all():
        raise ValueError("spectrogram contains NaN or inf values")
    return arr


def _device_of(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def check_tile(tile: int | str | None) -> None:
    """Raise if ``tile`` is not ``"auto"``, ``None`` or an int >= 512."""
    if tile is None or tile == "auto":
        return
    if isinstance(tile, str):
        raise ValueError(f"tile must be 'auto', None or an int, got {tile!r}")
    if isinstance(tile, bool) or not isinstance(tile, numbers.Integral):
        raise TypeError(f"tile must be 'auto', None or an int, got {tile!r}")
    if tile < MIN_TILE:
        raise ValueError(f"tile must be >= {MIN_TILE}, got {tile}")


def _plan_tiles(
    shape: tuple[int, int], tile: int | str | None
) -> tuple[int, int] | None:
    """Tile ``(height, width)`` for ``shape``, or ``None`` for one pass."""
    check_tile(tile)
    height, width = shape
    if tile is None:
        return None
    if tile == "auto":
        if height * width <= AUTO_TILE_PIXELS:
            return None
        tile_h = min(height, AUTO_TILE_HEIGHT)
        tile_w = max(MIN_TILE, (AUTO_TILE_PIXELS // tile_h) // ALIGN * ALIGN)
        return tile_h, min(width, tile_w)
    plan = min(height, int(tile)), min(width, int(tile))
    return None if plan == (height, width) else plan


def _axis_windows(n: int, size: int) -> list[tuple[int, int, int, int]]:
    """``(lo, hi, core_lo, core_hi)`` windows covering ``range(n)``.

    Cores partition ``[0, n)``; each window adds up to ``MARGIN`` of context
    on both sides and is at most ``size`` long.
    """
    if n <= size:
        return [(0, n, 0, n)]
    core = (size - 2 * MARGIN) // ALIGN * ALIGN
    windows = []
    for start in range(0, n, core):
        end = min(start + core, n)
        windows.append((max(0, start - MARGIN), min(n, end + MARGIN), start, end))
    return windows


def _run(
    model: nn.Module,
    arr: np.ndarray,
    plan: tuple[int, int] | None,
    device: torch.device,
) -> tuple[np.ndarray, bool]:
    """Standardize ``arr`` as a whole, then segment it in one pass or tiles.

    Returns ``(scores, exact)``. ``exact`` is ``False`` when a tiled run
    fell back to plain core cropping on ``model`` itself: no tile-exact view
    could be built (see :func:`tokeye._tiling.tiling_view`), or the view met
    a feature map it cannot place (:class:`tokeye._tiling.TileGeometryError`)
    and the tiles were rerun.
    """
    mean = float(arr.mean(dtype=np.float64))
    scale = float(arr.std(dtype=np.float64)) + STD_EPS
    height, width = arr.shape
    if plan is None:
        rows, cols = [(0, height, 0, height)], [(0, width, 0, width)]
        return _segment(model, arr, mean, scale, rows, cols, device), True
    from . import _tiling

    rows, cols = _axis_windows(height, plan[0]), _axis_windows(width, plan[1])
    grid = _tiling.TileGrid((height, width), align=ALIGN)
    view = _tiling.tiling_view(model, grid)
    if view is not None:
        try:
            return _segment(view, arr, mean, scale, rows, cols, device, grid), True
        except _tiling.TileGeometryError as exc:
            logger.debug("tile-exact view unavailable: %s", exc)
    return _segment(model, arr, mean, scale, rows, cols, device), False


def _segment(
    net: nn.Module,
    arr: np.ndarray,
    mean: float,
    scale: float,
    rows: list[tuple[int, int, int, int]],
    cols: list[tuple[int, int, int, int]],
    device: torch.device,
    grid: TileGrid | None = None,
) -> np.ndarray:
    """``net``'s scores for ``arr``, window by window, keeping each core.

    ``rows`` and ``cols`` are :func:`_axis_windows` lists; each block is
    standardized as ``(block - mean) / scale``. ``grid``, when given, is
    moved to each window before that window's forward pass.
    """
    height, width = arr.shape
    out: np.ndarray | None = None
    with torch.inference_mode():
        for r_lo, r_hi, rc_lo, rc_hi in rows:
            for c_lo, c_hi, cc_lo, cc_hi in cols:
                if grid is not None:
                    grid.offset = (r_lo, c_lo)
                    grid.window = (r_hi - r_lo, c_hi - c_lo)
                block = arr[r_lo:r_hi, c_lo:c_hi].astype(np.float64)
                x = ((block - mean) / scale).astype(np.float32)
                tensor = torch.from_numpy(x)[None, None].to(device)
                probs = torch.sigmoid(_unwrap(net(tensor))).float().cpu().numpy()
                if probs.shape[1:] != block.shape:
                    raise ValueError(
                        f"model output {probs.shape[1:]} does not match its "
                        f"input {block.shape}"
                    )
                if out is None:
                    out = np.empty((probs.shape[0], height, width), dtype=np.float32)
                out[:, rc_lo:rc_hi, cc_lo:cc_hi] = probs[
                    :, rc_lo - r_lo : rc_hi - r_lo, cc_lo - c_lo : cc_hi - c_lo
                ]
    return out


def _unwrap(output: object) -> torch.Tensor:
    """``(C, H, W)`` logits from a model's raw output."""
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("the model returned an empty output")
        output = output[0]
    if isinstance(output, dict):
        if "boxes" in output or "labels" in output:
            raise ValueError(
                "the model returned detections, not a segmentation mask; for "
                "ae_tf_maskrcnn use `tokeye alfvenspec` or tokeye.alfvenspec.detect"
            )
        raise TypeError(
            f"unexpected model output: dict with keys {sorted(output, key=str)}"
        )
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"unexpected model output type {type(output).__name__}")
    if output.ndim == 4:
        output = output[0]
    if output.ndim != 3:
        raise ValueError(
            f"expected a (B, C, H, W) model output, got shape {tuple(output.shape)}"
        )
    return output


def model_infer(
    inp_array: np.ndarray | None,
    model: nn.Module | None,
) -> np.ndarray | None:
    """Pre-1.0 inference helper: one untiled pass on the model's device.

    Unlike :func:`infer`, it does no input validation (shape, dtype,
    minimum size or finiteness). ``None`` for either argument passes
    through: it returns ``None`` and logs a warning. It has the same
    MPS->CPU fallback as :func:`infer`.

    Returns
    -------
    numpy.ndarray or None
        ``(C, H, W)`` sigmoid scores, or ``(H, W)`` for a one-channel model.
    """
    if inp_array is None or model is None:
        logger.warning("Missing input or model for inference")
        return None

    arr = np.asarray(inp_array)
    logger.info("Running inference on input shape: %s", arr.shape)
    out, _ = _with_cpu_fallback(
        model, _device_of(model), lambda target: _run(model, arr, None, target)
    )
    return out[0] if out.shape[0] == 1 else out


def signal_to_spectrogram(signal: np.ndarray, **stft_kwargs) -> np.ndarray:
    """Expand a 1D signal to (1, N) and run it through ``compute_stft``."""
    signal_data = np.expand_dims(signal, axis=0)
    return compute_stft(signal_data, **stft_kwargs)


def warmup(model: nn.Module, iterations: int = 10) -> None:
    """Run dummy forward passes to trigger lazy init / kernel autotuning."""
    device = _device_of(model)
    dummy_input = torch.randn(*WARMUP_INPUT_SHAPE, device=device, dtype=torch.float32)
    with torch.inference_mode():
        for _ in tqdm(range(iterations), disable=None):  # no bar off a terminal
            _ = model(dummy_input)
