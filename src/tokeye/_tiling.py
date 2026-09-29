"""Tile-exact inference: bilinear upsampling on the whole image's grid.

With ``align_corners=True``, where a bilinear upsample samples its input
depends on the input's size, so a tile interpolates on a different grid than
the whole image does -- everywhere in the tile, not only near its edges.
:func:`tiling_view` gives :func:`tokeye.inference.infer` a copy of the model
whose x2 bilinear ``align_corners=True`` upsamples sample on the whole
image's grid instead, so every tile core reproduces an untiled run to
float32 rounding.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

CHUNK = 16  # most channels one upsample interpolates at a time


class TileGrid:
    """Where the current window sits in the whole image.

    One tiled run owns it and updates ``offset`` and ``window`` before each
    window's forward pass.

    Parameters
    ----------
    full
        ``(H, W)`` of the whole image.
    """

    __slots__ = ("full", "offset", "window")

    def __init__(self, full: tuple[int, int]) -> None:
        self.full = full
        self.offset = (0, 0)  # (row, column) of the window's first pixel
        self.window = full  # (height, width) of the window


class GlobalGridUpsample(nn.Module):
    """x2 bilinear ``align_corners=True`` upsampling on a :class:`TileGrid`."""

    def __init__(self, grid: TileGrid) -> None:
        super().__init__()
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return global_grid_upsample(x, self.grid)


def global_grid_upsample(x: torch.Tensor, grid: TileGrid) -> torch.Tensor:
    """Upsample a window's ``(B, C, h, w)`` feature map x2 on the whole grid.

    Reproduces ``F.interpolate(scale_factor=2, mode="bilinear",
    align_corners=True)`` applied to the whole image's feature map at this
    pooling level, restricted to the window. Source taps outside the window
    are clamped to its edge; that only affects margin pixels, which tiling
    crops away.

    Raises
    ------
    RuntimeError
        If ``x`` is not a pooling level of the window, or the window's
        offset is not a multiple of that level's stride.
    """
    batch, channels, height, width = x.shape
    rows = _Axis(2, height, grid.window[0], grid.offset[0], grid.full[0], x)
    cols = _Axis(3, width, grid.window[1], grid.offset[1], grid.full[1], x)
    out = x.new_empty((batch, channels, 2 * height, 2 * width))
    # Separable, in channel chunks: the columns pass, then the rows pass,
    # computes PyTorch's (1-ly)*((1-lx)*a + lx*b) + ly*((1-lx)*c + lx*d).
    # The two buffers hold 1.5 chunks of output: at most 3/16 of the output
    # (for C >= 8).
    step = max(1, min(CHUNK, channels // 8))
    half = x.new_empty((batch, step, height, 2 * width))
    scratch = x.new_empty(batch * step * 4 * height * width)
    for lo in range(0, channels, step):
        part = x[:, lo : lo + step]
        cols.interpolate(part, half[:, : part.shape[1]], scratch)
        rows.interpolate(half[:, : part.shape[1]], out[:, lo : lo + step], scratch)
    return out


class _Axis:
    """One axis (``dim``) of a global-grid upsample of an ``n``-long map.

    Output ``o`` is ``src[i0[o]] * (1 - lam[o]) + src[i1[o]] * lam[o]``, with
    the taps of :func:`_taps`. Inside the map those are ``(q - 1, q)`` for
    ``o = 2q`` and ``(q, q + 1)`` for ``o = 2q + 1``, so shifted slices
    compute all outputs but the two ends at once. The ends, and the rare
    outputs whose float32 taps leave that pattern (where rounding crosses an
    integer), are then computed from their own taps.
    """

    def __init__(
        self, dim: int, n: int, window: int, offset: int, full: int, like: torch.Tensor
    ) -> None:
        i0, i1, lam = _taps(n, window, offset, full, like)
        out = torch.arange(2 * n, device=like.device)
        # The pattern's taps. At the two ends they fall outside the map (-1
        # and n), so the ends always get their own taps.
        pattern0 = (out - 1).div_(2, rounding_mode="floor")
        pattern1 = (out + 1).div_(2, rounding_mode="floor")
        own = ((i0 != pattern0) | (i1 != pattern1)).nonzero().flatten()
        tail = (1,) * (3 - dim)  # broadcast the weights along the later dims
        self.dim = dim
        self.w1 = lam[1:-1].view(n - 1, 2, *tail)  # outputs 1 .. 2n - 2
        self.w0 = 1 - self.w1
        self.own = own
        self.own_taps = i0[own], i1[own]
        self.own_w1 = lam[own].view(-1, *tail)
        self.own_w0 = 1 - self.own_w1

    def interpolate(
        self, src: torch.Tensor, out: torch.Tensor, scratch: torch.Tensor
    ) -> None:
        """Write ``src`` interpolated along ``dim`` (``n`` -> ``2n``) to ``out``.

        ``scratch`` is a flat buffer of at least ``out.numel()`` elements.
        """
        dim, n = self.dim, src.shape[self.dim]
        head, tail = src.narrow(dim, 0, n - 1), src.narrow(dim, 1, n - 1)
        # Outputs 2q + 1 and 2q + 2 both read taps (q, q + 1).
        pairs = out.narrow(dim, 1, 2 * n - 2).unflatten(dim, (n - 1, 2))
        if dim == out.ndim - 1:
            # Along the last axis, one strided pass per output parity is
            # faster than broadcasting over the pair axis.
            for p in (0, 1):
                w0, w1 = self.w0.select(1, p), self.w1.select(1, p)
                _lerp(pairs.select(dim + 1, p), head, tail, w0, w1, scratch)
        else:
            head, tail = head.unsqueeze(dim + 1), tail.unsqueeze(dim + 1)
            _lerp(pairs, head, tail, self.w0, self.w1, scratch)
        i0, i1 = self.own_taps
        own = src.index_select(dim, i0).mul_(self.own_w0)
        own.add_(src.index_select(dim, i1).mul_(self.own_w1))
        out.index_copy_(dim, self.own, own)


def _lerp(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    scratch: torch.Tensor,
) -> None:
    """``out = a * w0 + b * w1``, with ``b * w1`` staged in ``scratch``."""
    torch.mul(a, w0, out=out)
    staged = scratch[: out.numel()].view(out.shape)
    torch.mul(b, w1, out=staged)
    out.add_(staged)


def _taps(
    n: int, window: int, offset: int, full: int, like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local source indices ``(i0, i1)`` and weight ``lambda`` along one axis.

    The float32 arithmetic of PyTorch's ``align_corners=True`` kernel, on
    the whole image's grid at the level of an ``n``-long window map.
    """
    k = _level(window, n)
    if offset % (1 << k):
        raise RuntimeError(
            f"tile offset {offset} is not a multiple of 2**{k}, the stride of "
            f"pooling level {k}"
        )
    n_full, start = full >> k, offset >> k
    scale = np.float32(n_full - 1) / np.float32(2 * n_full - 1)
    index = torch.arange(2 * start, 2 * (start + n), device=like.device)
    src = index.to(torch.float32) * torch.tensor(scale, device=like.device)
    i0 = src.floor().to(torch.int64).clamp_(max=n_full - 1)
    lam = (src - i0.to(torch.float32)).clamp_(0, 1)
    i1 = (i0 + 1).clamp_(max=n_full - 1)
    return (
        (i0 - start).clamp_(0, n - 1),
        (i1 - start).clamp_(0, n - 1),
        lam.to(like.dtype),
    )


def _level(window: int, n: int) -> int:
    """The pooling level ``k`` with ``window >> k == n`` (MaxPool2d floors)."""
    k = 0
    while window >> k > n:
        k += 1
    if window >> k != n:
        raise RuntimeError(
            f"a {n}-px feature map is not a pooling level of a {window}-px tile"
        )
    return k


def _is_x2_bilinear(module: nn.Upsample) -> bool:
    factor = module.scale_factor
    factors = tuple(factor) if isinstance(factor, (tuple, list)) else (factor,) * 2
    return module.mode == "bilinear" and module.size is None and factors == (2, 2)


def tiling_view(model: nn.Module, grid: TileGrid) -> nn.Module | None:
    """``model`` prepared for tile-exact inference on ``grid``.

    Parameters
    ----------
    model
        The caller's model. It is never modified.
    grid
        The geometry the returned view's upsamples read.

    Returns
    -------
    torch.nn.Module or None
        - A copy of ``model`` sharing every parameter and buffer, with each
          x2 bilinear ``align_corners=True`` :class:`~torch.nn.Upsample`
          replaced by a :class:`GlobalGridUpsample` reading ``grid``;
        - ``model`` itself when it has no ``align_corners=True`` upsampling
          (plain core cropping already matches an untiled run);
        - ``None`` when it cannot be made tile-exact: it contains TorchScript
          or FX graph modules (which cannot be inspected) or other
          ``align_corners=True`` upsampling, or ``copy.deepcopy`` fails on it.
    """
    modules = list(model.modules())
    opaque = (torch.jit.ScriptModule, torch.fx.GraphModule)
    if any(isinstance(m, opaque) for m in modules):
        return None
    corner = [m for m in modules if isinstance(m, nn.Upsample) and m.align_corners]
    if not corner:
        return model
    if not all(_is_x2_bilinear(m) for m in corner):
        return None
    # Pre-filling the memo shares every tensor: the copy costs no weights.
    memo = {id(t): t for t in (*model.parameters(), *model.buffers())}
    try:
        view = copy.deepcopy(model, memo)
    except Exception:  # noqa: BLE001 - e.g. an attribute deepcopy cannot copy
        return None
    swaps = [
        (parent, name)
        for parent in view.modules()
        for name, child in parent.named_children()
        if isinstance(child, nn.Upsample) and child.align_corners
    ]
    for parent, name in swaps:
        setattr(parent, name, GlobalGridUpsample(grid))
    return view
