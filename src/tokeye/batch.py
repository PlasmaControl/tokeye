"""Headless batch inference: run TokEye over a list of files with no GUI.

Never imports gradio or ``matplotlib.pyplot`` (previews are drawn on a bare
``Figure``), so this module is safe on HPC login/compute nodes and in CI.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

from . import hub
from ._plotting import overlay_rgba, save_preview
from ._version import __version__
from .config import DEFAULT_CHANNELS, SpectrogramConfig, resolve_channels
from .inference import _device_of, _plan_tiles, check_tile, infer
from .io import DIRECTORY_SUFFIXES, load_signal
from .preprocess import Spectrogram, prepare
from .result import Segmentation

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch.nn as nn

logger = logging.getLogger(__name__)

FORMATS = ("npy", "npz")


def collect_inputs(inputs: list[str]) -> list[Path]:
    """Expand a list of files/directories/glob patterns into concrete paths.

    Each item is resolved as: an existing file (kept as-is), an existing
    directory (its files with a suffix in
    :data:`tokeye.io.DIRECTORY_SUFFIXES`, sorted), or otherwise a glob
    pattern (matches, sorted). Duplicates are dropped, preserving
    first-seen order.
    """
    collected: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file():
            found = [path]
        elif path.is_dir():
            found = sorted(
                p
                for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in DIRECTORY_SUFFIXES
            )
        else:
            # glob.glob (not Path.glob) so absolute patterns keep working:
            # Path(".").glob() rejects non-relative patterns outright.
            found = sorted(Path(match) for match in glob.glob(item))  # noqa: PTH207
        collected.extend(found)

    seen: set[Path] = set()
    result: list[Path] = []
    for path in collected:
        if path not in seen:
            seen.add(path)
            result.append(path)

    if not result:
        raise ValueError(f"No input files found for: {inputs}")

    return result


def load_spectrogram(
    path: str | Path,
    config: SpectrogramConfig | Mapping[str, Any] | None = None,
    *,
    fs: float | None = None,
) -> Spectrogram:
    """Read ``path`` with :func:`tokeye.io.load_signal` and :func:`prepare` it.

    An explicit ``fs`` wins over one found in the file.
    """
    data, file_fs = load_signal(path)
    return prepare(data, config, fs=fs if fs is not None else file_fs)


def load_input(path: Path, stft_kwargs: dict, log: bool = False) -> np.ndarray:
    """Pre-1.0 helper: the float32 spectrogram for ``path``.

    ``stft_kwargs`` are :class:`SpectrogramConfig` fields.
    """
    return load_spectrogram(path, {**stft_kwargs, "log": log}).values


def save_overlay_png(
    spectrogram: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    threshold: float = 0.5,
    dpi: int = 150,
) -> None:
    """Save a grayscale spectrogram with a semi-transparent mask overlay.

    Coherent activity (``mask[0]``) is tinted green, transient activity
    (``mask[1]``) red, both thresholded at ``threshold``.
    """
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.add_subplot()
    ax.imshow(spectrogram, cmap="gray", origin="lower", aspect="auto")
    ax.imshow(overlay_rgba(mask, threshold), origin="lower", aspect="auto")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)


def _coerce_config(config: Any) -> SpectrogramConfig:
    if isinstance(config, dict):
        warnings.warn(
            "passing a dict of STFT kwargs is deprecated; pass a "
            "tokeye.SpectrogramConfig (removed in 2.0)",
            DeprecationWarning,
            stacklevel=3,
        )
    return SpectrogramConfig.coerce(config)


def process_file(
    path: Path,
    model: nn.Module,
    config: SpectrogramConfig | dict | None,
    out_dir: Path,
    save_png: bool = True,
    threshold: float = 0.5,
    log: bool | None = None,
    *,
    fs: float | None = None,
    fmt: str = "npy",
    model_name: str = hub.DEFAULT_MODEL,
    channels: tuple[str, ...] | None = None,
    tile: int | str | None = "auto",
) -> Path:
    """Segment one input file and write its outputs to ``out_dir``.

    Writes ``<stem>_mask.npy`` (float32 ``(C, H, W)``) or, with
    ``fmt="npz"``, ``<stem>_tokeye.npz`` (a :class:`Segmentation` bundle);
    ``<stem>_preview.png`` unless ``save_png`` is off; and
    ``<stem>_params.json`` recording how the output was made. Returns the
    mask (or bundle) path.

    ``tile`` is passed to :func:`tokeye.inference.infer`. ``params.json`` is
    removed before the first write and written last, so it only ever sits
    beside a complete set of outputs.
    """
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")
    check_tile(tile)
    path, out_dir = Path(path), Path(out_dir)
    cfg = _coerce_config(config)
    if log is not None:
        cfg = cfg.replace(log=log)
    spec = load_spectrogram(path, cfg, fs=fs)
    mask = infer(model, spec.values, tile=tile)
    plan = _plan_tiles(spec.values.shape, tile)  # the plan infer just used
    names = resolve_channels(channels or DEFAULT_CHANNELS, mask.shape[0])
    seg = Segmentation(mask, spec, names, model_name)

    params_path = out_dir / f"{path.stem}_params.json"
    params_path.unlink(missing_ok=True)
    if fmt == "npz":
        output = seg.save(out_dir / f"{path.stem}_tokeye.npz")
    else:
        output = out_dir / f"{path.stem}_mask.npy"
        np.save(output, mask)

    if save_png:
        save_preview(seg, out_dir / f"{path.stem}_preview.png", threshold=threshold)

    params = {
        "tokeye_version": __version__,
        "model": model_name,
        "device": str(_device_of(model)),
        "input": str(path),
        "kind": spec.kind,
        "fs": spec.fs,
        "config": cfg.to_dict(),
        "mask_shape": list(mask.shape),
        "channels": list(names),
        "threshold": float(threshold),
        "tile": tile if tile is None or isinstance(tile, str) else int(tile),
        "tile_shape": None if plan is None else list(plan),
        "output": output.name,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    _write_atomic(params_path, json.dumps(params, indent=2) + "\n")
    return output


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to a temporary file beside ``path``, then rename it."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_batch(
    inputs: list[str],
    model: str | Path = hub.DEFAULT_MODEL,
    out_dir: Path = Path("tokeye_output"),
    stft_kwargs: dict | None = None,
    save_png: bool = True,
    threshold: float = 0.5,
    device: str = "auto",
    log: bool | None = None,
    *,
    config: SpectrogramConfig | None = None,
    fs: float | None = None,
    fmt: str = "npy",
    tile: int | str | None = "auto",
) -> int:
    """Segment every input, writing outputs to ``out_dir``.

    Parameters
    ----------
    inputs
        Files, directories or glob patterns (see :func:`collect_inputs`).
    model, device
        Passed to :func:`tokeye.hub.load_model`; must be a segmentation
        model.
    config
        Preprocessing settings (defaults when omitted).
    fs
        Sampling rate for every input (else read per file, if recorded).
    fmt
        ``"npy"`` (mask only) or ``"npz"`` (full bundle).
    tile
        ``"auto"`` (default), ``None`` or an int >= 512; passed to
        :func:`tokeye.inference.infer`.
    stft_kwargs, log
        Deprecated spellings of ``config``.

    Returns
    -------
    int
        The number of inputs that failed (each is logged).

    Raises
    ------
    ValueError
        Bad settings, an instance model, or no inputs found.
    TypeError
        ``tile`` is not ``"auto"``, ``None`` or an int.
    """
    if stft_kwargs is not None or log is not None:
        warnings.warn(
            "run_batch(stft_kwargs=..., log=...) is deprecated; pass "
            "config=tokeye.SpectrogramConfig(...) (removed in 2.0)",
            DeprecationWarning,
            stacklevel=2,
        )
    if stft_kwargs is not None and config is not None:
        raise ValueError("pass either config or stft_kwargs, not both")
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")
    check_tile(tile)
    cfg = SpectrogramConfig.coerce(config if stft_kwargs is None else stft_kwargs)
    if log is not None:
        cfg = cfg.replace(log=log)

    hub.require_task(model, "segmentation")
    paths = collect_inputs(inputs)
    loaded_model = hub.load_model(model, device)
    channels = hub.channels_for(model)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for path in tqdm(paths, desc="tokeye run"):
        try:
            process_file(
                path,
                loaded_model,
                cfg,
                out_dir,
                save_png=save_png,
                threshold=threshold,
                fs=fs,
                fmt=fmt,
                model_name=str(model),
                channels=channels,
                tile=tile,
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            logger.error("Failed to process %s: %s", path, exc)
            failures += 1

    return failures
