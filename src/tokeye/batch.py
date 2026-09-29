"""Headless batch inference: run TokEye over a list of files with no GUI.

Never imports gradio or ``matplotlib.pyplot`` (previews are drawn on a bare
``Figure``), so this module is safe on HPC login/compute nodes and in CI.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
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
from .io import (
    CONTAINER_SUFFIXES,
    DIRECTORY_SUFFIXES,
    SIGNAL_SUFFIXES,
    _check_key,
    load_signal,
)
from .preprocess import Spectrogram, _check_fs, prepare
from .result import Segmentation

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import torch.nn as nn

logger = logging.getLogger(__name__)

FORMATS = ("npy", "npz")


def collect_inputs(inputs: list[str]) -> list[Path]:
    """Expand a list of files/directories/glob patterns into concrete paths.

    Each item is resolved as:

    - an existing file: kept as-is, whatever its suffix (an unsupported one
      then fails in :func:`tokeye.io.load_signal`);
    - an existing directory: its files with a suffix in
      :data:`tokeye.io.DIRECTORY_SUFFIXES`, sorted;
    - otherwise a glob pattern: its matches that are files with a suffix in
      :data:`tokeye.io.SIGNAL_SUFFIXES` (``.csv``/``.txt`` included), sorted.

    A file reached twice under the same name (through its directory and by
    name, or by a relative and an absolute path) is kept once, in its
    first-seen position and spelling. A symlink with another name stays a
    separate input.

    Raises
    ------
    ValueError
        Nothing was collected. The message counts the glob matches that
        were skipped for their suffix, if any.
    """
    collected: list[Path] = []
    skipped: dict[Path, str] = {}  # resolved path -> suffix, for the error
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
            matches = sorted(Path(match) for match in glob.glob(item))  # noqa: PTH207
            files = [p for p in matches if p.is_file()]
            found = [p for p in files if p.suffix.lower() in SIGNAL_SUFFIXES]
            for p in files:
                if p.suffix.lower() not in SIGNAL_SUFFIXES:
                    skipped[p.resolve()] = p.suffix.lower() or "(none)"
        collected.extend(found)

    seen: set[tuple[Path, str]] = set()
    result: list[Path] = []
    for path in collected:
        key = (path.resolve(), path.name.casefold())
        if key not in seen:
            seen.add(key)
            result.append(path)

    if not result:
        detail = ""
        if skipped:
            count = len(skipped)
            detail = (
                f" ({count} {'match' if count == 1 else 'matches'} skipped with "
                f"unsupported suffixes: {', '.join(sorted(set(skipped.values())))}; "
                f"supported: {', '.join(SIGNAL_SUFFIXES)})"
            )
        raise ValueError(f"No input files found for: {inputs}{detail}")

    return result


def _join_limited(parts: Sequence[str], *, sep: str, limit: int = 5) -> str:
    """``parts`` joined by ``sep``, the first ``limit`` only, then a count."""
    text = sep.join(parts[:limit])
    remaining = len(parts) - limit
    if remaining > 0:
        text += f"{sep}… and {remaining} more"
    return text


def check_unique_stems(paths: Sequence[Path]) -> None:
    """Reject inputs whose outputs would overwrite each other.

    Every per-input output is named after the file stem, so ``shot.npy`` and
    ``shot.wav``, or ``a/shot.npy`` and ``b/shot.npy``, would write the same
    files. Stems are compared ignoring case (``Shot`` and ``shot`` are one
    file on the default macOS and Windows file systems).

    Parameters
    ----------
    paths
        The inputs, e.g. from :func:`collect_inputs`.

    Raises
    ------
    ValueError
        Two or more inputs share a stem. The one-line message names each
        group of inputs (the first 5, then a count).
    """
    groups: dict[str, list[Path]] = {}
    for path in map(Path, paths):
        groups.setdefault(path.stem.casefold(), []).append(path)
    clashes = [
        f"{', '.join(map(str, members))} -> {members[0].stem!r}"
        for members in groups.values()
        if len(members) > 1
    ]
    if clashes:
        raise ValueError(
            "inputs would overwrite each other's outputs (outputs are named "
            f"after the file stem, ignoring case): {_join_limited(clashes, sep='; ')}"
            ". Process files that share a stem in separate runs with different "
            "output directories, or rename them."
        )


def check_key_inputs(paths: Sequence[str | Path], key: str | None) -> None:
    """Reject a ``key`` that some inputs cannot use.

    ``key=`` selects an array inside a container, so every input must be a
    ``.npz``, ``.mat``, ``.h5`` or ``.hdf5`` file. Nothing is read.

    Parameters
    ----------
    paths
        The inputs, e.g. from :func:`collect_inputs`.
    key
        The array to read from each input; ``None`` checks nothing.

    Raises
    ------
    TypeError
        ``key`` is not a string.
    ValueError
        ``key`` is empty, or some inputs are not containers. The one-line
        message names them as given, in input order (the first 5, then a
        count).
    """
    if key is None:
        return
    _check_key(key)
    bad = [str(p) for p in paths if Path(p).suffix.lower() not in CONTAINER_SUFFIXES]
    if bad:
        raise ValueError(
            "key= (--key) applies only to .npz, .mat, .h5 and .hdf5 inputs, "
            f"not: {_join_limited(bad, sep=', ')}"
        )


def load_spectrogram(
    path: str | Path,
    config: SpectrogramConfig | Mapping[str, Any] | None = None,
    *,
    fs: float | None = None,
    key: str | None = None,
) -> Spectrogram:
    """Read ``path`` with :func:`tokeye.io.load_signal` and :func:`prepare` it.

    An explicit ``fs`` wins over one found in the file. ``key`` selects the
    array in a container (see :func:`tokeye.io.load_signal`).
    """
    data, file_fs = load_signal(path, key=key)
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


def _check_fmt(fmt: str) -> None:
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")


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
    key: str | None = None,
) -> Path:
    """Segment one input file and write its outputs to ``out_dir``.

    Writes ``<stem>_mask.npy`` (float32 ``(C, H, W)``) or, with
    ``fmt="npz"``, ``<stem>_tokeye.npz`` (a :class:`Segmentation` bundle);
    ``<stem>_preview.png`` unless ``save_png`` is off; and
    ``<stem>_params.json`` recording how the output was made. Returns the
    mask (or bundle) path.

    ``tile`` is passed to :func:`tokeye.inference.infer`, and ``key`` to
    :func:`tokeye.io.load_signal`. ``params.json`` is removed before the
    first write and written last, so it only ever sits beside a complete set
    of outputs.
    """
    _check_fmt(fmt)
    check_tile(tile)
    path, out_dir = Path(path), Path(out_dir)
    cfg = _coerce_config(config)
    if log is not None:
        cfg = cfg.replace(log=log)
    spec = load_spectrogram(path, cfg, fs=fs, key=key)
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
        "key": key,
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


def process_files(
    paths: Sequence[Path],
    model: nn.Module,
    config: SpectrogramConfig | Mapping[str, Any] | None,
    out_dir: Path,
    *,
    save_png: bool = True,
    threshold: float = 0.5,
    fs: float | None = None,
    fmt: str = "npy",
    tile: int | str | None = "auto",
    model_name: str = hub.DEFAULT_MODEL,
    channels: tuple[str, ...] | None = None,
    key: str | None = None,
    on_error: Callable[[Path, Exception], None] | None = None,
) -> int:
    """Segment each path with a loaded ``model``: the loop of :func:`run_batch`.

    Each path goes through :func:`process_file` into ``out_dir``, which must
    exist, under a progress bar. One input that fails does not stop the
    others. Inputs that share a stem are rejected before anything is
    written (:func:`check_unique_stems`).

    Parameters
    ----------
    paths
        Input files, e.g. from :func:`collect_inputs`.
    model
        A loaded segmentation model (:func:`tokeye.hub.load_model`).
    config
        Preprocessing settings (defaults when ``None``). A ``dict`` is
        deprecated and warns once per call.
    out_dir
        Existing directory for the outputs.
    save_png, threshold, fs, fmt, tile, model_name, channels, key
        As for :func:`process_file`.
    on_error
        Called as ``on_error(path, exc)`` for each input that fails. Without
        it, each failure is logged at ERROR (its traceback at DEBUG) on the
        ``tokeye.batch`` logger.

    Returns
    -------
    int
        The number of inputs that failed.

    Raises
    ------
    ValueError
        ``fmt`` is not ``"npy"`` or ``"npz"``, ``tile`` is below 512, or two
        inputs share a stem (their outputs would overwrite each other).
    TypeError
        ``tile`` is not ``"auto"``, ``None`` or an int.
    """
    _check_fmt(fmt)
    check_tile(tile)
    check_unique_stems(paths)
    cfg = _coerce_config(config)
    failures = 0
    for path in tqdm(paths, desc="tokeye run"):
        try:
            process_file(
                path,
                model,
                cfg,
                out_dir,
                save_png=save_png,
                threshold=threshold,
                fs=fs,
                fmt=fmt,
                model_name=model_name,
                channels=channels,
                tile=tile,
                key=key,
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            failures += 1
            # Clear the progress bar first, or the line is glued onto it.
            with tqdm.external_write_mode(file=sys.stderr):
                if on_error is not None:
                    on_error(path, exc)
                else:
                    logger.error(
                        "Failed to process %s: %s: %s", path, type(exc).__name__, exc
                    )
                    logger.debug("traceback for %s", path, exc_info=exc)
    return failures


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
    key: str | None = None,
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
    key
        The array to read from each input, which must then be a ``.npz``,
        ``.mat``, ``.h5`` or ``.hdf5`` file (see
        :func:`tokeye.io.load_signal`); by default each file's own signal
        array.
    stft_kwargs, log
        Deprecated spellings of ``config``.

    Returns
    -------
    int
        The number of inputs that failed (each is logged, see
        :func:`process_files`).

    Raises
    ------
    ValueError
        Bad settings, an ``fs`` that is not finite and positive, an empty
        ``key``, an instance model, no inputs found, inputs that share a
        stem, or a ``key`` with inputs that are not containers (both
        checked before the model loads; see :func:`check_unique_stems` and
        :func:`check_key_inputs`), an unknown or unavailable ``device``, or
        a model that cannot be loaded (see :func:`tokeye.hub.load_model`,
        which also lists its other errors).
    TypeError
        ``fs`` or ``key`` is of the wrong type, or ``tile`` is not
        ``"auto"``, ``None`` or an int.
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
    _check_fmt(fmt)
    _check_fs(fs)
    if key is not None:
        _check_key(key)
    check_tile(tile)
    cfg = SpectrogramConfig.coerce(config if stft_kwargs is None else stft_kwargs)
    if log is not None:
        cfg = cfg.replace(log=log)

    hub.require_task(model, "segmentation")
    paths = collect_inputs(inputs)
    check_unique_stems(paths)
    check_key_inputs(paths, key)
    loaded_model = hub.load_model(model, device)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    return process_files(
        paths,
        loaded_model,
        cfg,
        out_dir,
        save_png=save_png,
        threshold=threshold,
        fs=fs,
        fmt=fmt,
        tile=tile,
        model_name=str(model),
        channels=hub.channels_for(model),
        key=key,
    )
