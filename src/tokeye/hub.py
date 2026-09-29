"""Model registry, Hugging Face downloads and local checkpoint loading.

Gradio-free. Model architectures are imported lazily by their builders, so
torchvision is only needed for ``ae_tf_maskrcnn`` (the ``ae`` extra).
"""

from __future__ import annotations

import logging
import os
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import hf_hub_download, try_to_load_from_cache

from .config import DEFAULT_CHANNELS, DEFAULT_MODEL

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = os.environ.get("TOKEYE_HF_REPO", "nc1/big_tf_unet")

_PATH_SUFFIXES = {".pt", ".pt2"}
TASKS = ("segmentation", "instance")


@dataclass(frozen=True)
class ModelSpec:
    """A registered model.

    ``task`` is ``"segmentation"`` (``TokEye``, ``tokeye run``) or
    ``"instance"`` (``tokeye alfvenspec``); ``channels`` names the mask
    channels of a segmentation model; ``size_mb`` is the download size.
    """

    name: str
    filename: str  # file in the HF repo
    builder: Callable[[], nn.Module]
    repo_id: str | None = None  # None -> DEFAULT_REPO_ID (TOKEYE_HF_REPO override)
    task: str = "segmentation"
    channels: tuple[str, ...] = DEFAULT_CHANNELS
    size_mb: int = 0


def _build_big_tf_unet() -> nn.Module:
    from .models.big_tf_unet.config_big_tf_unet import BigTFUNetConfig
    from .models.big_tf_unet.model_big_tf_unet import BigTFUNetModel

    return BigTFUNetModel(BigTFUNetConfig())


def _build_ae_tf_maskrcnn() -> nn.Module:
    try:
        from .models.ae_tf_maskrcnn.config_ae_tf_maskrcnn import AETFMaskConfig
        from .models.ae_tf_maskrcnn.model_ae_tf_maskrcnn import AETFMaskModel
    except ImportError as exc:
        raise ImportError(
            "ae_tf_maskrcnn needs torchvision: pip install 'tokeye[ae]'"
        ) from exc
    return AETFMaskModel(AETFMaskConfig(weights=None))


# Insertion order matters: _build_from_state_dict tries specs in order, so the
# default segmentation model must stay first — U-Net checkpoints should never
# construct the (much slower) R-CNN builder.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "big_tf_unet": ModelSpec(
        "big_tf_unet",
        "big_tf_unet_251210.pt",
        _build_big_tf_unet,
        size_mb=31,
    ),
    "ae_tf_maskrcnn": ModelSpec(
        "ae_tf_maskrcnn",
        "ae_tf_maskrcnn_251223.pt",
        _build_ae_tf_maskrcnn,
        repo_id="nc1/ae_tf_maskrcnn",
        task="instance",
        channels=(),
        size_mb=184,
    ),
}


def _spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown model {name!r}; valid names: {sorted(MODEL_REGISTRY)}"
        ) from None


def repo_for(name: str) -> str:
    """Hugging Face repo a model name resolves to (for error messages)."""
    spec = MODEL_REGISTRY.get(str(name))
    if spec is not None and spec.repo_id is not None:
        return spec.repo_id
    return DEFAULT_REPO_ID


def model_names(task: str | None = None) -> list[str]:
    """Registry names, in registry order, optionally only for one ``task``."""
    return [n for n, s in MODEL_REGISTRY.items() if task is None or s.task == task]


def cached_path(name: str) -> Path | None:
    """Local path of a registry model's weights, or ``None`` if not cached."""
    spec = _spec(name)
    cached = try_to_load_from_cache(repo_for(name), spec.filename)
    return Path(cached) if isinstance(cached, str) else None


def is_cached(name: str) -> bool:
    """Whether a registry model's weights are already downloaded."""
    return cached_path(name) is not None


def require_task(source: str | Path, task: str) -> None:
    """Raise ``ValueError`` if registry model ``source`` is not a ``task`` model.

    Local checkpoint paths are not checked here (their task is only known
    once loaded); :func:`tokeye.inference.infer` rejects detection outputs.
    """
    spec = MODEL_REGISTRY.get(str(source))
    if spec is None or spec.task == task:
        return
    hint = {
        "instance": "use `tokeye alfvenspec` (Python: tokeye.alfvenspec.detect)",
        "segmentation": "use `tokeye run` (Python: tokeye.TokEye)",
    }[spec.task]
    raise ValueError(
        f"{spec.name!r} is an {spec.task} model, but this needs a {task} model; {hint}"
    )


def channels_for(source: str | Path) -> tuple[str, ...]:
    """Mask channel names for a model (the default names for local files)."""
    spec = MODEL_REGISTRY.get(str(source))
    return spec.channels if spec is not None else DEFAULT_CHANNELS


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(device: str = "auto") -> str:
    """``"auto"`` becomes ``"cuda"``, then ``"mps"``, then ``"cpu"``."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"
    return device


def download_model(name: str = DEFAULT_MODEL, repo_id: str | None = None) -> Path:
    """Download (or find in the cache) a registry model's weights."""
    spec = _spec(name)
    resolved_repo_id = repo_id or spec.repo_id or DEFAULT_REPO_ID
    return Path(hf_hub_download(resolved_repo_id, spec.filename))


def _build_from_state_dict(state_dict: Mapping, device: str) -> nn.Module:
    mismatches: list[str] = []
    for spec in MODEL_REGISTRY.values():
        try:
            model = spec.builder()
            model.load_state_dict(state_dict, strict=True)
        except (RuntimeError, ImportError) as exc:
            mismatches.append(f"{spec.name}: {exc}")
            continue
        return model.to(device).eval()

    details = "\n".join(mismatches)
    raise ValueError(
        "State dict does not match any known TokEye architecture "
        f"({', '.join(sorted(MODEL_REGISTRY))}).\n{details}"
    )


def _load_from_registry(name: str, device: str) -> nn.Module:
    spec = MODEL_REGISTRY[name]
    path = download_model(name)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model = spec.builder()
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _load_pt2(path: Path, device: str) -> nn.Module:
    module = torch.export.load(str(path)).module()
    return module.to(device)


def _load_pt(path: Path, device: str) -> nn.Module:
    try:
        loaded = torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        # Legacy checkpoint pickled as a full module (not just a state dict).
        # Only ever done for local files: the registry/download path above
        # always loads with weights_only=True.
        logger.warning(
            "%s could not be loaded safely (weights_only=True); falling back "
            "to unpickling the full file. Only do this for local files you "
            "trust.",
            path,
        )
        model = torch.load(path, map_location=device, weights_only=False)
        return model.to(device).eval()

    if isinstance(loaded, Mapping):
        return _build_from_state_dict(loaded, device)

    return loaded.to(device).eval()


def load_model(source: str | Path = DEFAULT_MODEL, device: str = "auto") -> nn.Module:
    """Load a registry model (downloading it once) or a local checkpoint.

    Parameters
    ----------
    source
        A registry name (see :data:`MODEL_REGISTRY`) or a path to a ``.pt``
        state dict / legacy pickled module, or a ``.pt2`` exported program.
    device
        ``"auto"`` (CUDA, then MPS, then CPU) or any torch device string.
    """
    resolved_device = resolve_device(device)
    name = str(source)

    if name in MODEL_REGISTRY:
        return _load_from_registry(name, resolved_device)

    path = Path(source)
    if not path.exists():
        if path.suffix in _PATH_SUFFIXES:
            raise FileNotFoundError(f"Model file not found: {name}")
        raise ValueError(
            f"Unknown model {name!r}; valid registry names: {sorted(MODEL_REGISTRY)}"
        )

    if path.suffix == ".pt2":
        return _load_pt2(path, resolved_device)
    if path.suffix == ".pt":
        return _load_pt(path, resolved_device)

    raise ValueError(f"Unsupported model file suffix: {path.suffix!r}")
