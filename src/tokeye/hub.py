"""Model registry, Hugging Face downloads and local checkpoint loading.

Gradio-free. Model architectures are imported lazily by their builders, so
torchvision is only needed for ``ae_tf_maskrcnn`` (the ``ae`` extra).
"""

from __future__ import annotations

import logging
import os
import pickle
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from .config import DEFAULT_CHANNELS, DEFAULT_MODEL

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = os.environ.get("TOKEYE_HF_REPO", "nc1/big_tf_unet")

_PATH_SUFFIXES = {".pt", ".pt2"}
TASKS = ("segmentation", "instance")
_ARTICLES = {"segmentation": "a", "instance": "an"}
DEVICE_FORMS = "cpu, cuda, cuda:N, mps or auto"
_NO_DEVICE_HINT = "; run `tokeye info` to see what this machine has"
_CUDA_DEVICE = re.compile(r"cuda(?::([0-9]+))?")
_MAX_MISMATCH_TEXT = 200


class DownloadError(OSError):
    """Hugging Face could not be reached, and the weights are not cached.

    It means what :class:`huggingface_hub.errors.LocalEntryNotFoundError`
    means, for the network failures huggingface_hub does not map to it.
    """


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
        f"{spec.name!r} is {_ARTICLES[spec.task]} {spec.task} model, but this needs "
        f"{_ARTICLES.get(task, 'a')} {task} model; {hint}"
    )


def channels_for(source: str | Path) -> tuple[str, ...]:
    """Mask channel names for a model (the default names for local files)."""
    spec = MODEL_REGISTRY.get(str(source))
    return spec.channels if spec is not None else DEFAULT_CHANNELS


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(device: str | torch.device = "auto") -> str:
    """Check ``device`` and return it; ``"auto"`` picks one.

    Parameters
    ----------
    device
        ``"auto"`` (CUDA, then MPS, then CPU), ``"cpu"``, ``"cuda"``,
        ``"cuda:N"`` or ``"mps"``, or a :class:`torch.device` of these.

    Returns
    -------
    str
        The device ``"auto"`` picks, else ``str(device)`` unchanged.

    Raises
    ------
    ValueError
        Any other value, or a device this machine does not have.
    """
    text = str(device)
    if text == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"
    if text == "cpu":
        return text
    cuda = _CUDA_DEVICE.fullmatch(text)
    if cuda is not None:
        if not torch.cuda.is_available():
            raise ValueError(f"device {text!r}: CUDA is not available{_NO_DEVICE_HINT}")
        count = torch.cuda.device_count()
        if cuda.group(1) is not None and int(cuda.group(1)) >= count:
            raise ValueError(
                f"device {text!r}: this machine has {count} CUDA device(s), "
                f"numbered from 0{_NO_DEVICE_HINT}"
            )
        return text
    if text == "mps":
        if not _mps_available():
            raise ValueError(f"device {text!r}: MPS is not available{_NO_DEVICE_HINT}")
        return text
    raise ValueError(f"unknown device {text!r}; use {DEVICE_FORMS}")


def _network_errors() -> tuple[type[Exception], ...]:
    """What ``hf_hub_download`` raises when the hub cannot be reached.

    ``RuntimeError`` too: on huggingface_hub 1.x a refused connection to an
    uncached file ends in "Cannot send a request, as the client has been
    closed" (its retry reuses the session it just closed).
    """
    errors: list[type[Exception]] = [RuntimeError]
    try:  # huggingface_hub >= 1 talks through httpx
        import httpx

        errors.append(httpx.HTTPError)
    except ImportError:
        pass
    try:  # huggingface_hub < 1 talks through requests
        import requests

        errors.append(requests.RequestException)
    except ImportError:
        pass
    return tuple(errors)


def download_model(name: str = DEFAULT_MODEL, repo_id: str | None = None) -> Path:
    """Download (or find in the cache) a registry model's weights.

    Raises
    ------
    ValueError
        ``name`` is not a registry name.
    huggingface_hub.errors.LocalEntryNotFoundError, DownloadError
        The hub cannot be reached and the weights are not cached.
    huggingface_hub.errors.HfHubHTTPError
        The hub answered with an error (a missing repo, for example).
    """
    spec = _spec(name)
    repo = repo_id or spec.repo_id or DEFAULT_REPO_ID
    try:
        return Path(hf_hub_download(repo, spec.filename))
    except (LocalEntryNotFoundError, HfHubHTTPError):
        # First: on huggingface_hub 1.x HfHubHTTPError is an httpx.HTTPError,
        # and on 0.x both are requests.RequestExceptions.
        raise
    except _network_errors() as exc:
        raise DownloadError(
            f"could not download {spec.filename} from {repo}: {exc}"
        ) from exc


def _mismatch(model: nn.Module, state_dict: Mapping) -> str:
    """``"3 missing, 2 unexpected keys"`` (plus wrong shapes), from the keys."""
    expected = model.state_dict()
    shared = expected.keys() & state_dict.keys()
    wrong_shape = sum(
        tuple(getattr(state_dict[key], "shape", ())) != tuple(expected[key].shape)
        for key in shared
    )
    text = (
        f"{len(expected.keys() - shared)} missing, "
        f"{len(state_dict.keys() - shared)} unexpected keys"
    )
    return f"{text}, {wrong_shape} of the wrong shape" if wrong_shape else text


def _build_from_state_dict(state_dict: Mapping, device: str) -> nn.Module:
    mismatches: list[str] = []
    for spec in MODEL_REGISTRY.values():
        try:
            model = spec.builder()
        except ImportError as exc:
            text = str(exc)
            mismatches.append(
                text if text.startswith(spec.name) else f"{spec.name}: {text}"
            )
            continue
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            logger.debug("state dict does not fit %s: %s", spec.name, exc)
            mismatches.append(f"{spec.name}: {_mismatch(model, state_dict)}")
            continue
        return model.to(device).eval()

    message = (
        "State dict does not match any known TokEye architecture "
        f"({', '.join(sorted(MODEL_REGISTRY))}). {'; '.join(mismatches)}"
    )
    if len(message) > _MAX_MISMATCH_TEXT:
        message = message[: _MAX_MISMATCH_TEXT - 1] + "\u2026"
    raise ValueError(message)


def _load_from_registry(name: str, device: str) -> nn.Module:
    spec = MODEL_REGISTRY[name]
    path = download_model(name)
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        logger.debug("cannot read the cached weights %s: %s", path, exc)
        # The resolved blob: deleting only the snapshot symlink would let
        # hf_hub_download re-link the same corrupt blob without downloading.
        raise ValueError(
            f"cached weights for {name!r} at {path.resolve()} are not readable "
            f"({type(exc).__name__}); delete that file and run "
            f"`tokeye download {name}`"
        ) from exc
    model = spec.builder()
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _unreadable(name: str, exc: Exception) -> ValueError:
    return ValueError(
        f"{name}: not a readable checkpoint ({type(exc).__name__}: {exc})"
    )


def _load_pt2(path: Path, name: str, device: str) -> nn.Module:
    try:
        program = torch.export.load(str(path))
    except OSError:
        raise
    except Exception as exc:
        raise _unreadable(name, exc) from exc
    return program.module().to(device)


def _load_legacy_module(path: Path, name: str, device: str) -> nn.Module:
    """Unpickle a checkpoint saved as a full module (not a state dict).

    Only ever done for local files: the registry path always loads with
    ``weights_only=True``.
    """
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except OSError:
        raise
    except Exception as exc:
        raise _unreadable(name, exc) from exc
    # Logged only once the load worked, so an unreadable file is one error.
    logger.warning(
        "%s could not be loaded safely (weights_only=True), so the full file "
        "was unpickled. Only load local files you trust this way.",
        name,
    )
    return model.to(device).eval()


def _load_pt(path: Path, name: str, device: str) -> nn.Module:
    try:
        loaded = torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        return _load_legacy_module(path, name, device)
    except OSError:
        raise
    except Exception as exc:
        raise _unreadable(name, exc) from exc

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
        ``"auto"`` (CUDA, then MPS, then CPU), ``"cpu"``, ``"cuda"``,
        ``"cuda:N"`` or ``"mps"`` (see :func:`resolve_device`).

    Raises
    ------
    ValueError
        An unknown or unavailable device, an unknown registry name, an
        unreadable checkpoint, or a state dict that fits no registered
        architecture.
    FileNotFoundError
        ``source`` looks like a ``.pt``/``.pt2`` path that does not exist.
    huggingface_hub.errors.LocalEntryNotFoundError, DownloadError
        The hub cannot be reached and the weights are not cached.
    huggingface_hub.errors.HfHubHTTPError
        The hub answered with an error.
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
        return _load_pt2(path, name, resolved_device)
    if path.suffix == ".pt":
        return _load_pt(path, name, resolved_device)

    raise ValueError(f"Unsupported model file suffix: {path.suffix!r}")
