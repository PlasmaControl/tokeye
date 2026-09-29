"""TokEye: classification and localization of fluctuating signals.

``from tokeye import TokEye`` is the one-import Python API. Public classes
are resolved lazily (PEP 562) so ``import tokeye`` -- and therefore the
CLI -- stays free of torch, scipy and gradio imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from tokeye._version import __version__

if TYPE_CHECKING:
    from tokeye.api import TokEye
    from tokeye.config import SpectrogramConfig
    from tokeye.result import Segmentation

__all__ = ["Segmentation", "SpectrogramConfig", "TokEye", "__version__"]

_LAZY = {
    "TokEye": "tokeye.api",
    "Segmentation": "tokeye.result",
    "SpectrogramConfig": "tokeye.config",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
