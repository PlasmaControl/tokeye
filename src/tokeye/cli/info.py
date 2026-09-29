"""``tokeye info`` — environment report for troubleshooting."""

from __future__ import annotations

import importlib.util
import platform
import sys
from typing import TYPE_CHECKING

from tokeye._version import __version__
from tokeye.cli import _common

if TYPE_CHECKING:
    import argparse

# extra name -> modules it provides
EXTRAS = {
    "app": ("gradio", "soundfile"),
    "ae": ("torchvision",),
    "hdf5": ("h5py",),
}


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "info",
        help="Show versions, the device auto picks, extras and cached models.",
        description=(
            "Print what tokeye sees on this machine. Paste the output into bug reports."
        ),
    )
    parser.set_defaults(handler=_handle)


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _extras_line() -> str:
    parts = []
    for extra, modules in EXTRAS.items():
        missing = [m for m in modules if not _installed(m)]
        if missing:
            parts.append(f"{extra}: missing (pip install 'tokeye[{extra}]')")
        else:
            parts.append(f"{extra}: installed")
    return "; ".join(parts)


def _cuda_line(torch) -> str:
    if not torch.cuda.is_available():
        return "not available"
    count = torch.cuda.device_count()
    return f"{torch.cuda.get_device_name(0)} ({count} device(s), CUDA {torch.version.cuda})"


def _handle(args: argparse.Namespace) -> int:
    import torch
    from huggingface_hub.constants import HF_HUB_CACHE

    from tokeye import hub

    rows = [
        ("tokeye", __version__),
        ("python", f"{platform.python_version()} ({sys.executable})"),
        ("platform", platform.platform()),
        ("torch", torch.__version__),
        ("cuda", _cuda_line(torch)),
        ("mps", "available" if hub._mps_available() else "not available"),
        ("device", f"{hub.resolve_device('auto')} (what --device auto picks)"),
        ("extras", _extras_line()),
        ("hf cache", str(HF_HUB_CACHE)),
    ]
    width = max(len(key) for key, _ in rows)
    for key, value in rows:
        print(f"{key:<{width}}  {value}")

    print("models")
    name_width = max(len(name) for name in hub.MODEL_REGISTRY)
    for name, spec in hub.MODEL_REGISTRY.items():
        path = hub.cached_path(name)
        where = str(path) if path else f"not cached (run: tokeye download {name})"
        size = f"~{spec.size_mb} MB"
        print(f"  {name:<{name_width}}  {spec.task:<12}  {size:<8}  {where}")
    return _common.EXIT_OK
