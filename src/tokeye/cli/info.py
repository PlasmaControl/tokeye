"""``tokeye info`` — environment report for troubleshooting."""

from __future__ import annotations

import importlib.util
import platform
import sys
from typing import TYPE_CHECKING

from tokeye._version import __version__
from tokeye.cli import _common, _options

if TYPE_CHECKING:
    import argparse

# extra name -> modules it provides
EXTRAS = {
    "app": ("gradio", "soundfile"),
    "ae": ("torchvision",),
    "hdf5": ("h5py",),
}

# The row keys, in print order; they fix the width of the key column.
ROW_KEYS = (
    "tokeye",
    "python",
    "platform",
    "torch",
    "cuda",
    "mps",
    "device",
    "extras",
    "hf cache",
)
KEY_WIDTH = max(len(key) for key in ROW_KEYS)

SKIPPED_NO_TORCH = "skipped (torch failed to import)"
SKIPPED_NO_HUB = "skipped (tokeye.hub failed to import)"


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "info",
        parents=[_options.VERBOSE],
        help="Show versions, the device auto picks, extras and cached models.",
        description=(
            "Print what tokeye sees on this machine. Paste the output into bug "
            "reports. Exits 1 if a check fails (a broken torch, say); a missing "
            "extra or an uncached model is information, not a failure."
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
            parts.append(
                f'{extra}: missing {", ".join(missing)} (pip install "tokeye[{extra}]")'
            )
        else:
            parts.append(f"{extra}: installed")
    return "; ".join(parts)


def _cuda_line(torch) -> str:
    if not torch.cuda.is_available():
        return "not available"
    count = torch.cuda.device_count()
    return f"{torch.cuda.get_device_name(0)} ({count} device(s), CUDA {torch.version.cuda})"


def _failed(exc: BaseException) -> str:
    return f"failed: {_common.one_line(exc)}"


class _Report:
    """Prints each row as soon as it is known; remembers whether one failed."""

    def __init__(self) -> None:
        self.ok = True

    def row(self, key: str, value: str, *, ok: bool = True) -> None:
        self.ok = self.ok and ok
        print(f"{key:<{KEY_WIDTH}}  {value}", flush=True)

    def guarded(self, key: str, compute) -> None:
        """Print ``compute()`` as the row's value, or the failure."""
        try:
            value = compute()
        except Exception as exc:
            self.row(key, _failed(exc), ok=False)
        else:
            self.row(key, value)


def _handle(args: argparse.Namespace) -> int:
    report = _Report()
    report.row("tokeye", __version__)
    report.row("python", f"{platform.python_version()} ({sys.executable})")
    report.row("platform", platform.platform())

    torch = None
    try:
        import torch

        report.row("torch", torch.__version__)
    except Exception as exc:
        torch = None
        report.row("torch", _failed(exc), ok=False)

    hub = None
    hub_error: Exception | None = None
    if torch is not None:
        try:
            from tokeye import hub
        except Exception as exc:
            hub_error = exc

    def device_row(key: str, compute) -> None:
        if torch is None:
            report.row(key, SKIPPED_NO_TORCH, ok=False)
        else:
            report.guarded(key, compute)

    def hub_row(key: str, compute) -> None:
        if torch is None:
            report.row(key, SKIPPED_NO_TORCH, ok=False)
        elif hub is None:
            report.row(key, _failed(hub_error), ok=False)
        else:
            report.guarded(key, compute)

    device_row("cuda", lambda: _cuda_line(torch))
    hub_row("mps", lambda: "available" if hub._mps_available() else "not available")
    hub_row(
        "device", lambda: f"{hub.resolve_device('auto')} (what --device auto picks)"
    )
    report.row("extras", _extras_line())

    def hf_cache() -> str:
        from huggingface_hub.constants import HF_HUB_CACHE

        return str(HF_HUB_CACHE)

    report.guarded("hf cache", hf_cache)

    print("models", flush=True)
    if torch is None:
        report.ok = False
        print(f"  {SKIPPED_NO_TORCH}", flush=True)
    elif hub is None:
        report.ok = False
        print(f"  {SKIPPED_NO_HUB}", flush=True)
    else:
        name_width = max(len(name) for name in hub.MODEL_REGISTRY)
        for name, spec in hub.MODEL_REGISTRY.items():
            try:
                path = hub.cached_path(name)
            except Exception as exc:
                report.ok = False
                where = _failed(exc)
            else:
                where = (
                    str(path) if path else f"not cached (run: tokeye download {name})"
                )
            size = f"~{spec.size_mb} MB"
            print(
                f"  {name:<{name_width}}  {spec.task:<12}  {size:<8}  {where}",
                flush=True,
            )
    return _common.EXIT_OK if report.ok else _common.EXIT_FAILED
