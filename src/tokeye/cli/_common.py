"""Exit codes and error reporting shared by the CLI subcommands."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import torch.nn as nn

EXIT_OK = 0  # every input succeeded
EXIT_FAILED = 1  # at least one input failed
EXIT_USAGE = 2  # bad flag, unknown model, no inputs, missing extra
EXIT_INTERRUPTED = 130  # Ctrl-C

NO_INPUT_HINT = " (no data yet? create a demo signal with: tokeye example)"


def error(message: str) -> int:
    """Print ``error: message`` to stderr and return :data:`EXIT_USAGE`."""
    print(f"error: {message}", file=sys.stderr)
    return EXIT_USAGE


def print_hub_error(name: str, exc: Exception) -> None:
    """Print a friendly message for a failed Hugging Face model download."""
    from tokeye.hub import repo_for

    print(
        f"error: could not download model {name!r} from Hugging Face "
        f"repo {repo_for(name)!r}: {exc}. If the repo has moved, set "
        "TOKEYE_HF_REPO to override.",
        file=sys.stderr,
    )


def collect_inputs_or_report(inputs: Sequence[str]) -> list[Path] | None:
    """:func:`tokeye.batch.collect_inputs`, or ``None`` after printing why."""
    from tokeye import batch

    try:
        return batch.collect_inputs(list(inputs))
    except ValueError as exc:
        error(f"{exc}{NO_INPUT_HINT}")
        return None


def load_model_or_report(name: str, device: str, task: str) -> nn.Module | None:
    """Check ``name`` suits ``task`` and load it, or print why and return None."""
    from huggingface_hub.errors import HfHubHTTPError

    from tokeye import hub

    try:
        hub.require_task(name, task)
        return hub.load_model(name, device)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        error(str(exc))
    except (HfHubHTTPError, OSError) as exc:
        print_hub_error(name, exc)
    return None
