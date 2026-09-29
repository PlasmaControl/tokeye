"""Exit codes, error reporting and the start-up checks shared by the CLI.

Every usage or configuration error is one ``error:`` line on stderr with
:data:`EXIT_USAGE`, never a traceback; ``-v`` adds the debug logs.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tokeye.cli import _options

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterator

    import torch.nn as nn

    from tokeye.config import SpectrogramConfig

logger = logging.getLogger(__name__)

EXIT_OK = 0  # every input succeeded
EXIT_FAILED = 1  # at least one input failed
# bad flag, unknown model, no inputs, missing extra, model cannot be loaded
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130  # Ctrl-C

NO_INPUT_HINT = " (no data yet? create a demo signal with: tokeye example)"


def one_line(text: object) -> str:
    """``text`` with every run of whitespace (newlines too) made one space."""
    return " ".join(str(text).split())


def error(message: str) -> int:
    """Print ``error: message`` to stderr and return :data:`EXIT_USAGE`."""
    print(one_line(f"error: {message}"), file=sys.stderr)
    return EXIT_USAGE


def print_hub_error(name: str, exc: Exception) -> None:
    """Print one line for a Hugging Face error (the hub answered)."""
    from tokeye.hub import repo_for

    text = one_line(exc)
    if not text.endswith((".", "!", "?")):
        text += "."
    print(
        one_line(
            f"error: could not download model {name!r} from Hugging Face "
            f"repo {repo_for(name)!r}: {text} If the repo has moved, set "
            "TOKEYE_HF_REPO to override."
        ),
        file=sys.stderr,
    )


def report_failure(path: Path, exc: Exception) -> None:
    """Print the one-line failure of one input; ``-v`` adds its traceback."""
    # The path and the fixed words stay as they are; only the exception text
    # (which can hold newlines) is collapsed.
    print(
        f"error: failed to process {path}: {type(exc).__name__}: {one_line(exc)}",
        file=sys.stderr,
    )
    logger.debug("failed to process %s", path, exc_info=exc)


def report_unexpected(exc: Exception) -> int:
    """The last resort for an exception that escaped a handler."""
    logger.debug("unexpected error", exc_info=exc)
    print(
        one_line(
            f"error: unexpected {type(exc).__name__}: {exc} "
            "(rerun with -v for the traceback)"
        ),
        file=sys.stderr,
    )
    return EXIT_FAILED


def report_model_error(name: str, exc: Exception, *, download: bool = False) -> int:
    """Print one line for a failed model load (or ``tokeye download``).

    The checks run in this order, because the types overlap: on
    huggingface_hub 0.x, ``LocalEntryNotFoundError`` is also an
    ``HfHubHTTPError``, a ``FileNotFoundError`` and a ``ValueError``.
    """
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    from tokeye.hub import DownloadError

    if isinstance(exc, (LocalEntryNotFoundError, DownloadError)):
        logger.debug("model %r: %s", name, exc)
        if download:
            return error(
                f"cannot reach Hugging Face to download {name!r}: {exc}; check "
                "the network connection (and that HF_HUB_OFFLINE is not set)"
            )
        return error(
            f"weights for {name!r} are not in the local cache and Hugging Face "
            f"cannot be reached; run `tokeye download {name}` where there is "
            "internet access (e.g. a login node)"
        )
    if isinstance(exc, HfHubHTTPError):
        print_hub_error(name, exc)
        return EXIT_USAGE
    if isinstance(exc, (ValueError, ImportError, OSError)):
        return error(str(exc))
    logger.debug("model %r could not be loaded", name, exc_info=exc)
    return error(f"{type(exc).__name__}: {exc}")


@contextlib.contextmanager
def verbose_logging(enabled: bool) -> Iterator[None]:
    """With ``-v``, send the ``tokeye`` logger (only) to stderr at DEBUG.

    The handler is added for this call and removed afterwards: ``main`` can
    run many times in one process, each time with its own ``sys.stderr``.
    """
    if not enabled:
        yield
        return
    package_logger = logging.getLogger("tokeye")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    level = package_logger.level
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        package_logger.removeHandler(handler)
        package_logger.setLevel(level)


@contextlib.contextmanager
def quiet_hub_logs(verbose: bool) -> Iterator[None]:
    """Without ``-v``, hide huggingface_hub's retry WARNINGs (one-line rule)."""
    if verbose:
        yield
        return
    hub_logger = logging.getLogger("huggingface_hub")
    level = hub_logger.level
    hub_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        hub_logger.setLevel(level)


@dataclass(frozen=True)
class Setup:
    """What :func:`setup_or_report` checked and loaded."""

    config: SpectrogramConfig
    paths: list[Path]
    model: nn.Module
    channels: tuple[str, ...]
    out_dir: Path
    device: str


def setup_or_report(
    args: argparse.Namespace, task: str, *, unique_stems: bool
) -> Setup | None:
    """The start-up checks of ``tokeye run``, ``elmspec`` and ``alfvenspec``.

    In order: the spectrogram flags, ``--window``, ``--device``, ``--tile``
    (when the command has it), the inputs, with ``unique_stems`` that no
    two share a stem (:func:`tokeye.batch.check_unique_stems`), the model
    (which must suit ``task``), and the output directory, which only the
    last step creates.

    Parameters
    ----------
    args
        The parsed command line.
    task
        ``"segmentation"`` or ``"instance"``.
    unique_stems
        Whether this run writes per-input files named after the stem. It
        has no default, so a new command cannot skip the check by omission.

    Returns
    -------
    Setup or None
        ``None`` after printing one ``error:`` line for the first failure;
        the handler then returns :data:`EXIT_USAGE`.
    """
    try:
        return _setup(args, task, unique_stems=unique_stems)
    except Exception as exc:  # noqa: BLE001 - one line, never a traceback
        logger.debug("start-up failed", exc_info=exc)
        error(f"{type(exc).__name__}: {exc}")
        return None


def _window_error(config: SpectrogramConfig) -> str | None:
    """Why scipy rejects ``config.window``, or ``None`` if it is usable."""
    from scipy.signal import get_window

    try:
        get_window(config.window, config.n_fft)
    except (ValueError, TypeError) as exc:
        return f"--window {config.window!r}: {exc}"
    return None


def _setup(args: argparse.Namespace, task: str, *, unique_stems: bool) -> Setup | None:
    from tokeye import batch, hub
    from tokeye.inference import check_tile

    try:
        config = _options.config_from_args(args)
    except (ValueError, TypeError) as exc:
        error(str(exc))
        return None
    window_error = _window_error(config)
    if window_error is not None:
        error(window_error)
        return None
    try:
        device = hub.resolve_device(args.device)
        if hasattr(args, "tile"):
            check_tile(args.tile)
    except (ValueError, TypeError) as exc:
        error(str(exc))
        return None
    try:
        paths = batch.collect_inputs(list(args.inputs))
    except ValueError as exc:
        error(f"{exc}{NO_INPUT_HINT}")
        return None
    if unique_stems:
        try:
            batch.check_unique_stems(paths)
        except ValueError as exc:  # no NO_INPUT_HINT: there are inputs
            error(str(exc))
            return None
    try:
        with quiet_hub_logs(getattr(args, "verbose", False)):
            hub.require_task(args.model, task)
            model = hub.load_model(args.model, device)
    except Exception as exc:  # noqa: BLE001 - report_model_error maps every type
        report_model_error(args.model, exc)
        return None
    out_dir = Path(args.output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        error(f"cannot create output directory {out_dir}: {exc.strerror or exc}")
        return None
    return Setup(config, paths, model, hub.channels_for(args.model), out_dir, device)
