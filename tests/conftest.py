"""Suite-wide fixtures, collection guards and the torch thread cap.

Tests stay offline by default: the hub is mocked everywhere except in tests
marked ``weights``, which need the real ``big_tf_unet`` weights already in the
Hugging Face cache (``tokeye download`` fetches them).

Optional extras are not installed everywhere (the CI floor job has no gradio,
Windows has no ``fcntl``), so test modules that import them at module level
are skipped at collection time instead of erroring.
"""

from __future__ import annotations

import importlib.util
import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _missing(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is None
    except ModuleNotFoundError:  # parent package itself is missing
        return True


collect_ignore_glob: list[str] = []
if _missing("gradio"):
    collect_ignore_glob.append("test_app_*.py")
if _missing("omegaconf") or _missing("fcntl"):
    collect_ignore_glob.append("test_ablation_orchestrator_paths.py")
if _missing("tokeye.training.big_tf_unet_ablation"):
    collect_ignore_glob += ["test_ablation_*.py", "test_window_filter.py"]


@pytest.fixture(scope="session")
def real_weights() -> Path:
    """Path to the cached default weights; skips when they are not cached."""
    import golden_utils

    return golden_utils.weights_or_skip()


SERIAL_MAX_THREADS = 8


def pytest_configure(config: pytest.Config) -> None:
    """Cap torch's thread pool: a share per xdist worker, at most 8 serially.

    Under ``pytest -n 8`` every worker's thread pool otherwise spans all the
    machine's cores, and torch-heavy tests running side by side thrash (on a
    40-core node the suite took 5.5 minutes instead of 30 seconds). The
    tests' tensors are small, so a serial run gains nothing from more than
    a few threads either: on that node, all 40 made it several times slower
    than a cap. The xdist controller runs no tests, so it skips the torch
    import.
    """
    worker = "PYTEST_XDIST_WORKER" in os.environ
    if not worker and config.getoption("numprocesses", default=None):
        return
    import torch

    if worker:
        workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
        torch.set_num_threads(max(1, torch.get_num_threads() // workers))
    else:
        torch.set_num_threads(min(torch.get_num_threads(), SERIAL_MAX_THREADS))
