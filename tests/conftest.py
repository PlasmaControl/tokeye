"""Suite-wide fixtures, collection guards and the xdist thread cap.

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
from pathlib import Path

import pytest


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
    from huggingface_hub import try_to_load_from_cache

    from tokeye import hub

    spec = hub.MODEL_REGISTRY[hub.DEFAULT_MODEL]
    cached = try_to_load_from_cache(hub.repo_for(spec.name), spec.filename)
    if not isinstance(cached, str):
        pytest.skip(f"{spec.name} weights not cached; run: tokeye download")
    return Path(cached)


def pytest_configure(config: pytest.Config) -> None:
    """Give each xdist worker its share of the torch threads.

    Under ``pytest -n 8`` every worker's thread pool otherwise spans all the
    machine's cores, and torch-heavy tests running side by side thrash (on a
    40-core node the suite took 5.5 minutes instead of 30 seconds). Serial
    runs keep every thread.
    """
    if "PYTEST_XDIST_WORKER" not in os.environ:
        return
    import torch

    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    torch.set_num_threads(max(1, torch.get_num_threads() // workers))
