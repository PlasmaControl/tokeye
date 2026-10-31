"""CPU inference for the DIII-D (Omega) deployment: thread sizing + OpenVINO.

Omega is mostly a CPU cluster, so the deployed env ships CPU-only torch plus
OpenVINO, and this module picks the fastest available engine:

* :func:`configure_threads` — size the intra-op thread pool to what this process
  may actually use (``$TOKEYE_THREADS`` > Slurm allocation > cgroup CPU quota >
  CPU affinity). The Omega login profile exports ``OMP_NUM_THREADS=1``, which
  would otherwise pin torch to a single core.
* :func:`load_runner` — a cached model *runner* with ``.infer(spec) -> mask``.
  ``backend="auto"`` (default, or ``$TOKEYE_BACKEND``) uses OpenVINO when it is
  installed and falls back to torch. The OpenVINO IR is converted once and
  cached next to the weights (``$TOKEYE_OV_DIR``), so later loads skip the
  torch trace entirely.

Runners are accepted anywhere a torch model is: :func:`tokeye.inference.model_infer`
dispatches on ``infer_array``, so the GUI, the web app and the batch CLIs all
pick up the faster engine through :func:`tokeye.app.analyze.load.model_load`.

Import-cheap: numpy only at module scope; torch/OpenVINO load on first use.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

BACKENDS = ("auto", "openvino", "torch")
_THREADS: int | None = None
_RUNNERS: dict[tuple[str, str, int], object] = {}
_RUNNERS_LOCK = threading.Lock()


# ── threads ──────────────────────────────────────────────────────────────────
def _cgroup_cpu_quota() -> int | None:
    """CPUs allowed by the cgroup CPU quota (v1 or v2), or ``None`` if unlimited.

    On somega every user slice is capped (``CPUQuota=1000%`` -> 10 CPUs) although
    the node has 128; running 128 threads under a 10-CPU quota thrashes.
    """
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    candidates: list[tuple[Path, str]] = []
    for line in lines:
        _, controllers, rel = line.split(":", 2)
        rel = rel.lstrip("/")
        if controllers == "":  # cgroup v2
            base = Path("/sys/fs/cgroup") / rel
            candidates.append((base / "cpu.max", "v2"))
        elif "cpu" in controllers.split(","):
            for mount in ("cpu,cpuacct", "cpu", "cpuacct,cpu"):
                base = Path("/sys/fs/cgroup") / mount / rel
                candidates.append((base, "v1"))
    best: int | None = None
    for path, kind in candidates:
        try:
            if kind == "v2":
                quota_s, period_s = path.read_text().split()[:2]
                if quota_s == "max":
                    continue
                quota, period = int(quota_s), int(period_s)
            else:
                quota = int((path / "cpu.cfs_quota_us").read_text())
                period = int((path / "cpu.cfs_period_us").read_text())
        except (OSError, ValueError):
            continue
        if quota > 0 and period > 0:
            n = max(1, quota // period)
            best = n if best is None else min(best, n)
    return best


def available_cpus() -> int:
    """How many CPUs this process can usefully run on."""
    env = os.environ.get("TOKEYE_THREADS")
    if env and env.strip().isdigit() and int(env) > 0:
        return int(env)
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        n = min(n, int(slurm))
    quota = _cgroup_cpu_quota()
    if quota is not None:
        n = min(n, quota)
    return max(1, n)


def configure_threads(n: int | None = None) -> int:
    """Set (and remember) the inference thread count; returns it.

    Applies to torch immediately if it is already imported, and to every runner
    created afterwards (OpenVINO reads it at compile time).
    """
    global _THREADS
    _THREADS = int(n) if n and int(n) > 0 else available_cpus()
    import sys

    if "torch" in sys.modules:
        _set_torch_threads(_THREADS)
    return _THREADS


def threads() -> int:
    return _THREADS if _THREADS is not None else configure_threads()


def _set_torch_threads(n: int) -> None:
    import torch

    if torch.get_num_threads() != n:
        torch.set_num_threads(n)


# ── spectrogram with axes ───────────────────────────────────────────────────────
def spectrogram(
    x: np.ndarray,
    fs: float,
    t0_ms: float = 0.0,
    n_fft: int = 1024,
    hop: int = 128,
    clip_dc: bool = True,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(spec, t_ms, f_khz)`` — :func:`tokeye.transforms.compute_stft` + exact axes.

    The image is exactly what the GUI feeds the model; the axes come from the
    same ``scipy.signal.ShortTimeFFT`` geometry (frame centres, incl. the
    half-window frames scipy adds at both ends).
    """
    from scipy import signal as sps

    from tokeye.transforms import compute_stft

    x = np.asarray(x, dtype=np.float64).ravel()
    spec = compute_stft(
        x[None, :],
        n_fft=n_fft,
        hop=hop,
        window=window,
        clip_dc=clip_dc,
        clip_low=clip_low,
        clip_high=clip_high,
    )
    sft = sps.ShortTimeFFT(win=sps.get_window(window, n_fft), hop=hop, fs=float(fs))
    t_ms = float(t0_ms) + sft.t(x.size) * 1e3
    f_khz = sft.f / 1e3
    if clip_dc:
        f_khz = f_khz[1:]
    if spec.shape != (f_khz.size, t_ms.size):  # pragma: no cover - scipy contract
        raise RuntimeError(
            f"STFT shape {spec.shape} does not match axes ({f_khz.size}, {t_ms.size})"
        )
    return spec.astype(np.float32, copy=False), t_ms, f_khz


# ── runners ────────────────────────────────────────────────────────────────────
def _normalize(spec: np.ndarray) -> np.ndarray:
    """Per-image standardisation — byte-for-byte what ``model_infer`` does."""
    spec = np.asarray(spec, dtype=np.float64)
    out = (spec - spec.mean()) / (spec.std() + 1e-6)
    return out.astype(np.float32, copy=False)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float32)
    np.negative(z, out=out)
    np.exp(out, out=out)
    out += 1.0
    np.reciprocal(out, out=out)
    return out


class TorchRunner:
    """Plain torch (CPU or GPU) — the reference implementation."""

    backend = "torch"

    def __init__(self, model: nn.Module, model_name: str) -> None:
        self.model = model
        self.model_name = model_name

    def logits(self, x: np.ndarray) -> np.ndarray:
        """``(B, 1, H, W)`` normalized input -> ``(B, 2, H, W)`` logits."""
        import torch

        device = next(self.model.parameters()).device
        if device.type == "cpu":
            _set_torch_threads(threads())
        with torch.inference_mode():
            out = self.model(torch.from_numpy(x).to(device))
        if isinstance(out, (tuple, list)):  # the U-Net returns a 1-tuple
            out = out[0]
        return out.float().cpu().numpy()

    def infer(self, spec: np.ndarray) -> np.ndarray:
        """``(H, W)`` spectrogram -> ``(2, H, W)`` float32 probabilities."""
        x = _normalize(spec)[None, None]
        return _sigmoid(self.logits(x)[0])


class OpenVINORunner(TorchRunner):
    """OpenVINO CPU runtime on a dynamic-shape IR of the same weights."""

    backend = "openvino"

    def __init__(self, compiled, model_name: str, precision: str) -> None:
        self.compiled = compiled
        self.model_name = model_name
        self.precision = precision
        self._lock = threading.Lock()

    def logits(self, x: np.ndarray) -> np.ndarray:
        with self._lock:  # one request at a time; the GUI serializes anyway
            result = self.compiled(np.ascontiguousarray(x, dtype=np.float32))
        return np.asarray(result[0], dtype=np.float32)


def resolve_backend(backend: str | None = None) -> str:
    choice = (backend or os.environ.get("TOKEYE_BACKEND") or "auto").strip().lower()
    if choice not in BACKENDS:
        raise ValueError(f"unknown backend {choice!r} (choices: {', '.join(BACKENDS)})")
    return choice


def model_fingerprint(model: nn.Module) -> str:
    """Short hash of a module's actual weights (names, shapes, dtypes, bytes).

    The OpenVINO IR cache is keyed on this — the content of the very module that
    gets converted — so a cached IR can never stand in for different weights
    (keying on a file name/digest let a stubbed model poison the cache).
    """
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        t = tensor.detach().cpu().contiguous()
        h.update(f"{name}:{tuple(t.shape)}:{t.dtype}".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


def _save_ir(ov, ov_model, xml: Path) -> None:
    """Write the IR atomically: into a fresh private ``mkdtemp`` dir (no
    predictable names, nothing to follow), then rename .bin before .xml.
    Skipped silently if the cache dir is read-only."""
    import shutil
    import tempfile

    tmpdir = None
    try:
        xml.parent.mkdir(parents=True, exist_ok=True)
        tmpdir = Path(tempfile.mkdtemp(dir=xml.parent, prefix=".ir-"))
        tmp = tmpdir / xml.name
        ov.save_model(ov_model, str(tmp), compress_to_fp16=False)
        for f in (tmp.with_suffix(".bin"), tmp):
            f.chmod(0o664)
        tmp.with_suffix(".bin").replace(xml.with_suffix(".bin"))
        tmp.replace(xml)
    except OSError as exc:  # read-only shared cache: just don't persist
        logger.info("not caching OpenVINO IR at %s: %s", xml, exc)
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)


def ov_cache_dir() -> Path:
    env = os.environ.get("TOKEYE_OV_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "tokeye" / "openvino"


def _openvino_runner(model: str, n_threads: int) -> OpenVINORunner:
    import openvino as ov
    import torch

    from tokeye import hub

    torch_model = hub.load_model(model, "cpu").eval()
    xml = ov_cache_dir() / f"{Path(model).stem}_{model_fingerprint(torch_model)}.xml"
    core = ov.Core()
    if xml.is_file() and xml.with_suffix(".bin").is_file():
        ov_model = core.read_model(str(xml))
    else:
        logger.info("converting %s to OpenVINO IR (one-time)", model)
        ov_model = ov.convert_model(
            torch_model,
            example_input=torch.zeros(1, 1, 512, 512),
            input=[ov.PartialShape([-1, 1, -1, -1])],
        )
        _save_ir(ov, ov_model, xml)
    precision = os.environ.get("TOKEYE_OV_PRECISION", "f32")
    config = {
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_NUM_THREADS": int(n_threads),
        "INFERENCE_PRECISION_HINT": precision,
    }
    compiled = core.compile_model(ov_model, "CPU", config)
    return OpenVINORunner(compiled, Path(str(model)).stem, precision)


def load_runner(
    model: str | Path | None = None,
    backend: str | None = None,
    device: str = "cpu",
):
    """Cached runner for ``model`` with ``.infer(spec) -> (2, H, W)`` probabilities.

    ``device`` other than ``cpu`` (e.g. ``cuda``/``auto`` on a GPU node) always
    uses torch. On CPU, ``auto`` prefers OpenVINO and silently falls back to torch
    if it is missing or the conversion fails.
    """
    from tokeye import hub

    name = str(model) if model else hub.DEFAULT_MODEL
    choice = resolve_backend(backend)
    resolved_device = hub.resolve_device(device)
    n_threads = threads()
    if resolved_device != "cpu":
        choice = "torch"
    key = (name, f"{choice}:{resolved_device}", n_threads)
    with _RUNNERS_LOCK:
        cached = _RUNNERS.get(key)
        if cached is not None:
            return cached
        runner = None
        if choice in ("auto", "openvino"):
            try:
                runner = _openvino_runner(name, n_threads)
            except ImportError:
                if choice == "openvino":
                    raise
            except Exception as exc:
                if choice == "openvino":
                    raise
                logger.warning("OpenVINO unavailable for %s (%s); using torch", name, exc)
        if runner is None:
            torch_model = hub.load_model(name, resolved_device)
            if resolved_device == "cpu":
                _set_torch_threads(n_threads)
            runner = TorchRunner(torch_model, Path(name).stem)
        _RUNNERS[key] = runner
        return runner


def git_commit() -> str | None:
    """Short commit of the checkout this package runs from (``None`` if not a repo)."""
    # Read .git directly: no subprocess (the launcher's LD_LIBRARY_PATH points at
    # the conda env, which system git should not be made to load).
    git = Path(__file__).resolve().parents[2] / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head[:7]
        ref = head.split(None, 1)[1]
        loose = git / ref
        if loose.is_file():
            return loose.read_text().strip()[:7]
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0][:7]
    except (OSError, IndexError):
        return None
    return None
