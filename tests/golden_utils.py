"""Golden-file helpers for the numbers guard (C0).

The golden files pin the numerical output of the default path:

- ``data/golden_big_tf_unet.npz`` -- ``big_tf_unet`` on the seeded example
  signal with the default config (summary statistics, an 8x8 average-pooled
  mask, and a full-resolution centre crop);
- ``data/stft_contract.npz`` -- ``compute_stft`` on a 1D and a 2-row input.

Regenerate one file at a time, each with a CHANGELOG entry: the golden ONLY
when the default weights change on purpose, the STFT contract ONLY when the
STFT contract changes on purpose::

    uv run --no-sync python tests/golden_utils.py --write golden
    uv run --no-sync python tests/golden_utils.py --write stft
"""

from __future__ import annotations

import contextlib
import hashlib
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

USAGE = "usage: python tests/golden_utils.py --write golden|stft"

DATA_DIR = Path(__file__).resolve().parent / "data"
GOLDEN_PATH = DATA_DIR / "golden_big_tf_unet.npz"
STFT_PATH = DATA_DIR / "stft_contract.npz"

POOL = 8
CROP = 128
STFT_KW = {"n_fft": 256, "hop": 64}
STFT_N = 4096


def pool(mask: np.ndarray, k: int = POOL) -> np.ndarray:
    """``k x k`` average pool of a ``(C, H, W)`` mask (edges trimmed)."""
    c, h, w = mask.shape
    trimmed = mask[:, : h - h % k, : w - w % k].astype(np.float64)
    return trimmed.reshape(c, h // k, k, w // k, k).mean(axis=(2, 4))


def channel_stats(mask: np.ndarray) -> np.ndarray:
    """``(C, 5)``: mean, std, min, max and fraction >= 0.5 per channel."""
    m = mask.astype(np.float64).reshape(mask.shape[0], -1)
    return np.stack(
        [m.mean(1), m.std(1), m.min(1), m.max(1), (m >= 0.5).mean(1)], axis=1
    )


def spec_stats(spec: np.ndarray) -> np.ndarray:
    s = spec.astype(np.float64)
    return np.array([s.mean(), s.std(), s.min(), s.max()])


def crop_origin(shape: tuple[int, int], crop: int = CROP) -> tuple[int, int]:
    h, w = shape
    return (h - crop) // 2, (w - crop) // 2


def centre_crop(mask: np.ndarray, crop: int = CROP) -> np.ndarray:
    r0, c0 = crop_origin(mask.shape[1:], crop)
    return mask[:, r0 : r0 + crop, c0 : c0 + crop]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stft_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Two seeded, partly coherent signals for the STFT contract."""
    rng = np.random.default_rng(1234)
    t = np.arange(STFT_N)
    x = np.sin(2 * np.pi * 0.05 * t) + 0.1 * rng.standard_normal(STFT_N)
    y = np.sin(2 * np.pi * 0.05 * t + 0.3) + 0.1 * rng.standard_normal(STFT_N)
    return x, y


def golden_inputs() -> np.ndarray:
    from tokeye.examples import make_example_signal

    return make_example_signal()


def cached_weights() -> Path | None:
    """The cached default weights, or ``None`` when they are not cached."""
    from tokeye import hub

    return hub.cached_path(hub.DEFAULT_MODEL)


def weights_or_skip() -> Path:
    """The cached default weights; skips the calling test when they are absent."""
    import pytest

    from tokeye import hub

    path = cached_weights()
    if path is None:
        pytest.skip(f"{hub.DEFAULT_MODEL} weights not cached; run: tokeye download")
    return path


def _weights_path() -> Path:
    path = cached_weights()
    if path is None:
        raise SystemExit("weights not cached; run: tokeye download")
    return path


@contextlib.contextmanager
def fp32_cuda() -> Iterator[None]:
    """Run CUDA convolutions and matmuls in full fp32 (no TF32) inside the block.

    cuDNN convolutions default to TF32 on Ampere and newer GPUs, which moves
    the mask by more than the golden's tolerance. ``benchmark`` and
    ``deterministic`` keep their current values, and the matmul precision is
    restored afterwards, also when the block raises.
    """
    import torch

    cudnn = torch.backends.cudnn
    prec = torch.get_float32_matmul_precision()
    with cudnn.flags(
        enabled=True,
        benchmark=cudnn.benchmark,
        deterministic=cudnn.deterministic,
        allow_tf32=False,
    ):
        try:
            torch.set_float32_matmul_precision("highest")
            yield
        finally:
            torch.set_float32_matmul_precision(prec)


def _atol(device: str, golden: Mapping[str, Any]) -> float:
    """1e-5 where the golden was made, 1e-4 elsewhere.

    "Where" is the CPU of the golden's platform and machine, with its torch
    base version. Goldens without ``platform``/``machine`` were made on
    linux/x86_64.
    """
    import torch

    base = torch.__version__.split("+")[0]
    golden_base = str(golden["torch_version"]).split("+")[0]
    made_on = str(golden.get("platform", "linux"))
    made_machine = str(golden.get("machine", "x86_64"))
    exact = (
        device == "cpu"
        and sys.platform == made_on
        and platform.machine() == made_machine
        and base == golden_base
    )
    return 1e-5 if exact else 1e-4


def _write_golden() -> None:
    """Rewrite ``GOLDEN_PATH`` only: TokEye on the CPU, with the weights' sha256."""
    import torch

    from tokeye import TokEye

    eye = TokEye(device="cpu")
    signal = golden_inputs()
    spec = eye.spectrogram(signal)
    mask = eye.predict(signal)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        GOLDEN_PATH,
        shape=np.array(mask.shape),
        channel_stats=channel_stats(mask),
        spec_stats=spec_stats(spec),
        pool=pool(mask).astype(np.float32),
        crop=centre_crop(mask).astype(np.float32),
        crop_origin=np.array(crop_origin(mask.shape[1:])),
        torch_version=np.array(torch.__version__),
        platform=np.array(sys.platform),
        machine=np.array(platform.machine()),
        weights_sha256=np.array(sha256(_weights_path())),
    )
    print(f"{GOLDEN_PATH}  {GOLDEN_PATH.stat().st_size / 1e3:.0f} kB")


def _write_stft() -> None:
    """Rewrite ``STFT_PATH`` only: ``compute_stft``, with no torch or weights."""
    from tokeye.transforms import compute_stft

    x, y = stft_inputs()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        STFT_PATH,
        one=compute_stft(x[np.newaxis], **STFT_KW),
        cross=compute_stft(np.stack([x, y]), **STFT_KW),
    )
    print(f"{STFT_PATH}  {STFT_PATH.stat().st_size / 1e3:.0f} kB")


def main(argv: list[str] | None = None) -> None:
    """``--write golden`` or ``--write stft``; anything else exits with USAGE."""
    args = sys.argv[1:] if argv is None else argv
    # Looked up at call time, so a monkeypatched writer is the one called.
    if args == ["--write", "golden"]:
        _write_golden()
    elif args == ["--write", "stft"]:
        _write_stft()
    else:
        raise SystemExit(USAGE)


if __name__ == "__main__":
    main()
