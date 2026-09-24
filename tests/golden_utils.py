"""Golden-file helpers for the numbers guard (C0).

The golden files pin the numerical output of the default path:

- ``data/golden_big_tf_unet.npz`` -- ``big_tf_unet`` on the seeded example
  signal with the default config (summary statistics, an 8x8 average-pooled
  mask, and a full-resolution centre crop);
- ``data/stft_contract.npz`` -- ``compute_stft`` on a 1D and a 2-row input.

Regenerate ONLY when the default weights change on purpose (and add a
CHANGELOG entry)::

    uv run --no-sync python tests/golden_utils.py --write
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

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


def _weights_path() -> Path:
    from huggingface_hub import try_to_load_from_cache

    from tokeye import hub

    spec = hub.MODEL_REGISTRY[hub.DEFAULT_MODEL]
    cached = try_to_load_from_cache(hub.repo_for(spec.name), spec.filename)
    if not isinstance(cached, str):
        raise SystemExit("weights not cached; run: tokeye download")
    return Path(cached)


def _generate() -> None:
    import torch

    from tokeye import TokEye
    from tokeye.transforms import compute_stft

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
        weights_sha256=np.array(sha256(_weights_path())),
    )
    x, y = stft_inputs()
    np.savez_compressed(
        STFT_PATH,
        one=compute_stft(x[np.newaxis], **STFT_KW),
        cross=compute_stft(np.stack([x, y]), **STFT_KW),
    )
    for path in (GOLDEN_PATH, STFT_PATH):
        print(f"{path}  {path.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: python tests/golden_utils.py --write")
    _generate()
