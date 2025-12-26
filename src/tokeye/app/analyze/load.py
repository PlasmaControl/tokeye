import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .transforms import compute_stft

logger = logging.getLogger(__name__)

DUMMY_INPUT_SHAPE = (1, 1, 512, 512)  # (batch_size, channels, height, width)
WARMUP_ITERATIONS = 10
MODEL_EXTENSIONS = [".pt", ".pt2"]
MODEL_DIR = Path("model")
SIGNAL_EXTENSIONS = [".npy"]
SIGNAL_DIR = Path("data/input")


# Directory Scanning
def find_models() -> list[str]:
    model_dir = MODEL_DIR
    if not model_dir.exists():
        return []
    models = []
    for ext in MODEL_EXTENSIONS:
        models.extend(model_dir.glob(f"*{ext}"))
    return sorted([str(m) for m in models])


def find_signals(
    dir: str | None = None,
) -> list[str]:
    if dir is None:
        return []

    dir_path = Path(dir)
    if not dir_path.exists() or not dir_path.is_dir():
        return []

    signals = []
    for ext in SIGNAL_EXTENSIONS:
        signals.extend(dir_path.glob(f"*{ext}"))
    return sorted([s.name for s in signals])


# Model Functions
def model_load(
    filepath: Path,
    device: str = "auto",
) -> nn.Module | torch.export.ExportedProgram:
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model: {filepath.name} on {device}")

    match filepath.suffix:
        case ".pt":
            model = torch.load(
                str(filepath),
                map_location=device,
                weights_only=False,
            )
            model.eval()
        case ".pt2":
            module = torch.export.load(str(filepath))
            model = module.module()
            model.to(device)
        case _:
            raise ValueError(f"Unsupported model format: {filepath.suffix}")

    logger.info(f"Warming up model ({WARMUP_ITERATIONS} iterations)...")
    dummy_input = torch.randn(*DUMMY_INPUT_SHAPE, device=device, dtype=torch.float32)
    with torch.no_grad():
        for _ in tqdm(range(WARMUP_ITERATIONS)):
            _ = model(dummy_input)
    logger.info("Model ready for inference")
    return model


def model_infer(
    inp_array: np.ndarray | None,
    model: nn.Module | torch.export.ExportedProgram | None,
) -> np.ndarray | None:
    if inp_array is None or model is None:
        logger.warning("Missing input or model for inference")
        return None

    logger.info(f"Running inference on input shape: {inp_array.shape}")

    device = next(model.parameters()).device
    inp_array = (inp_array - inp_array.mean()) / (inp_array.std() + 1e-6)
    inp_tensor = torch.from_numpy(inp_array)
    inp_tensor = inp_tensor.unsqueeze(0).unsqueeze(0).float()
    inp_tensor = inp_tensor.to(device)

    with torch.no_grad():
        out_tensor = model(inp_tensor)
    out_tensor = out_tensor[0]

    out_tensor = torch.sigmoid(out_tensor)
    out_tensor = out_tensor.squeeze(0).squeeze(0).cpu()
    return out_tensor.numpy()


# Signal Functions
def signal_load(filepath) -> np.ndarray | None:
    try:
        signal = np.load(Path(filepath))
    except Exception as e:
        logger.error(f"Failed to load signal: {e}")
        return None
    if signal.ndim != 1:
        logger.error("Signal must be 1D array")
        return None
    if signal.size == 0:
        logger.error("Signal is empty")
        return None
    return signal


def load_single(
    filepath: Path,
    transform_args: dict,
) -> np.ndarray | None:
    # Load signal
    signal = signal_load(filepath)
    if signal is None:
        logger.error("Failed to load signal")
        return None

    signal_data = np.expand_dims(signal, axis=0)
    logger.info(f"Raw signal shape: {signal.shape}")

    # Apply STFT transform (generalize later)
    n_fft = transform_args.get("n_fft", 1024)
    hop = transform_args.get("hop_length", 256)
    clip_dc = transform_args.get("clip_dc", True)
    clip_low = transform_args.get("percentile_low", 1.0)
    clip_high = transform_args.get("percentile_high", 99.0)

    return compute_stft(
        signal_data,
        n_fft=n_fft,
        hop=hop,
        clip_dc=clip_dc,
        clip_low=clip_low,
        clip_high=clip_high,
    )


def load_multi(
    filepaths: list[Path],
    transform_args: dict,
) -> np.ndarray | None:
    # Load both signals
    signal = []
    for filepath in filepaths:
        signal.append(signal_load(filepath))
    if any(s is None for s in signal):
        logger.error("Failed to load one or more signals")
        return None

    signal = np.array(signal)
    logger.info(f"Raw signal shape: {signal.shape}")

    # Apply STFT to both signals (generalize later)
    n_fft = transform_args.get("n_fft", 1024)
    hop = transform_args.get("hop_length", 256)
    clip_dc = transform_args.get("clip_dc", True)
    clip_low = transform_args.get("percentile_low", 1.0)
    clip_high = transform_args.get("percentile_high", 99.0)

    return compute_stft(
        signal,
        n_fft=n_fft,
        hop=hop,
        clip_dc=clip_dc,
        clip_low=clip_low,
        clip_high=clip_high,
    )
