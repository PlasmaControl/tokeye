"""
Multiscale Data Preparation for UNet Training

This script prepares training data for multiscale spectrogram segmentation.
For each diagnostic channel:
1. Load raw time-series data
2. Create base-scale spectrogram (nfft=1024, hop=128) with DC bin removed
3. Run pretrained model to get segmentation labels (in chunks to avoid OOM)
4. Save timeseries + base label for each channel

During training, spectrograms at different scales are created on-the-fly,
and labels are resized to match using bicubic interpolation.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from TokEye.autoprocess.utils.configuration import load_input_paths, setup_directory
from tqdm import tqdm

torch._dynamo.config.recompile_limit = 128

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base scale parameters (original model trained on these)
BASE_NFFT = 1024
BASE_HOP = 128

# Channel selection per diagnostic
# Only process these channels for each diagnostic type
DIAGNOSTIC_CHANNELS = {
    "bes": [26, 28, 30, 32, 34, 36, 38, 40],
    "co2": None,  # None means all channels
    "ece": list(range(9, 19)),  # 9 to 18 inclusive
    "mhr": [3, 4, 5, 6],
}


def make_spectrogram(
    data: np.ndarray,
    window_size: int,
    hop_size: int,
) -> torch.Tensor:
    assert data.ndim == 1, "Data must be 1D"
    data_t = torch.from_numpy(data).float().to(device)

    Sxx = torch.stft(
        data_t,
        n_fft=window_size,
        window=torch.hann_window(window_size, device=device),
        hop_length=hop_size,
        return_complex=True,
    )
    Sxx = Sxx.abs() ** 2
    Sxx = Sxx.log1p()
    if Sxx.shape[1] % 2 == 1:
        Sxx = Sxx[:, :-1]
    return Sxx


def normalize_spectrogram(Sxx: torch.Tensor) -> torch.Tensor:
    Sxx_norm = (Sxx - Sxx.mean()) / (Sxx.std() + 1e-8)
    vmin = torch.quantile(Sxx_norm.flatten(), 0.01)
    vmax = torch.quantile(Sxx_norm.flatten(), 0.99)
    return torch.clip(Sxx_norm, vmin, vmax)


def process_spectrogram(
    Sxx: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
    Sxx_in = Sxx.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred = model(Sxx_in)[0]
    pred = pred.squeeze(0)
    pred = torch.sigmoid(pred)

    pred_min, pred_max = 0.2, 1
    pred[pred < pred_min] = 0
    pred[pred > pred_max] = 1

    return pred.cpu()


def process_diagnostic(
    diagnostic_key: str,
    data_dir: Path,
    output_dir: Path,
    model: torch.nn.Module,
    channels: list[int] | None = None,
):
    if not data_dir.exists():
        print(f"Skipping {diagnostic_key}: no data found at {data_dir}")
        return

    # Setup output directory for this diagnostic
    diag_output_dir = setup_directory(output_dir / diagnostic_key, overwrite=True)

    # Get all data files
    input_paths = load_input_paths(data_dir)
    print(f"Processing {diagnostic_key}: {len(input_paths)} files")

    sample_idx = 0

    for file_path in tqdm(input_paths, desc=f"Processing {diagnostic_key}"):
        try:
            data = joblib.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        shot_name = file_path.stem.split("_")[0]
        shot_dir = setup_directory(diag_output_dir / shot_name, overwrite=True)

        # Get time-series data: shape (num_channels, 1, time_samples)
        timeseries = data[diagnostic_key].squeeze(1)  # (num_channels, time_samples)
        num_channels = timeseries.shape[0]

        # Determine which channels to process
        if channels is None:
            channels_to_process = list(range(num_channels))
        else:
            # Filter to only valid channel indices
            channels_to_process = [ch for ch in channels if ch < num_channels]

        # Process each selected channel
        for ch_idx in channels_to_process:
            channel_data = timeseries[ch_idx]  # Shape: (time_samples,)

            Sxx = make_spectrogram(channel_data, BASE_NFFT, BASE_HOP)
            Sxx = normalize_spectrogram(Sxx)
            label = process_spectrogram(Sxx, model)

            timeseries_path = shot_dir / f"ch{ch_idx:02d}_timeseries.npy"
            label_path = shot_dir / f"ch{ch_idx:02d}_label.npy"

            np.save(timeseries_path, channel_data.astype(np.float32))
            np.save(label_path, label.numpy().astype(np.float32))

            sample_idx += 1

    print(f"Saved {sample_idx} channel samples for {diagnostic_key}")


def main():
    """Main entry point for data preparation."""

    # Configuration
    library_dir = Path("/scratch/gpfs/nc1514/TokEye/data/autoprocess/library")
    output_dir = Path("/scratch/gpfs/nc1514/TokEye/data/.cache/multiscale_prep")
    model_path = Path("/scratch/gpfs/nc1514/TokEye/model/big_mode_v1.pt")

    # Load pretrained model
    print(f"Loading model from {model_path}")
    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
    model.eval()
    print("Model loaded successfully")

    # Diagnostics to process
    diagnostics = {
        "bes": library_dir / "bes_250" / "step_0a_extract_faithdata",
        "co2": library_dir / "co2_250" / "step_0a_extract_faithdata",
        "ece": library_dir / "ece_250" / "step_0a_extract_faithdata",
        "mhr": library_dir / "mhr_250" / "step_0a_extract_faithdata",
    }

    # Setup output directory (don't overwrite root to preserve other diagnostics)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each diagnostic with its specific channel selection
    for diag_key, diag_dir in diagnostics.items():
        if diag_dir.exists():
            channels = DIAGNOSTIC_CHANNELS.get(diag_key)
            process_diagnostic(
                diagnostic_key=diag_key,
                data_dir=diag_dir,
                output_dir=output_dir,
                model=model,
                channels=channels,
            )
        else:
            print(f"Skipping {diag_key}: directory not found at {diag_dir}")

    print("\nData preparation complete!")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    # python /scratch/gpfs/nc1514/TokEye/dev/notebooks/evaluation/make_multiscale.py
    main()
