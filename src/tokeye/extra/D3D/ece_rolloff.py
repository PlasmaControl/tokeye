from pathlib import Path
import argparse
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

default_settings = {
    "csv_path": "data/extra/D3D/ece_rolloff.csv",
}


def load_rolloff(csv_path: str = default_settings["csv_path"]):
    """Load rolloff frequency and coefficients from CSV."""
    df = pd.read_csv(csv_path, header=None)
    freq_rolloff = df.iloc[:, 0].values
    rolloff_coeff = df.iloc[:, 1].values
    return freq_rolloff, rolloff_coeff


def interpolate_rolloff(spec_shape: tuple, freq_rolloff: np.ndarray, 
                        rolloff_coeff: np.ndarray, fs: int) -> np.ndarray:
    """Interpolate rolloff coefficients to match spectrogram frequency bins."""
    freq_bins = spec_shape[0]
    max_freq = freq_rolloff.max()
    freq_per_bin = max_freq / fs
    spec_freq_array = np.arange(freq_bins) * freq_per_bin
    return np.interp(spec_freq_array, freq_rolloff, rolloff_coeff)


def apply_rolloff(spec: np.ndarray, rolloff_coeff: np.ndarray) -> np.ndarray:
    """Apply rolloff correction to spectrogram."""
    return spec * rolloff_coeff[:, np.newaxis]


def save_spectrogram(spec: np.ndarray, input_path: Path, output_dir: Path = Path("data/output")):
    """Save corrected spectrogram to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name
    np.save(output_path, spec)
    print(f"Saved corrected spectrogram to: {output_path}")
    return output_path


def process_spectrogram(input_path: str, fs: int, csv_path: str) -> Path:
    """Main processing pipeline for applying ECE rolloff correction."""
    logger.info(f"Apply rolloff with a sampling frequency of {fs} kHz")
    
    spec = np.load(Path(input_path))
    freq_rolloff, rolloff_coeff = load_rolloff(csv_path)
    spec_rolloff_coeff = interpolate_rolloff(spec.shape, freq_rolloff, rolloff_coeff, fs)
    corrected_spec = apply_rolloff(spec, spec_rolloff_coeff)
    
    return save_spectrogram(corrected_spec, Path(input_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="Path to input spectrogram")
    parser.add_argument("fs", type=int, help="Sampling frequency [kHz]", default=500)
    parser.add_argument("--csv", type=str, help="Path to CSV file", default=default_settings["csv_path"])
    args = parser.parse_args()
    
    process_spectrogram(args.input_path, args.fs, args.csv)
