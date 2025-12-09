import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_rolloff_coefficients(file_path="data/extra/D3D/ece_rolloff.csv"):
    """Load rolloff correction coefficients from CSV file."""
    try:
        df = pd.read_csv(file_path, header=None)
        freq_rolloff = df.iloc[:, 0].values
        rolloff_coeff = df.iloc[:, 1].values
        return freq_rolloff, rolloff_coeff
    except Exception as e:
        raise RuntimeError(f"Error loading rolloff coefficients: {e}")


if __name__ == "__main__":
    # Parse arguments
    if len(sys.argv) < 3:
        print(
            "Usage: python ece_rolloff.py <spectrogram.npy> <bin_index> [--csv <path>]"
        )
        print("  bin_index: integer or 'max' for highest frequency bin")
        sys.exit(1)

    input_path = sys.argv[1]
    bin_arg = sys.argv[2]

    # Parse optional CSV path
    csv_path = "data/extra/D3D/ece_rolloff.csv"
    if "--csv" in sys.argv:
        csv_idx = sys.argv.index("--csv") + 1
        if csv_idx < len(sys.argv):
            csv_path = sys.argv[csv_idx]

    # Load spectrogram
    try:
        spec = np.load(Path(input_path))
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if spec.ndim != 2:
        print(f"Error: Input must be 2D spectrogram, got shape {spec.shape}")
        sys.exit(1)

    # Load rolloff coefficients
    freq_rolloff, rolloff_coeff = load_rolloff_coefficients(csv_path)

    # Parse bin index
    if bin_arg.lower() == "max":
        bin_index = spec.shape[0] - 1
    else:
        try:
            bin_index = int(bin_arg)
            if bin_index <= 0 or bin_index >= spec.shape[0]:
                print(
                    f"Error: bin_index must be in range (0, {spec.shape[0]}), got {bin_index}"
                )
                sys.exit(1)
        except ValueError:
            print(f"Error: bin_index must be integer or 'max', got '{bin_arg}'")
            sys.exit(1)

    # Calculate frequency mapping
    freq_bins = spec.shape[0]
    max_freq = freq_rolloff.max()
    freq_per_bin = max_freq / bin_index

    # Generate frequency array for all spectrogram bins
    spec_freq_array = np.arange(freq_bins) * freq_per_bin

    # Interpolate rolloff coefficients to match spectrogram bins
    spec_rolloff_coeff = np.interp(spec_freq_array, freq_rolloff, rolloff_coeff)

    # Apply correction (broadcast multiply across time axis)
    corrected_spec = spec * spec_rolloff_coeff[:, np.newaxis]

    # Save output
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_name = Path(input_path).name
    output_path = output_dir / input_name

    np.save(output_path, corrected_spec)
    print(f"Saved corrected spectrogram to: {output_path}")
