from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import ShortTimeFFT, get_window
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


data_path = Path("/scratch/gpfs/nc1514/TokEye/data/eval/D3D2025")
model_path = Path("/scratch/gpfs/nc1514/TokEye/model/big_mode_v1.pt")

model = torch.load(
    model_path,
    weights_only=False,
    )
model.to(device)
model.eval()
print("Model loaded")

SFT = ShortTimeFFT(
    win=get_window("hann", 1024),
    hop=512,
    fs=500,
)

# Find and loop through shot numbers
shotns = [d.name for d in data_path.iterdir() if d.is_dir() and d.name.isdigit()]
for shotn in tqdm(shotns, desc="Processing shot numbers"):
    raw_path = data_path / shotn / "raw"
    spec_path = data_path / shotn / "spec"
    spec_path.mkdir(parents=True, exist_ok=True)

    # Find and loop through signal files
    signal_files = list(raw_path.glob("*.npy"))
    for signal_path in tqdm(signal_files, desc="Processing signal files"):
        try:
            signal_name = signal_path.name

            # Load signal
            signal = np.load(signal_path)
            signal = SFT.stft(signal)
            signal = np.abs(signal)**2
            signal = np.log1p(signal)
            vmin, vmax = np.percentile(signal, (1, 99))
            signal = np.clip(signal, vmin, vmax)
            signal = signal[1:]

            # Model inference
            signal_tensor = torch.from_numpy(signal).to(device)
            signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0).float()
            signal_tensor = (signal_tensor - signal_tensor.mean()) / signal_tensor.std()
            with torch.no_grad():
                output = model(signal_tensor)
                output = output[0]
            output = output.squeeze(0)
            output = torch.sigmoid(output)
            output = output[0].cpu().numpy()
            output[output < 0.15] = 0

            # Save output
            fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(15, 6))
            axes[0].imshow(signal, aspect='auto', origin='lower', cmap='gist_heat')
            axes[0].set_ylabel('Frequency')
            axes[1].imshow(output, aspect='auto', origin='lower', cmap='gist_heat', vmin=0, vmax=1)
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Frequency')
            plt.suptitle(f'{signal_path.stem}')
            plt.tight_layout()

            # Save outputs
            output_spec_path = spec_path / f"{signal_path.stem}.npy"
            output_fig_path = spec_path / f"{signal_path.stem}.png"
            np.save(output_spec_path, signal)
            plt.savefig(output_fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error processing {signal_path}: {e}")
            continue
