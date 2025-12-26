import csv
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from tokeye.extra.eval.silbidopy.data import AudioTonalDataset
from tokeye.extra.eval.silbidopy.eval import Metrics

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Settings
SETTINGS = {"threshold": 0.5}


# Helper functions
class Processor:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std


# Statistics
STATS = {
    "Delphinus capensis": {
        "mean": 0.7080333218912784,
        "std": 0.051389602618240104,
    },
    "Delphinus delphis": {
        "mean": 0.6542849744749529,
        "std": 0.049230019876810985,
    },
    "Peponocephala electra": {
        "mean": 0.7745788766427888,
        "std": 0.06212775435505397,
    },
    "StenellaLongirostrisLongirostris": {
        "mean": 0.7827474513079127,
        "std": 0.05873317039496551,
    },
    "Tursiops truncatus-SoCal": {
        "mean": 0.3825516525048606,
        "std": 0.06272170328810193,
    },
}

# Paths
root_path = Path("/scratch/gpfs/nc1514/TokEye")
eval_base_path = root_path / "data" / "eval" / "DCLDE2011"
model_path = root_path / "model" / "big_mode_v1-5.pt"
output_path = root_path / "data" / "eval" / "results" / "DCLDE2011.csv"

# Load model
model = torch.load(model_path, weights_only=False, map_location=device)
model.eval()
print("Model loaded")

results = []

# Process each species
for species_name in STATS:
    try:
        print(f"\nProcessing {species_name}...")

        metrics = Metrics(device="cpu")

        post_processing_function = Processor(
            mean=STATS[species_name]["mean"],
            std=STATS[species_name]["std"],
        )

        data_dir = eval_base_path / species_name
        dataset = AudioTonalDataset(
            data_dir,
            data_dir,
            annotation_extension="ann",
            time_patch_frames=250,
            freq_patch_frames=250,
            post_processing_function=post_processing_function,
        )

        for i in tqdm(range(len(dataset))):
            # for i in range(len(dataset)):
            spec, ann = dataset[i]

            spec = np.flip(spec, axis=0)
            ann = np.flip(ann, axis=0)

            spec, ann = spec.copy(), ann.copy()

            ann_tensor = torch.from_numpy(ann)
            ann_tensor = ann_tensor.unsqueeze(0).unsqueeze(0).float()

            spec_tensor = torch.from_numpy(spec)
            spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(0).float()

            with torch.no_grad():
                spec_tensor = spec_tensor.to(device)
                out_tensor = model(spec_tensor)[0]

            out_tensor = out_tensor[:, 0:1]
            out_tensor = torch.sigmoid(out_tensor)
            out_tensor = out_tensor.cpu()

            metrics.update(out_tensor > SETTINGS["threshold"], ann_tensor)

        scores = metrics.compute()
        scores["species"] = species_name
        results.append(scores)

    except Exception as e:
        print(e)

# Write to CSV
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["species"] + list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults written to {output_path}")
