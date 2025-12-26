import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from tokeye.extra.eval.silbidopy.eval import Metrics

device = "cuda" if torch.cuda.is_available() else "cpu"

# Settings
SETTINGS = {"threshold": 0.5}

# Statistics
MEAN = 17.84620821169868
STD = 25.016818830630463

# Paths
root_path = Path("/scratch/gpfs/nc1514/TokEye")
data_path = Path("/scratch/gpfs/nc1514/TokEye/data/eval/TJII2021")
model_path = root_path / "model" / "big_mode_v1-5.pt"
output_path = root_path / "data" / "eval" / "results" / "TJII2021.csv"

# Load model
model = torch.load(
    model_path,
    weights_only=False,
    map_location=device,
)
model.eval()
print("Model loaded")

# Run evaluation
shotns = [name.stem.split("_")[1] for name in data_path.glob("input/*.png")]
metrics = Metrics(device="cpu")

for shotn in tqdm(shotns):
    try:
        input_path = data_path / "input" / f"spectrogram_{shotn}.png"
        gt_path = data_path / "gt" / f"spectrogram_{shotn}.png"

        spec = Image.open(input_path).convert("L")
        ann = Image.open(gt_path).convert("L")

        spec, ann = np.array(spec), np.array(ann)
        spec, ann = np.flip(spec, axis=0), np.flip(ann, axis=0)
        spec, ann = spec.copy(), ann.copy()

        spec = (spec - MEAN) / STD
        ann = ann // 255.0

        ann_tensor = torch.from_numpy(ann)
        ann_tensor = ann_tensor.unsqueeze(0).unsqueeze(0).float()

        input_tensor = torch.from_numpy(spec).float()
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            spec_tensor = input_tensor.to(device)
            out_tensor = model(spec_tensor)[0]

        out_tensor = out_tensor[:, 0:1]
        out_tensor = torch.sigmoid(out_tensor)
        out_tensor = out_tensor.cpu()

        metrics.update(out_tensor > SETTINGS["threshold"], ann_tensor)
    except Exception as e:
        print(f"Error processing shot {shotn}: {e}")
        continue

scores = metrics.compute()
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(scores.keys()))
    writer.writeheader()
    writer.writerow(scores)

print(f"\n Results saved to {output_path}")
