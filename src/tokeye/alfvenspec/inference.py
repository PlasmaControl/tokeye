"""Inference for the R-CNN detection contract (list of images -> detections).

Kept separate from :mod:`tokeye.inference`, whose ``model_infer`` assumes the
segmentation contract ``(B, 1, H, W) -> (B, 2, H, W)`` + sigmoid.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from pathlib import Path

    import torch.nn as nn


def detect(
    spectrogram: np.ndarray,
    model: nn.Module,
    *,
    score_min: float = 0.5,
    mean: float | None = None,
    std: float | None = None,
) -> dict[str, np.ndarray]:
    """Run an R-CNN detection model on one ``(H, W)`` spectrogram.

    The image is standardized with ``mean``/``std`` (per-sample statistics
    when omitted; training used dataset-level stats, so pass them if known)
    and fed as a single-channel image — the model's ``GeneralizedRCNNTransform``
    broadcasts it across its 3-element mean/std. Returns numpy arrays:
    ``boxes`` (N, 4) xyxy, ``labels`` (N,), ``scores`` (N,), ``masks`` (N, H, W),
    filtered to ``scores >= score_min``.
    """
    arr = np.asarray(spectrogram, dtype=np.float32)
    resolved_mean = float(arr.mean()) if mean is None else mean
    resolved_std = float(arr.std()) + 1e-6 if std is None else std

    device = next(model.parameters()).device
    image = torch.from_numpy((arr - resolved_mean) / resolved_std)
    image = image.unsqueeze(0).float().to(device)

    model.eval()
    with torch.no_grad():
        output = model([image])[0]

    keep = output["scores"] >= score_min
    return {
        "boxes": output["boxes"][keep].cpu().numpy(),
        "labels": output["labels"][keep].cpu().numpy(),
        "scores": output["scores"][keep].cpu().numpy(),
        "masks": output["masks"][keep].squeeze(1).cpu().numpy(),
    }


DEFAULT_WINDOW_COLS = 710  # training window width; full shots must be windowed
_MIN_WINDOW_COLS = 32


def _paint_instances(
    target: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    first_id: int,
    col_offset: int,
) -> None:
    """Paint soft instance ``masks`` into the integer label map ``target``.

    Detection ``i`` is painted as ``first_id + i + 1`` wherever its mask is
    >= 0.5, starting at column ``col_offset``. Lower scores are painted first
    so the most confident detection wins where instances overlap.
    """
    width = masks.shape[-1]
    region = target[:, col_offset : col_offset + width]
    for index in np.argsort(scores, kind="stable"):
        region[masks[index] >= 0.5] = first_id + int(index) + 1


def detect_windowed(
    spectrogram: np.ndarray,
    model: nn.Module,
    *,
    window_cols: int = DEFAULT_WINDOW_COLS,
    score_min: float = 0.5,
    mean: float | None = None,
    std: float | None = None,
) -> dict[str, np.ndarray | None]:
    """Run :func:`detect` over non-overlapping column windows and merge.

    The R-CNN transform resizes inputs to at most ``max_size`` (1333) columns,
    so a full-shot spectrogram (tens of thousands of columns) gets crushed
    horizontally and yields nothing; the model was trained on
    ~:data:`DEFAULT_WINDOW_COLS`-column views. Box x-coordinates are shifted
    back to global columns; ``window_cols <= 0`` disables windowing.

    Besides ``boxes``/``labels``/``scores``, the result has ``instance_map``:
    an ``(H, W)`` int32 map of the whole input where 0 is background and
    ``i + 1`` marks detection ``i`` (its row in the detections CSV).
    ``masks`` (per-detection soft masks) is ``None`` whenever more than one
    window is used, since per-window masks have no common global shape.

    Raises
    ------
    ValueError
        If ``0 < window_cols < 32``: windows that narrow mean one model
        pass per few columns (60,000 passes for a 60,000-column shot at 1).
    """
    if 0 < window_cols < _MIN_WINDOW_COLS:
        raise ValueError(
            f"window_cols must be 0 (no windowing) or >= {_MIN_WINDOW_COLS}, "
            f"got {window_cols}"
        )
    n_rows, n_cols = spectrogram.shape
    instance_map = np.zeros((n_rows, n_cols), dtype=np.int32)
    if window_cols <= 0 or n_cols <= window_cols:
        result = detect(spectrogram, model, score_min=score_min, mean=mean, std=std)
        _paint_instances(instance_map, result["masks"], result["scores"], 0, 0)
        return {**result, "instance_map": instance_map}

    starts = list(range(0, n_cols, window_cols))
    if len(starts) > 1 and n_cols - starts[-1] < _MIN_WINDOW_COLS:
        starts.pop()  # fold a sliver of a final window into the previous one

    merged: dict[str, list[np.ndarray]] = {"boxes": [], "labels": [], "scores": []}
    n_found = 0
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else n_cols
        window = spectrogram[:, start:end]
        result = detect(window, model, score_min=score_min, mean=mean, std=std)
        result["boxes"][:, [0, 2]] += start
        _paint_instances(
            instance_map, result["masks"], result["scores"], n_found, start
        )
        n_found += len(result["scores"])
        for key in merged:
            merged[key].append(result[key])

    return {
        "boxes": np.concatenate(merged["boxes"]),
        "labels": np.concatenate(merged["labels"]),
        "scores": np.concatenate(merged["scores"]),
        "masks": None,
        "instance_map": instance_map,
    }


DETECTION_FIELDS = ("input", "detection", "x1", "y1", "x2", "y2", "label", "score")


def write_detections_csv(
    path: Path, per_input: list[tuple[str, dict[str, np.ndarray]]]
) -> None:
    """One row per detection: box corners (pixel coords), label, score."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DETECTION_FIELDS)
        writer.writeheader()
        for name, detections in per_input:
            for index, box in enumerate(detections["boxes"]):
                x1, y1, x2, y2 = (float(coord) for coord in box)
                writer.writerow(
                    {
                        "input": name,
                        "detection": index,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "label": int(detections["labels"][index]),
                        "score": float(detections["scores"][index]),
                    }
                )
