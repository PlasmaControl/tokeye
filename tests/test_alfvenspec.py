from __future__ import annotations

import csv

import numpy as np
import torch
import torch.nn as nn

from tokeye.alfvenspec import detect, detect_windowed, write_detections_csv


class _StubRCNN(nn.Module):
    """Returns canned torchvision-style detections; records its input.

    Detection ``i`` has a box and a mask over rows ``i:i+2``, cols ``0:2`` of
    whatever image it is given (masks follow the input shape, like the real
    model's post-processing).
    """

    def __init__(self, n: int = 3):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.seen = None
        self.n = n

    def forward(self, images):
        self.seen = images
        _, height, width = images[0].shape
        masks = torch.zeros(self.n, 1, height, width)
        for i in range(self.n):
            masks[i, 0, i : i + 2, 0:2] = 1.0
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]] * self.n),
                "labels": torch.ones(self.n, dtype=torch.int64),
                "scores": torch.tensor([0.9, 0.6, 0.2][: self.n]),
                "masks": masks,
            }
        ]


def test_detect_filters_by_score_and_returns_numpy():
    model = _StubRCNN()
    spectrogram = np.random.default_rng(0).normal(size=(8, 6)).astype(np.float32)

    result = detect(spectrogram, model, score_min=0.5)

    assert isinstance(result["boxes"], np.ndarray)
    assert result["boxes"].shape == (2, 4)  # score 0.2 filtered out
    assert result["scores"].shape == (2,)
    assert result["labels"].shape == (2,)
    assert result["masks"].shape == (2, 8, 6)  # channel dim squeezed


def test_detect_feeds_single_channel_standardized_image():
    model = _StubRCNN()
    spectrogram = (np.random.default_rng(1).normal(size=(8, 6)) * 5 + 50).astype(
        np.float32
    )

    detect(spectrogram, model)

    (img,) = model.seen
    assert img.shape == (1, 8, 6)
    assert abs(float(img.mean())) < 1e-4  # standardized per-sample
    # torch .std() is ddof=1 vs numpy's ddof=0 used for standardization
    assert abs(float(img.std()) - 1.0) < 2e-2


def test_detect_honors_explicit_mean_std():
    model = _StubRCNN()
    spectrogram = np.full((8, 6), 10.0, dtype=np.float32)

    detect(spectrogram, model, mean=8.0, std=2.0)

    (img,) = model.seen
    assert torch.allclose(img, torch.ones_like(img))


def test_detect_windowed_offsets_boxes_to_global_columns():
    model = _StubRCNN(n=1)  # one detection (score 0.9) per window call
    spectrogram = np.zeros((8, 112), dtype=np.float32)

    result = detect_windowed(spectrogram, model, window_cols=40)

    # windows: [0:40], [40:80], [80:112] -> 3 detections
    assert result["boxes"].shape == (3, 4)
    np.testing.assert_allclose(result["boxes"][:, 0], [0.0, 40.0, 80.0])
    np.testing.assert_allclose(result["boxes"][:, 2], [2.0, 42.0, 82.0])
    assert result["masks"] is None


def test_detect_windowed_folds_sliver_into_previous_window():
    model = _StubRCNN(n=1)
    spectrogram = np.zeros((8, 100), dtype=np.float32)

    result = detect_windowed(spectrogram, model, window_cols=40)

    # final 20-column sliver folds into [40:100] instead of being dropped
    assert result["boxes"].shape == (2, 4)
    np.testing.assert_allclose(result["boxes"][:, 0], [0.0, 40.0])


def test_detect_windowed_falls_back_to_single_window():
    model = _StubRCNN(n=1)
    spectrogram = np.zeros((8, 30), dtype=np.float32)

    result = detect_windowed(spectrogram, model, window_cols=40)

    assert result["boxes"].shape == (1, 4)
    assert result["masks"] is not None  # single window keeps masks


def test_instance_map_single_window_most_confident_wins():
    model = _StubRCNN()  # scores 0.9, 0.6 kept; 0.2 filtered
    spectrogram = np.zeros((8, 30), dtype=np.float32)

    result = detect_windowed(spectrogram, model, window_cols=40)

    instance_map = result["instance_map"]
    assert instance_map.shape == (8, 30)
    assert instance_map.dtype == np.int32
    assert instance_map[0, 0] == 1  # detection 0 only
    assert instance_map[1, 0] == 1  # overlap: score 0.9 beats 0.6
    assert instance_map[2, 0] == 2  # detection 1 only
    assert instance_map[3, 0] == 0  # detection 2 was filtered out
    assert set(np.unique(instance_map)) == {0, 1, 2}


def test_instance_map_windowed_uses_global_ids_and_columns():
    model = _StubRCNN(n=1)
    spectrogram = np.zeros((8, 112), dtype=np.float32)

    result = detect_windowed(spectrogram, model, window_cols=40)

    instance_map = result["instance_map"]
    assert instance_map.shape == (8, 112)
    assert instance_map[0, 0] == 1
    assert instance_map[0, 40] == 2
    assert instance_map[0, 80] == 3
    assert instance_map[0, 2] == 0
    assert result["masks"] is None  # soft masks are single-window only


def test_write_detections_csv(tmp_path):
    out = tmp_path / "ae_detections.csv"
    detections = {
        "boxes": np.array([[1.0, 2.0, 3.0, 4.0]]),
        "labels": np.array([1]),
        "scores": np.array([0.9]),
        "masks": np.zeros((1, 8, 6)),
    }
    write_detections_csv(out, [("shot1.npy", detections)])

    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["input"] == "shot1.npy"
    assert rows[0]["detection"] == "0"
    assert [rows[0][k] for k in ("x1", "y1", "x2", "y2")] == [
        "1.0",
        "2.0",
        "3.0",
        "4.0",
    ]
    assert rows[0]["score"] == "0.9"
