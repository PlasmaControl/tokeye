"""``tokeye alfvenspec`` per-input outputs: the deprecated ``_ae_masks.npy``,
what the CSV keeps, and a failed write failing only its own input."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from cli_helpers import _StubRCNN, one_error_line

from tokeye.cli import main

H, W = 32, 64
NOTE = (
    "note: <stem>_ae_masks.npy (per-detection soft masks) is deprecated and is "
    "no longer written from 2.0; use <stem>_ae_instances.npy"
)


def _two_masks(height: int, width: int) -> torch.Tensor:
    masks = torch.zeros(2, 1, height, width)
    masks[0, 0, 0:2, 0:2] = 0.9
    masks[1, 0, 4:8, 10:20] = 0.7
    return masks


class _TwoDetections(nn.Module):
    """Instance stub: two detections with distinct soft masks."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, images):
        _, height, width = images[0].shape
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0], [10.0, 4.0, 20.0, 8.0]]),
                "labels": torch.ones(2, dtype=torch.int64),
                "scores": torch.tensor([0.9, 0.8]),
                "masks": _two_masks(height, width),
            }
        ]


@pytest.fixture
def inputs(tmp_path):
    """Two 2D inputs with distinct stems."""
    paths = []
    for name in ("shot.npy", "shot2.npy"):
        path = tmp_path / name
        np.save(path, np.random.default_rng(len(paths)).random((H, W)))
        paths.append(path)
    return paths


def _use(monkeypatch, model: nn.Module) -> None:
    monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: model)


def _run(paths, out, *extra) -> int:
    return main(["alfvenspec", *map(str, paths), "--output-dir", str(out), *extra])


def _rows(out):
    with (out / "ae_detections.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_unwindowed_inputs_keep_the_deprecated_masks_file(
    inputs, tmp_path, monkeypatch, capsys
):
    _use(monkeypatch, _TwoDetections())
    out = tmp_path / "out"

    assert _run(inputs, out) == 0

    for path in inputs:
        masks = np.load(out / f"{path.stem}_ae_masks.npy")
        assert masks.shape == (2, H, W)
        np.testing.assert_array_equal(masks, _two_masks(H, W)[:, 0].numpy())
        assert (out / f"{path.stem}_ae_instances.npy").exists()
    captured = capsys.readouterr()
    assert captured.err.splitlines().count(NOTE) == 1
    assert captured.err.strip() == NOTE
    assert captured.out.splitlines()[-1] == str(out / "ae_detections.csv")


@pytest.mark.parametrize(
    "extra",
    [("--window-cols", "32"), ("--score-min", "0.95"), ("--no-masks",)],
    ids=["windowed", "no-detections", "no-masks"],
)
def test_no_masks_file_and_no_note(inputs, tmp_path, monkeypatch, capsys, extra):
    _use(monkeypatch, _StubRCNN())
    out = tmp_path / "out"

    assert _run(inputs[:1], out, *extra) == 0

    assert not list(out.glob("*_ae_masks.npy"))
    assert "note:" not in capsys.readouterr().err


def test_the_csv_gets_only_boxes_labels_and_scores(inputs, tmp_path, monkeypatch):
    _use(monkeypatch, _TwoDetections())
    seen = []

    def spy(path, per_input):
        seen.extend(per_input)
        path.write_text("", encoding="utf-8")

    monkeypatch.setattr("tokeye.alfvenspec.write_detections_csv", spy)

    assert _run(inputs, tmp_path / "out") == 0

    assert [name for name, _ in seen] == [str(p) for p in inputs]
    for _, detections in seen:
        assert set(detections) == {"boxes", "labels", "scores"}


def test_a_failed_write_fails_only_its_input(inputs, tmp_path, monkeypatch, capsys):
    _use(monkeypatch, _StubRCNN())
    out = tmp_path / "out"
    real_save = np.save
    first = f"{inputs[0].stem}_ae_instances.npy"

    def fake_save(file, arr, *args, **kwargs):
        if Path(file).name == first:
            raise OSError(28, "No space left on device")
        return real_save(file, arr, *args, **kwargs)

    monkeypatch.setattr("numpy.save", fake_save)

    exit_code = _run(inputs, out, "--window-cols", "32")

    assert exit_code == 1
    line = one_error_line(capsys.readouterr().err)
    assert line.startswith(f"error: failed to process {inputs[0]}: OSError: ")
    assert not (out / first).exists()
    assert (out / f"{inputs[1].stem}_ae_instances.npy").exists()
    rows = _rows(out)
    assert rows
    assert {row["input"] for row in rows} == {str(inputs[1])}
