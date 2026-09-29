"""CLI tests for the mode-analysis suite subcommands."""

from __future__ import annotations

import csv

import numpy as np
import pytest
from cli_helpers import _StubRCNN, _TransientStub

from tokeye.cli import alfvenspec as alfvenspec_cli
from tokeye.cli import build_parser, main


def _must_not_load(*args, **kwargs):
    pytest.fail("the model was loaded")


@pytest.fixture
def elm_spectrogram(tmp_path):
    """A 2D input with full-column bursts at columns 10-12 and 30-32."""
    spec = np.zeros((32, 64), dtype=np.float32)
    spec[:, 10:13] = 1.0
    spec[:, 30:33] = 1.0
    path = tmp_path / "shot.npy"
    np.save(path, spec)
    return path


def _read_csv(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_elmspec_defaults():
    args = build_parser().parse_args(["elmspec", "input.npy"])
    assert args.command == "elmspec"
    assert args.inputs == ["input.npy"]
    assert args.model == "big_tf_unet"
    assert args.output_dir == "tokeye_elms"
    assert args.threshold == 0.5
    assert args.activity_min == 0.1
    assert args.min_gap_cols == 3
    assert args.min_duration_cols == 1
    assert args.fs is None
    assert args.dt is None
    assert args.png is False


def test_alfvenspec_defaults():
    args = build_parser().parse_args(["alfvenspec", "input.npy"])
    assert args.command == "alfvenspec"
    assert args.model == "ae_tf_maskrcnn"
    assert args.output_dir == "tokeye_ae"
    assert args.score_min == 0.5
    assert args.window_cols == 710
    assert args.mean is None
    assert args.std is None
    assert args.masks is True
    assert not hasattr(args, "png")


def test_alfvenspec_window_default_matches_the_library():
    from tokeye.alfvenspec import DEFAULT_WINDOW_COLS

    assert alfvenspec_cli.DEFAULT_WINDOW_COLS == DEFAULT_WINDOW_COLS


def test_elmspec_missing_input_exits_2(tmp_path, capsys):
    exit_code = main(["elmspec", str(tmp_path / "nope_*.npy")])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "No input files found" in err
    assert "tokeye example" in err


class TestElmspec:
    @pytest.fixture(autouse=True)
    def stub(self, monkeypatch):
        model = _TransientStub()
        monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: model)

    def test_frames_only_without_fs(self, elm_spectrogram, tmp_path, capsys):
        out = tmp_path / "out"

        exit_code = main(["elmspec", str(elm_spectrogram), "--output-dir", str(out)])

        assert exit_code == 0
        rows = _read_csv(out / "elm_events.csv")
        assert [(r["start_col"], r["end_col"]) for r in rows] == [
            ("10", "12"),
            ("30", "32"),
        ]
        assert rows[0]["t_start_s"] == ""
        summary = _read_csv(out / "elm_summary.csv")
        assert summary[0]["n_events"] == "2"
        assert summary[0]["elm_freq_hz"] == ""
        assert "2 ELM event(s)" in capsys.readouterr().out

    def test_fs_on_a_spectrogram_assumes_hop_and_says_so(
        self, elm_spectrogram, tmp_path, capsys
    ):
        out = tmp_path / "out"

        exit_code = main(
            [
                "elmspec",
                str(elm_spectrogram),
                "--output-dir",
                str(out),
                "--fs",
                "1000",
                "--hop",
                "10",
            ]
        )

        assert exit_code == 0
        rows = _read_csv(out / "elm_events.csv")
        assert float(rows[0]["t_start_s"]) == pytest.approx(10 * 10 / 1000)
        assert "--hop=10" in capsys.readouterr().err

    def test_dt_wins(self, elm_spectrogram, tmp_path, capsys):
        out = tmp_path / "out"

        main(
            [
                "elmspec",
                str(elm_spectrogram),
                "--output-dir",
                str(out),
                "--fs",
                "1000",
                "--dt",
                "0.5",
            ]
        )

        rows = _read_csv(out / "elm_events.csv")
        assert float(rows[1]["t_start_s"]) == pytest.approx(15.0)
        summary = _read_csv(out / "elm_summary.csv")
        assert float(summary[0]["elm_freq_hz"]) == pytest.approx(2 / (64 * 0.5))
        assert "note:" not in capsys.readouterr().err

    def test_png_and_failures(self, elm_spectrogram, tmp_path):
        bad = tmp_path / "bad.npy"
        np.save(bad, np.zeros((2, 3, 4)))
        out = tmp_path / "out"

        exit_code = main(
            [
                "elmspec",
                str(elm_spectrogram),
                str(bad),
                "--output-dir",
                str(out),
                "--png",
            ]
        )

        assert exit_code == 1
        assert (out / "shot_elm_preview.png").exists()
        assert len(_read_csv(out / "elm_summary.csv")) == 1

    def test_rejects_an_instance_model(
        self, elm_spectrogram, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr("tokeye.hub.load_model", _must_not_load)
        out = tmp_path / "out"

        exit_code = main(
            [
                "elmspec",
                str(elm_spectrogram),
                "--model",
                "ae_tf_maskrcnn",
                "--output-dir",
                str(out),
            ]
        )

        assert exit_code == 2
        assert "tokeye alfvenspec" in capsys.readouterr().err
        assert not out.exists()


class TestAlfvenspec:
    @pytest.fixture(autouse=True)
    def stub(self, monkeypatch):
        model = _StubRCNN()
        monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: model)

    def test_writes_csv_and_instance_map(self, elm_spectrogram, tmp_path):
        out = tmp_path / "out"

        exit_code = main(["alfvenspec", str(elm_spectrogram), "--output-dir", str(out)])

        assert exit_code == 0
        rows = _read_csv(out / "ae_detections.csv")
        assert len(rows) == 1
        instances = np.load(out / "shot_ae_instances.npy")
        assert instances.shape == (32, 64)
        assert instances.dtype == np.int32
        assert instances[0, 0] == 1

    def test_windowed_runs_still_write_the_instance_map(
        self, elm_spectrogram, tmp_path
    ):
        out = tmp_path / "out"

        main(
            [
                "alfvenspec",
                str(elm_spectrogram),
                "--output-dir",
                str(out),
                "--window-cols",
                "32",
            ]
        )

        instances = np.load(out / "shot_ae_instances.npy")
        assert instances[0, 0] == 1
        assert instances[0, 32] == 2
        assert len(_read_csv(out / "ae_detections.csv")) == 2

    def test_no_masks_skips_the_map(self, elm_spectrogram, tmp_path):
        out = tmp_path / "out"

        main(
            ["alfvenspec", str(elm_spectrogram), "--output-dir", str(out), "--no-masks"]
        )

        assert not (out / "shot_ae_instances.npy").exists()
        assert (out / "ae_detections.csv").exists()

    def test_rejects_a_segmentation_model(
        self, elm_spectrogram, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr("tokeye.hub.load_model", _must_not_load)
        out = tmp_path / "out"

        exit_code = main(
            [
                "alfvenspec",
                str(elm_spectrogram),
                "--model",
                "big_tf_unet",
                "--output-dir",
                str(out),
            ]
        )

        assert exit_code == 2
        assert "tokeye run" in capsys.readouterr().err
        assert not out.exists()
