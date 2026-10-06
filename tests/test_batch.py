from __future__ import annotations

import json

import numpy as np
import pytest
import torch.nn as nn

from tokeye import SpectrogramConfig, batch
from tokeye.result import Segmentation


@pytest.fixture
def stub_model(monkeypatch):
    """A cheap Conv2d(1, 2, 1) stands in for the real (heavy) TokEye model."""
    model = nn.Conv2d(1, 2, kernel_size=1)
    monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: model)
    return model


@pytest.fixture
def signal_npy(tmp_path):
    path = tmp_path / "signal.npy"
    sig = np.random.default_rng(0).normal(size=8192).astype(np.float32)
    np.save(path, sig)
    return path


@pytest.fixture
def spectrogram_npy(tmp_path):
    path = tmp_path / "spectrogram.npy"
    spec = np.random.default_rng(1).normal(size=(64, 32)).astype(np.float32)
    np.save(path, spec)
    return path


class TestCollectInputs:
    def test_single_file(self, signal_npy):
        assert batch.collect_inputs([str(signal_npy)]) == [signal_npy]

    def test_directory_finds_signal_files_sorted(self, tmp_path):
        (tmp_path / "b.npy").touch()
        (tmp_path / "a.npy").touch()
        (tmp_path / "c.wav").touch()
        (tmp_path / "ignore.txt").touch()
        (tmp_path / "notes.csv").touch()
        (tmp_path / "subdir.npy").mkdir()

        result = batch.collect_inputs([str(tmp_path)])

        assert result == [tmp_path / "a.npy", tmp_path / "b.npy", tmp_path / "c.wav"]

    def test_glob_pattern(self, tmp_path):
        (tmp_path / "x1.npy").touch()
        (tmp_path / "x2.npy").touch()

        result = batch.collect_inputs([str(tmp_path / "x*.npy")])

        assert result == [tmp_path / "x1.npy", tmp_path / "x2.npy"]

    def test_mixed_inputs_dedup_preserving_order(self, tmp_path):
        (tmp_path / "a.npy").touch()
        (tmp_path / "b.npy").touch()

        # The directory glob would also match a.npy; it should appear once,
        # in its first-seen position.
        result = batch.collect_inputs([str(tmp_path / "a.npy"), str(tmp_path)])

        assert result == [tmp_path / "a.npy", tmp_path / "b.npy"]

    def test_empty_result_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="No input files found"):
            batch.collect_inputs([str(tmp_path / "does_not_exist_*.npy")])


class TestLoadInput:
    def test_1d_signal_becomes_spectrogram(self, signal_npy):
        spec = batch.load_input(signal_npy, {"n_fft": 256, "hop": 64})
        assert spec.ndim == 2

    def test_2d_spectrogram_passes_through(self, spectrogram_npy):
        spec = batch.load_input(spectrogram_npy, {})
        assert spec.shape == (64, 32)
        assert np.issubdtype(spec.dtype, np.floating)

    def test_3d_array_raises_value_error(self, tmp_path):
        path = tmp_path / "bad.npy"
        np.save(path, np.zeros((2, 3, 4)))

        with pytest.raises(ValueError, match="ndim=3"):
            batch.load_input(path, {})

    def test_log_applies_log1p_to_2d_input(self, tmp_path):
        arr = np.random.default_rng(0).random((16, 16))
        path = tmp_path / "linear.npy"
        np.save(path, arr)

        spec = batch.load_input(path, {}, log=True)

        np.testing.assert_allclose(spec, np.log1p(arr))

    def test_log_rejects_negative_2d_input(self, tmp_path):
        path = tmp_path / "db_scaled.npy"
        np.save(path, np.full((8, 8), -30.0))

        with pytest.raises(ValueError, match="negative"):
            batch.load_input(path, {}, log=True)


class TestLoadSpectrogram:
    def test_fs_from_filename(self, tmp_path):
        path = tmp_path / "shot_sr2000.npy"
        np.save(path, np.random.default_rng(0).normal(size=4096))

        spec = batch.load_spectrogram(path, SpectrogramConfig(n_fft=256, hop=64))

        assert spec.kind == "stft"
        assert spec.fs == 2000.0

    def test_explicit_fs_wins(self, tmp_path):
        path = tmp_path / "shot_sr2000.npy"
        np.save(path, np.random.default_rng(0).normal(size=4096))

        spec = batch.load_spectrogram(path, {"n_fft": 256, "hop": 64}, fs=500.0)

        assert spec.fs == 500.0


class TestRunBatch:
    def test_on_1d_signal_writes_mask_and_preview(
        self, stub_model, signal_npy, tmp_path
    ):
        out_dir = tmp_path / "out"
        failures = batch.run_batch(
            [str(signal_npy)],
            out_dir=out_dir,
            config=SpectrogramConfig(n_fft=256, hop=64),
        )

        assert failures == 0
        mask_path = out_dir / "signal_mask.npy"
        preview_path = out_dir / "signal_preview.png"
        assert mask_path.exists()
        assert preview_path.exists()
        assert preview_path.stat().st_size > 0

        mask = np.load(mask_path)
        assert mask.ndim == 3
        assert mask.shape[0] == 2
        assert mask.dtype == np.float32
        assert np.all(mask >= 0.0) and np.all(mask <= 1.0)

    def test_on_2d_spectrogram_writes_mask_and_preview(
        self, stub_model, spectrogram_npy, tmp_path
    ):
        out_dir = tmp_path / "out"
        failures = batch.run_batch([str(spectrogram_npy)], out_dir=out_dir)

        assert failures == 0
        mask = np.load(out_dir / "spectrogram_mask.npy")
        assert mask.shape == (2, 64, 32)
        assert mask.dtype == np.float32
        assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
        assert (out_dir / "spectrogram_preview.png").exists()

    def test_save_png_false_skips_preview(self, stub_model, spectrogram_npy, tmp_path):
        out_dir = tmp_path / "out"
        failures = batch.run_batch(
            [str(spectrogram_npy)], out_dir=out_dir, save_png=False
        )

        assert failures == 0
        assert (out_dir / "spectrogram_mask.npy").exists()
        assert not (out_dir / "spectrogram_preview.png").exists()

    def test_one_bad_file_among_good_ones_counts_as_failure(
        self, stub_model, spectrogram_npy, tmp_path
    ):
        bad_path = tmp_path / "bad.npy"
        np.save(bad_path, np.zeros((2, 3, 4)))

        out_dir = tmp_path / "out"
        failures = batch.run_batch(
            [str(spectrogram_npy), str(bad_path)], out_dir=out_dir
        )

        assert failures == 1
        assert (out_dir / "spectrogram_mask.npy").exists()
        assert not (out_dir / "bad_mask.npy").exists()

    def test_loads_model_once_via_hub(self, tmp_path, monkeypatch):
        """Multiple input files -> exactly one load_model call (per run, not
        per file)."""
        model = nn.Conv2d(1, 2, kernel_size=1)
        calls = []

        def fake_load_model(source, device):
            calls.append((source, device))
            return model

        monkeypatch.setattr("tokeye.hub.load_model", fake_load_model)

        rng = np.random.default_rng(2)
        for name in ("first.npy", "second.npy"):
            np.save(tmp_path / name, rng.normal(size=(64, 32)).astype(np.float32))

        out_dir = tmp_path / "out"
        failures = batch.run_batch(
            [str(tmp_path / "first.npy"), str(tmp_path / "second.npy")],
            out_dir=out_dir,
            device="cpu",
        )

        assert failures == 0
        assert (out_dir / "first_mask.npy").exists()
        assert (out_dir / "second_mask.npy").exists()
        assert calls == [(batch.hub.DEFAULT_MODEL, "cpu")]

    def test_params_json_records_provenance(self, stub_model, signal_npy, tmp_path):
        out_dir = tmp_path / "out"
        cfg = SpectrogramConfig(n_fft=256, hop=64)
        batch.run_batch([str(signal_npy)], out_dir=out_dir, config=cfg, fs=1000.0)

        params = json.loads((out_dir / "signal_params.json").read_text("utf-8"))

        assert params["model"] == batch.hub.DEFAULT_MODEL
        assert params["device"] == "cpu"
        assert params["kind"] == "stft"
        assert params["fs"] == 1000.0
        assert params["config"] == cfg.to_dict()
        assert params["channels"] == ["coherent", "transient"]
        assert params["output"] == "signal_mask.npy"
        assert params["mask_shape"][0] == 2
        assert {"tokeye_version", "input", "threshold", "created_utc"} <= set(params)

    def test_npz_format_writes_a_loadable_bundle(
        self, stub_model, signal_npy, tmp_path
    ):
        out_dir = tmp_path / "out"
        failures = batch.run_batch(
            [str(signal_npy)],
            out_dir=out_dir,
            config=SpectrogramConfig(n_fft=256, hop=64),
            fs=1000.0,
            fmt="npz",
        )

        assert failures == 0
        assert not (out_dir / "signal_mask.npy").exists()
        seg = Segmentation.load(out_dir / "signal_tokeye.npz")
        assert seg.mask.shape[0] == 2
        assert seg.fs == 1000.0
        assert seg.spectrogram.config == SpectrogramConfig(n_fft=256, hop=64)

    def test_bad_format_raises_before_loading(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tokeye.hub.load_model", pytest.fail)
        with pytest.raises(ValueError, match="fmt must be one of"):
            batch.run_batch([str(tmp_path)], out_dir=tmp_path, fmt="csv")

    def test_instance_model_is_rejected_before_loading(
        self, spectrogram_npy, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("tokeye.hub.load_model", pytest.fail)
        with pytest.raises(ValueError, match="alfvenspec"):
            batch.run_batch(
                [str(spectrogram_npy)], model="ae_tf_maskrcnn", out_dir=tmp_path
            )


class TestDeprecations:
    def test_stft_kwargs_warns_and_still_works(self, stub_model, signal_npy, tmp_path):
        out_dir = tmp_path / "out"
        with pytest.warns(DeprecationWarning, match="config="):
            failures = batch.run_batch(
                [str(signal_npy)], out_dir=out_dir, stft_kwargs={"n_fft": 256}
            )

        assert failures == 0
        params = json.loads((out_dir / "signal_params.json").read_text("utf-8"))
        assert params["config"]["n_fft"] == 256

    def test_log_kwarg_warns(self, stub_model, spectrogram_npy, tmp_path):
        with pytest.warns(DeprecationWarning):
            batch.run_batch([str(spectrogram_npy)], out_dir=tmp_path, log=False)

    def test_config_and_stft_kwargs_together_raise(self, tmp_path):
        with (
            pytest.warns(DeprecationWarning),
            pytest.raises(ValueError, match="not both"),
        ):
            batch.run_batch(
                [str(tmp_path)],
                config=SpectrogramConfig(),
                stft_kwargs={"n_fft": 256},
            )

    def test_process_file_dict_config_warns(
        self, stub_model, spectrogram_npy, tmp_path
    ):
        with pytest.warns(DeprecationWarning, match="SpectrogramConfig"):
            batch.process_file(spectrogram_npy, stub_model, {"hop": 64}, tmp_path)

        assert (tmp_path / "spectrogram_mask.npy").exists()

    def test_process_file_stft_kwargs_warns_and_matches_config(
        self, stub_model, signal_npy, tmp_path
    ):
        by_config, by_kwargs = tmp_path / "config", tmp_path / "kwargs"
        by_config.mkdir()
        by_kwargs.mkdir()
        batch.process_file(
            signal_npy,
            stub_model,
            config=SpectrogramConfig(n_fft=256, hop=64),
            out_dir=by_config,
            fs=1000.0,
        )

        # The 0.12.0 keyword spelling: stft_kwargs instead of config.
        with pytest.warns(DeprecationWarning, match="SpectrogramConfig") as record:
            batch.process_file(
                signal_npy,
                stub_model,
                stft_kwargs={"n_fft": 256, "hop": 64, "fs": 1000.0},
                out_dir=by_kwargs,
            )

        deprecations = [w for w in record if w.category is DeprecationWarning]
        assert len(deprecations) == 1
        assert deprecations[0].filename == __file__  # points at the caller
        np.testing.assert_array_equal(
            np.load(by_kwargs / "signal_mask.npy"),
            np.load(by_config / "signal_mask.npy"),
        )
        kwargs_params = json.loads((by_kwargs / "signal_params.json").read_text())
        config_params = json.loads((by_config / "signal_params.json").read_text())
        assert kwargs_params["fs"] == config_params["fs"] == 1000.0
        assert kwargs_params["config"] == config_params["config"]

    def test_process_file_config_and_stft_kwargs_together_raise(
        self, stub_model, signal_npy, tmp_path
    ):
        with pytest.raises(TypeError, match="pass either config or stft_kwargs"):
            batch.process_file(
                signal_npy,
                stub_model,
                SpectrogramConfig(),
                tmp_path,
                stft_kwargs={"n_fft": 256},
            )

        assert not list(tmp_path.glob("signal_*"))

    def test_process_file_still_needs_out_dir(self, stub_model, signal_npy):
        with pytest.raises(TypeError, match="out_dir"):
            batch.process_file(signal_npy, stub_model, stft_kwargs={"n_fft": 256})
