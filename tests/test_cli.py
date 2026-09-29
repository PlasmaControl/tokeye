from __future__ import annotations

import httpx
import numpy as np
import pytest
import torch.nn as nn
from huggingface_hub.errors import RepositoryNotFoundError

import tokeye
from tokeye import SpectrogramConfig
from tokeye.cli import build_parser, main
from tokeye.cli._options import config_from_args


def _repository_not_found_error() -> RepositoryNotFoundError:
    """Build a real RepositoryNotFoundError the way huggingface_hub does.

    ``HfHubHTTPError`` (its base class) requires a ``response`` kwarg with no
    default, so a bare ``RepositoryNotFoundError("msg")`` raises ``TypeError``.
    """
    response = httpx.Response(
        404, request=httpx.Request("GET", "https://huggingface.co/does/not/exist")
    )
    return RepositoryNotFoundError("Repository Not Found", response=response)


@pytest.fixture
def stub_model(monkeypatch):
    """A cheap Conv2d(1, 2, 1) stands in for the real model."""
    model = nn.Conv2d(1, 2, kernel_size=1)
    monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: model)
    return model


@pytest.fixture
def spectrogram_npy(tmp_path):
    path = tmp_path / "input.npy"
    np.save(path, np.random.default_rng(0).random((64, 32)).astype(np.float32))
    return path


class TestBuildParser:
    def test_run_subcommand_parses_options(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "run",
                "a.npy",
                "--hop",
                "64",
                "--no-png",
                "--model",
                "model/x.pt",
                "--log",
                "--no-clip-dc",
                "--format",
                "npz",
                "--fs",
                "1000",
            ]
        )

        assert args.inputs == ["a.npy"]
        assert args.hop == 64
        assert args.png is False
        assert args.model == "model/x.pt"
        assert args.log is True
        assert args.clip_dc is False
        assert args.fmt == "npz"
        assert args.fs == 1000.0

    def test_run_subcommand_defaults(self):
        parser = build_parser()

        args = parser.parse_args(["run", "a.npy"])

        assert args.model == "big_tf_unet"
        assert args.output_dir == "tokeye_output"
        # spectrogram flags default to None = "use SpectrogramConfig's value"
        for name in ("n_fft", "hop", "window", "clip_dc", "clip_low", "clip_high"):
            assert getattr(args, name) is None
        assert args.log is None
        assert args.keep_dc is False
        assert args.threshold == 0.5
        assert args.png is True
        assert args.device == "auto"
        assert args.fs is None
        assert args.fmt == "npy"

    def test_config_from_args_defaults_to_the_training_recipe(self):
        args = build_parser().parse_args(["run", "a.npy"])

        assert config_from_args(args) == SpectrogramConfig()
        assert config_from_args(args).hop == 128

    def test_config_from_args_applies_overrides(self):
        args = build_parser().parse_args(
            [
                "run",
                "a.npy",
                "--n-fft",
                "512",
                "--clip-high",
                "98",
                "--window",
                "hamming",
            ]
        )

        assert config_from_args(args) == SpectrogramConfig(
            n_fft=512, clip_high=98.0, window="hamming"
        )

    def test_keep_dc_is_a_deprecated_alias(self, capsys):
        args = build_parser().parse_args(["run", "a.npy", "--keep-dc"])

        assert config_from_args(args).clip_dc is False
        assert "--keep-dc is deprecated" in capsys.readouterr().err

    def test_help_lists_generated_flags_with_defaults(self, capsys):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--help"])

        out = capsys.readouterr().out
        assert "--n-fft" in out
        assert "(default: 1024)" in out
        assert "(default: 128)" in out
        assert "--no-clip-dc" in out
        assert "--keep-dc" not in out  # hidden alias

    @pytest.mark.parametrize("value", ["-1", "0", "nan", "abc"])
    def test_bad_fs_is_a_usage_error(self, value):
        with pytest.raises(SystemExit) as exc_info:
            build_parser().parse_args(["run", "a.npy", "--fs", value])
        assert exc_info.value.code == 2

    def test_app_subcommand_defaults(self):
        parser = build_parser()

        args = parser.parse_args(["app"])

        assert args.port == 7860
        assert args.share is False
        assert args.open_browser is False


class TestMain:
    def test_version_prints_package_version(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])

        assert exc_info.value.code == 0
        assert capsys.readouterr().out.strip() == f"tokeye {tokeye.__version__}"

    def test_no_subcommand_returns_two(self, capsys):
        assert main([]) == 2
        assert "usage: tokeye" in capsys.readouterr().err

    def test_run_with_nonexistent_input_returns_two_clean_error(self, capsys):
        exit_code = main(["run", "does_not_exist_anywhere_xyz.npy"])

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "does_not_exist_anywhere_xyz.npy" in err
        assert "tokeye example" in err

    def test_run_with_missing_model_path_returns_two_clean_error(
        self, spectrogram_npy, capsys
    ):
        exit_code = main(["run", str(spectrogram_npy), "--model", "nope/missing.pt"])

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "nope/missing.pt" in err
        assert "Traceback" not in err

    def test_run_success_returns_zero_and_writes_outputs(
        self, stub_model, spectrogram_npy, tmp_path
    ):
        out_dir = tmp_path / "out"

        exit_code = main(["run", str(spectrogram_npy), "--output-dir", str(out_dir)])

        assert exit_code == 0
        assert (out_dir / "input_mask.npy").exists()
        assert (out_dir / "input_preview.png").exists()
        assert (out_dir / "input_params.json").exists()

    def test_run_failures_return_one_not_a_count(
        self, stub_model, spectrogram_npy, tmp_path
    ):
        for name in ("bad1.npy", "bad2.npy"):
            np.save(tmp_path / name, np.zeros((2, 3, 4)))

        exit_code = main(
            [
                "run",
                str(spectrogram_npy),
                str(tmp_path / "bad1.npy"),
                str(tmp_path / "bad2.npy"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )

        assert exit_code == 1

    def test_run_npz_format(self, stub_model, spectrogram_npy, tmp_path):
        out_dir = tmp_path / "out"

        exit_code = main(
            [
                "run",
                str(spectrogram_npy),
                "--output-dir",
                str(out_dir),
                "--format",
                "npz",
            ]
        )

        assert exit_code == 0
        assert (out_dir / "input_tokeye.npz").exists()
        assert not (out_dir / "input_mask.npy").exists()

    def test_run_rejects_an_instance_model(self, spectrogram_npy, monkeypatch, capsys):
        monkeypatch.setattr("tokeye.hub.load_model", pytest.fail)

        exit_code = main(["run", str(spectrogram_npy), "--model", "ae_tf_maskrcnn"])

        assert exit_code == 2
        assert "tokeye alfvenspec" in capsys.readouterr().err

    def test_run_bad_config_is_a_usage_error(self, spectrogram_npy, capsys):
        exit_code = main(
            ["run", str(spectrogram_npy), "--clip-low", "99", "--clip-high", "1"]
        )

        assert exit_code == 2
        assert "clip_low" in capsys.readouterr().err

    def test_keep_dc_with_clip_dc_is_a_usage_error(self, spectrogram_npy, capsys):
        exit_code = main(["run", str(spectrogram_npy), "--keep-dc", "--clip-dc"])

        assert exit_code == 2
        assert "contradict" in capsys.readouterr().err

    def test_ctrl_c_returns_130(self, spectrogram_npy, monkeypatch, capsys):
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr("tokeye.batch.run_batch", interrupted)

        assert main(["run", str(spectrogram_npy)]) == 130
        assert "interrupted" in capsys.readouterr().err

    def test_example_writes_file_and_returns_zero(self, tmp_path, capsys):
        out_path = tmp_path / "example.npy"

        exit_code = main(
            [
                "example",
                "--output",
                str(out_path),
                "--duration",
                "0.01",
                "--fs",
                "10000",
            ]
        )

        assert exit_code == 0
        assert out_path.exists()
        sig = np.load(out_path)
        assert sig.shape[0] == 100

    def test_example_extensionless_output_prints_actual_file(self, tmp_path, capsys):
        """np.save appends .npy; the printed path must be the file that exists."""
        exit_code = main(
            [
                "example",
                "--output",
                str(tmp_path / "demo"),
                "--duration",
                "0.01",
                "--fs",
                "10000",
            ]
        )

        assert exit_code == 0
        printed = capsys.readouterr().out.strip()
        assert printed == str(tmp_path / "demo.npy")
        assert (tmp_path / "demo.npy").exists()

    def test_download_unknown_model_returns_two_clean_error(self, capsys):
        exit_code = main(["download", "not_a_real_model_name"])

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "not_a_real_model_name" in err

    def test_download_hub_error_returns_two_clean_error(self, monkeypatch, capsys):
        def fake_download_model(name, repo_id=None):
            raise _repository_not_found_error()

        monkeypatch.setattr("tokeye.hub.download_model", fake_download_model)

        exit_code = main(["download"])

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "nc1/big_tf_unet" in err
        assert "TOKEYE_HF_REPO" in err
        assert "Traceback" not in err

    def test_run_hub_error_returns_two_clean_error(
        self, spectrogram_npy, monkeypatch, capsys
    ):
        input_path = spectrogram_npy

        def fake_load_model(source, device="auto"):
            raise _repository_not_found_error()

        monkeypatch.setattr("tokeye.hub.load_model", fake_load_model)

        exit_code = main(["run", str(input_path), "--device", "cpu"])

        assert exit_code == 2
        err = capsys.readouterr().err
        assert "nc1/big_tf_unet" in err
        assert "TOKEYE_HF_REPO" in err
        assert "Traceback" not in err

    def test_app_subcommand_delegates_to_app_main(self, monkeypatch):
        calls = {}

        def fake_app_main(port, share, open_browser):
            calls["port"] = port
            calls["share"] = share
            calls["open_browser"] = open_browser

        monkeypatch.setattr("tokeye.app.__main__.main", fake_app_main)

        exit_code = main(["app", "--port", "1234", "--share"])

        assert exit_code == 0
        assert calls == {"port": 1234, "share": True, "open_browser": False}


@pytest.mark.parametrize(
    "argv",
    [["--help"], ["run", "--help"], ["download", "--help"], ["example", "--help"]],
)
def test_help_does_not_crash(argv, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 0
