"""``tokeye info`` prints as it goes; ``tokeye example`` keeps the given name.

No gradio is needed, so this file runs in the dependency-floor job too.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import pytest

import tokeye
from tokeye.cli import info as info_cli
from tokeye.cli import main
from tokeye.cli.example import _shell_quote
from tokeye.examples import (
    DEFAULT_FS,
    example_filename,
    format_rate,
    write_example_signal,
)
from tokeye.io import fs_from_name, load_signal

TORCH_SKIPPED = "skipped (torch failed to import)"


@pytest.fixture
def uncached(monkeypatch):
    monkeypatch.setattr("tokeye.hub.cached_path", lambda name: None)


class TestInfo:
    def test_every_row_prints_and_nothing_failing_exits_zero(self, uncached, capsys):
        exit_code = main(["info"])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "failed" not in out
        assert "skipped" not in out
        assert out.splitlines()[0].startswith("tokeye   ")
        assert "\nmodels\n" in out

    def test_a_cached_model_shows_its_path(self, monkeypatch, capsys):
        where = Path("cache") / "big_tf_unet.pt"
        monkeypatch.setattr(
            "tokeye.hub.cached_path",
            lambda name: where if name == "big_tf_unet" else None,
        )

        exit_code = main(["info"])

        lines = capsys.readouterr().out.splitlines()
        (line,) = [x for x in lines if x.strip().startswith("big_tf_unet")]
        assert str(where) in line
        assert exit_code == 0

    def test_a_torch_that_does_not_import_still_prints(
        self, uncached, monkeypatch, capsys
    ):
        monkeypatch.setitem(sys.modules, "torch", None)

        exit_code = main(["info"])

        out = capsys.readouterr().out
        rows = {line.split()[0]: line for line in out.splitlines() if line.strip()}
        assert out.startswith(f"tokeye    {tokeye.__version__}")
        assert rows["python"] and rows["platform"]
        assert "failed:" in rows["torch"]
        for key in ("cuda", "mps", "device"):
            assert TORCH_SKIPPED in rows[key]
        models = out.split("\nmodels\n")[1]
        assert models == f"  {TORCH_SKIPPED}\n"
        assert exit_code == 1

    def test_a_hub_that_does_not_import_is_named(self, uncached, monkeypatch, capsys):
        monkeypatch.delattr(tokeye, "hub", raising=False)
        monkeypatch.setitem(sys.modules, "tokeye.hub", None)

        exit_code = main(["info"])

        out = capsys.readouterr().out
        rows = {line.split()[0]: line for line in out.splitlines() if line.strip()}
        assert "failed:" in rows["mps"]
        assert "failed:" in rows["device"]
        assert "tokeye.hub" in rows["mps"]
        assert out.split("\nmodels\n")[1] == "  skipped (tokeye.hub failed to import)\n"
        assert "torch" in rows and "failed" not in rows["torch"]
        assert exit_code == 1

    def test_a_failing_cuda_probe_is_one_failed_row(
        self, uncached, monkeypatch, capsys
    ):
        def broken(torch):
            raise RuntimeError("driver\nmismatch")

        monkeypatch.setattr(info_cli, "_cuda_line", broken)

        exit_code = main(["info"])

        out = capsys.readouterr().out
        (cuda,) = [x for x in out.splitlines() if x.startswith("cuda")]
        assert cuda.endswith("failed: driver mismatch")
        assert "\ndevice" in out
        assert exit_code == 1

    def test_a_failing_cached_path_is_that_models_failed_line(
        self, monkeypatch, capsys
    ):
        def broken(name):
            raise OSError("disk")

        monkeypatch.setattr("tokeye.hub.cached_path", broken)

        exit_code = main(["info"])

        out = capsys.readouterr().out
        assert "failed: disk" in out
        assert exit_code == 1

    def test_a_missing_extra_names_the_module_and_is_not_a_failure(
        self, uncached, monkeypatch, capsys
    ):
        monkeypatch.setattr(info_cli, "_installed", lambda module: module != "gradio")

        exit_code = main(["info"])

        out = capsys.readouterr().out
        assert 'app: missing gradio (pip install "tokeye[app]")' in out
        assert exit_code == 0

    def test_every_missing_module_is_named_in_order(self, monkeypatch):
        monkeypatch.setattr(info_cli, "_installed", lambda module: False)

        line = info_cli._extras_line()

        assert 'app: missing gradio, soundfile (pip install "tokeye[app]")' in line
        assert 'ae: missing torchvision (pip install "tokeye[ae]")' in line

    def test_a_failing_hf_cache_import_is_one_failed_row(
        self, uncached, monkeypatch, capsys
    ):
        monkeypatch.setitem(sys.modules, "huggingface_hub.constants", None)

        exit_code = main(["info"])

        out = capsys.readouterr().out
        (row,) = [x for x in out.splitlines() if x.startswith("hf cache")]
        assert "failed:" in row
        assert exit_code == 1


class TestExample:
    def test_a_fractional_rate_in_the_name_is_kept(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)

        exit_code = main(
            [
                "example",
                "--output",
                "demo_sr1234.5",
                "--fs",
                "1234.5",
                "--duration",
                "0.01",
            ]
        )

        assert exit_code == 0
        path = tmp_path / "demo_sr1234.5.npy"
        assert path.exists()
        assert load_signal(path)[1] == 1234.5
        captured = capsys.readouterr()
        assert captured.out.strip() == "demo_sr1234.5.npy"
        assert "--fs" not in captured.err

    def test_a_name_without_a_rate_gets_the_rate_flag_quoted(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)

        main(["example", "--output", "my demo.npy", "--duration", "0.01"])

        err = capsys.readouterr().err
        assert f"next: tokeye run {_shell_quote('my demo.npy')} --fs 200000" in err
        assert err.strip().splitlines()[-1].endswith("--fs 200000")

    def test_the_path_is_quoted_for_the_platform_shell(self):
        expected = '"my demo.npy"' if os.name == "nt" else shlex.quote("my demo.npy")

        assert _shell_quote("my demo.npy") == expected
        assert _shell_quote("plain.npy") == "plain.npy"

    def test_a_rate_that_differs_from_the_name_is_passed(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)

        main(
            [
                "example",
                "--output",
                "demo_sr1000.npy",
                "--fs",
                "2000",
                "--duration",
                "0.01",
            ]
        )

        assert capsys.readouterr().err.strip().endswith("demo_sr1000.npy --fs 2000")

    def test_a_matching_rate_in_the_name_needs_no_flag(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)

        main(["example", "--duration", "0.01"])

        err = capsys.readouterr().err
        assert err.strip() == "next: tokeye run tokeye_example_sr200000.npy"

    def test_other_suffixes_get_npy_appended(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)

        exit_code = main(["example", "--output", "out.txt", "--duration", "0.01"])

        assert exit_code == 0
        assert (tmp_path / "out.txt.npy").exists()
        assert not (tmp_path / "out.npy").exists()
        assert capsys.readouterr().out.strip() == "out.txt.npy"

    def test_the_default_name_records_the_default_rate(self, tmp_path):
        path = write_example_signal(tmp_path)

        assert path.name == "tokeye_example_sr200000.npy"
        assert fs_from_name(path) == 200_000.0

    def test_there_is_one_default_rate(self):
        from tokeye.cli import example as example_cli

        assert example_cli.DEFAULT_FS == DEFAULT_FS

    @pytest.mark.parametrize(
        ("fs", "text"),
        [(200_000.0, "200000"), (44_100, "44100"), (1234.5, "1234.5")],
    )
    def test_the_rate_is_formatted_like_the_file_name(self, fs, text):
        assert format_rate(fs) == text
        assert example_filename(fs) == f"tokeye_example_sr{text}.npy"
        assert fs_from_name(example_filename(fs)) == fs

    def test_the_wrapper_keeps_the_old_name_rule(self):
        from tokeye.cli.example import default_output

        assert default_output(1234.5) == example_filename(1234.5)


def test_no_install_hint_is_single_quoted():
    """Windows cmd.exe does not treat single quotes as quotes."""
    root = Path(tokeye.__file__).parent
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if "training" not in path.relative_to(root).parts
        and "'tokeye[" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
