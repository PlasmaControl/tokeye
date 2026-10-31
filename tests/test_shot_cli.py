"""``tokeye <shot> <signal> [params]`` — parsing, naming, routing, end-to-end."""

from __future__ import annotations

import json

import numpy as np
import pytest
from torch import nn

from tokeye.cli import main, shot


class TestNormalize:
    def test_key_value_and_comma_lists(self):
        assert shot.normalize_argv(["tmin=2000,tmax=3000", "nfft=2048"]) == [
            "--tmin=2000",
            "--tmax=3000",
            "--nfft=2048",
        ]

    def test_modespec_aliases(self):
        assert shot.normalize_argv(["xmin=1", "xmax=2", "decimation=4", "n_fft=512"]) == [
            "--tmin=1",
            "--tmax=2",
            "--decim=4",
            "--nfft=512",
        ]

    def test_bare_flag_words_and_dash_aliases(self):
        assert shot.normalize_argv(["noplot", "--outdir", "/x", "--n-fft=256"]) == [
            "--noplot",
            "--out",
            "/x",
            "--nfft=256",
        ]

    def test_unknown_key_is_an_error(self, capsys):
        with pytest.raises(SystemExit) as exc:
            shot.normalize_argv(["nfftt=10"])
        assert exc.value.code == 2
        assert "unknown parameter 'nfftt'" in capsys.readouterr().err


class TestParse:
    def test_defaults_are_auto_rate_and_training_recipe(self):
        args = shot.parse(["190904", "MPI66M067D"])
        assert (args.shot, args.signal) == (190904, "MPI66M067D")
        assert (args.nfft, args.hop) == (1024, 128)
        assert args.decim is None and args.fs is None  # auto: ~500 kHz
        assert args.tmin is None and args.tmax is None
        assert args.out is None and args.backend is None

    def test_positional_time_window_like_modespec(self):
        args = shot.parse(["190904", "MPI66M067D", "2000", "3000"])
        assert (args.tmin, args.tmax) == (2000.0, 3000.0)

    def test_signal_is_optional(self):
        args = shot.parse(["190904", "2000", "3000"])
        assert args.signal is None
        assert (args.tmin, args.tmax) == (2000.0, 3000.0)

    def test_keywords_and_flags_mix(self):
        args = shot.parse(
            ["190904", "B1", "tmin=100,tmax=200", "--hop", "64", "out=/tmp/x", "keep_dc"]
        )
        assert (args.tmin, args.tmax, args.hop, args.out) == (100.0, 200.0, 64, "/tmp/x")
        assert args.keep_dc is True

    def test_too_many_numbers(self):
        with pytest.raises(SystemExit):
            shot.parse(["190904", "MPI66M067D", "1", "2", "3"])


class TestFileStem:
    def test_default_run_has_plain_name(self):
        args = shot.parse(["190904", "MPI66M067D"])
        assert shot.file_stem(190904, "MPI66M067D", args) == "190904_MPI66M067D"

    def test_non_default_data_params_are_tagged(self):
        args = shot.parse(["190904", "x", "tmin=2000,tmax=3000", "nfft=2048", "decim=4"])
        assert (
            shot.file_stem(190904, "MPI66M067D", args)
            == "190904_MPI66M067D_t2000-3000_nfft2048_decim4"
        )

    def test_fs_target_is_tagged_unless_default(self):
        tag = lambda *a: shot.file_stem(1, "S", shot.parse(["1", "x", *a]))  # noqa: E731
        assert tag("fs=500") == "1_S"
        assert tag("fs=native") == "1_S_native"
        assert tag("fs=250") == "1_S_fs250k"

    def test_display_only_params_do_not_rename(self):
        args = shot.parse(["190904", "x", "fmax=100", "thresh=0.7", "cmap=magma"])
        assert shot.file_stem(190904, "MPI66M067D", args) == "190904_MPI66M067D"


class TestRouting:
    def test_numeric_first_arg_routes_to_shot(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(shot, "run", lambda args: seen.setdefault("args", args) and 0)
        assert main(["190904", "MPI66M067D", "tmin=1", "tmax=2"]) == 0
        assert seen["args"].shot == 190904
        assert seen["args"].tmax == 2.0

    def test_explicit_shot_subcommand(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(shot, "run", lambda args: seen.setdefault("args", args) and 0)
        assert main(["shot", "190904", "mag", "nfft=512"]) == 0
        assert seen["args"].nfft == 512

    def test_webui_is_the_gradio_subcommand(self):
        from tokeye.cli import build_parser

        args = build_parser().parse_args(["webui", "--port", "7861"])
        assert args.port == 7861


class _FakeSource:
    def __init__(self, fs=200e3, n=60_000):
        self.fs, self.n = fs, n

    def fetch(self, shot_no, pointname, tlim=None):
        t = np.arange(self.n) / self.fs * 1e3 + 1000.0
        x = np.sin(2 * np.pi * 20e3 * t / 1e3) + 0.1 * np.random.default_rng(0).standard_normal(
            self.n
        )
        if tlim is not None:
            keep = (t >= tlim[0]) & (t <= tlim[1])
            t, x = t[keep], x[keep]
        return t, x, self.fs


def test_end_to_end_writes_png_npz_json(tmp_path, monkeypatch, capsys):
    from tokeye import accel

    monkeypatch.setattr("tokeye.sources.factory.get_source", lambda *a, **k: _FakeSource())
    runner = accel.TorchRunner(nn.Conv2d(1, 2, 1), "stub")
    monkeypatch.setattr(accel, "load_runner", lambda *a, **k: runner)
    monkeypatch.setenv("TOKEYE_RESULTS", str(tmp_path))

    assert main(["190904", "MPI66M067D", "1000", "1200"]) == 0

    out = capsys.readouterr().out
    shot_dir = tmp_path / "190904"
    assert f"Saved to {shot_dir}/" in out
    stem = "190904_MPI66M067D_t1000-1200"
    for ext in ("png", "npz", "json"):
        assert (shot_dir / f"{stem}.{ext}").is_file()
    d = np.load(shot_dir / f"{stem}.npz")
    assert d["mask"].dtype == np.uint8
    assert d["mask"].shape == (2, *d["spectrogram"].shape)
    assert d["t_ms"].size == d["spectrogram"].shape[1]
    assert d["f_khz"].size == d["spectrogram"].shape[0] == 512
    meta = json.loads((shot_dir / f"{stem}.json").read_text())
    assert meta["fs_hz"] == meta["fs_native_hz"] == 200e3
    assert meta["decimation"] == 1
    assert (meta["n_fft"], meta["hop"]) == (1024, 128)
    assert meta["files"] == [f"{stem}.png", f"{stem}.npz", f"{stem}.json"]
    assert shot_dir.stat().st_mode & 0o020  # group-writable shared shot dir


@pytest.mark.parametrize(
    ("argv_extra", "fs_out", "decim", "stem"),
    [
        ([], 500e3, 4, "190904_B1_t1000-1100"),  # 2 MHz -> x4 -> 500 kHz
        (["fs=native"], 2e6, 1, "190904_B1_t1000-1100_native"),
        (["decim=2"], 1e6, 2, "190904_B1_t1000-1100_decim2"),
    ],
)
def test_fast_signals_are_decimated_to_500khz_by_default(
    tmp_path, monkeypatch, capsys, argv_extra, fs_out, decim, stem
):
    from tokeye import accel

    monkeypatch.setattr(
        "tokeye.sources.factory.get_source", lambda *a, **k: _FakeSource(fs=2e6, n=400_000)
    )
    monkeypatch.setattr(
        accel, "load_runner", lambda *a, **k: accel.TorchRunner(nn.Conv2d(1, 2, 1), "stub")
    )
    monkeypatch.setenv("TOKEYE_RESULTS", str(tmp_path))

    assert main(["190904", "B1", "1000", "1100", "noplot", *argv_extra]) == 0

    meta = json.loads((tmp_path / "190904" / f"{stem}.json").read_text())
    assert meta["fs_native_hz"] == 2e6
    assert meta["fs_hz"] == fs_out
    assert meta["decimation"] == decim
    d = np.load(tmp_path / "190904" / f"{stem}.npz")
    assert d["f_khz"][-1] == pytest.approx(fs_out / 2 / 1e3)


def test_auto_decimation_factors():
    from tokeye.transforms import auto_decimation

    assert auto_decimation(2e6) == 4
    assert auto_decimation(1e6) == 2
    assert auto_decimation(1.6667e6) == 3  # CO2 -> ~556 kHz, never below 500
    assert auto_decimation(500e3) == 1
    assert auto_decimation(499_999.7) == 1
    assert auto_decimation(200e3) == 1  # never upsample
    assert auto_decimation(2e6, 0) == 1  # fs=native


def test_out_argument_overrides_results_root(tmp_path, monkeypatch, capsys):
    from tokeye import accel

    monkeypatch.setattr("tokeye.sources.factory.get_source", lambda *a, **k: _FakeSource())
    monkeypatch.setattr(
        accel, "load_runner", lambda *a, **k: accel.TorchRunner(nn.Conv2d(1, 2, 1), "stub")
    )
    monkeypatch.setenv("TOKEYE_RESULTS", str(tmp_path / "ignored"))

    assert main(["190904", "mag", f"out={tmp_path / 'mine'}", "noplot", "nodata"]) == 0

    assert f"Saved to {tmp_path / 'mine' / '190904'}/" in capsys.readouterr().out
    assert (tmp_path / "mine" / "190904" / "190904_MPI66M067D.json").is_file()
    assert not (tmp_path / "ignored").exists()


# ── shared results dir: never write through a planted symlink ──────────────────
def _stub_run(monkeypatch, tmp_path):
    from tokeye import accel

    monkeypatch.setattr("tokeye.sources.factory.get_source", lambda *a, **k: _FakeSource())
    monkeypatch.setattr(
        accel, "load_runner", lambda *a, **k: accel.TorchRunner(nn.Conv2d(1, 2, 1), "stub")
    )
    monkeypatch.setenv("TOKEYE_RESULTS", str(tmp_path / "results"))


def test_planted_symlink_at_result_name_is_replaced_not_followed(tmp_path, monkeypatch):
    _stub_run(monkeypatch, tmp_path)
    victim = tmp_path / "victim_file"
    victim.write_text("precious")
    victim.chmod(0o600)
    shot_dir = tmp_path / "results" / "190904"
    shot_dir.mkdir(parents=True)
    for ext in ("png", "npz", "json"):
        (shot_dir / f"190904_MPI66M067D_t1000-1200.{ext}").symlink_to(victim)

    assert main(["190904", "MPI66M067D", "1000", "1200"]) == 0

    assert victim.read_text() == "precious"  # never written through
    assert (victim.stat().st_mode & 0o777) == 0o600  # never chmod-ed through
    for ext in ("png", "npz", "json"):
        out = shot_dir / f"190904_MPI66M067D_t1000-1200.{ext}"
        assert out.is_file() and not out.is_symlink()
    assert not list(shot_dir.glob(".*.tmp"))  # no temp files left behind


def test_new_shot_dirs_are_sticky(tmp_path, monkeypatch):
    _stub_run(monkeypatch, tmp_path)
    assert main(["190904", "MPI66M067D", "1000", "1200", "noplot", "nodata"]) == 0
    mode = (tmp_path / "results" / "190904").stat().st_mode
    assert mode & 0o1000 and mode & 0o2000  # sticky + setgid: nobody replaces others' files


def test_someone_elses_result_is_kept_and_we_save_our_own_copy(tmp_path, monkeypatch, capsys):
    import getpass

    from tokeye.cli import shot as shot_mod

    _stub_run(monkeypatch, tmp_path)
    shot_dir = tmp_path / "results" / "190904"
    shot_dir.mkdir(parents=True)
    theirs = shot_dir / "190904_MPI66M067D_t1000-1200.json"
    theirs.write_text("{}")
    monkeypatch.setattr(shot_mod, "_owned_by_someone_else", lambda p: p == theirs)

    assert main(["190904", "MPI66M067D", "1000", "1200", "noplot", "nodata"]) == 0

    assert theirs.read_text() == "{}"
    mine = shot_dir / f"190904_MPI66M067D_t1000-1200_{getpass.getuser()}.json"
    assert mine.is_file()
    assert mine.name in capsys.readouterr().out
