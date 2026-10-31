"""The Omega launcher (deploy/omega/fusion/bin/tokeye) must set up per-user state
from scratch on every start, never assuming a folder's contents survived.

Settings live in ~/.tokeye (a commented ``settings`` template is written when
missing); data lives in /cscratch/$USER/.tokeye, whose files are deleted daily.
These run the real script against a fake install whose "python" just prints the
environment it was handed.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "deploy" / "omega" / "fusion" / "bin" / "tokeye"
_SHOWN = (
    "TOKEYE_CONFIG_DIR|TOKEYE_DATA_DIR|TOKEYE_CACHE|TOKEYE_RUNS_DIR|TOKEYE_RESULTS|"
    "TOKEYE_FS_KHZ|TOKEYE_THREADS|TOKEYE_BACKEND|GRADIO_TEMP_DIR|MPLCONFIGDIR"
)


@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "install"
    (root / "bin").mkdir(parents=True)
    (root / "env" / "bin").mkdir(parents=True)
    shutil.copy(LAUNCHER, root / "bin" / "tokeye")
    py = root / "env" / "bin" / "python"
    py.write_text(f"#!/bin/bash\nenv | grep -E '^({_SHOWN})='\necho \"ARGS $*\"\n")
    py.chmod(0o755)
    return root


@pytest.fixture
def dirs(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    return home, home / ".tokeye", tmp_path / "scratch" / ".tokeye"


def _run(root, home, data, *args, **extra_env):
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),
        "USER": os.environ.get("USER", "tester"),
        "TOKEYE_DATA_DIR": str(data),
        **extra_env,
    }
    r = subprocess.run(
        ["bash", str(root / "bin" / "tokeye"), *args],
        env=env, capture_output=True, text=True, check=False,
    )
    r.vars = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
    return r


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


def test_first_start_builds_settings_and_data_dirs(fake_root, dirs):
    home, cfg, data = dirs
    r = _run(fake_root, home, data, "--version")
    assert r.returncode == 0, r.stderr
    # settings in ~/.tokeye
    assert _mode(cfg) == 0o700
    assert (cfg / "mpl").is_dir()
    text = (cfg / "settings").read_text()
    assert "#TOKEYE_RESULTS=/fusion/projects/results/tokeye" in text
    assert _mode(cfg / "settings") == 0o600
    # data on scratch
    assert _mode(data) == 0o700
    for sub in ("cache", "runs", "gradio"):
        assert (data / sub).is_dir(), sub
    assert not (data / "mpl").exists() and not (cfg / "cache").exists()
    v = r.vars
    assert v["TOKEYE_CONFIG_DIR"] == str(cfg)
    assert v["MPLCONFIGDIR"] == str(cfg / "mpl")
    assert v["TOKEYE_DATA_DIR"] == str(data)
    assert v["TOKEYE_CACHE"] == str(data / "cache")
    assert v["TOKEYE_RUNS_DIR"] == str(data / "runs")
    assert v["GRADIO_TEMP_DIR"] == str(data / "gradio")
    assert "TOKEYE_RESULTS" not in v  # template is all comments: built-in defaults
    assert "ARGS -m tokeye.cli --version" in r.stdout


def test_empty_folders_are_not_trusted(fake_root, dirs):
    """The daily /cscratch sweep (or a wiped ~/.tokeye) leaves folders, no files."""
    home, cfg, data = dirs
    cfg.mkdir(mode=0o755)
    (cfg / "settings").write_text("")  # emptied file -> template rewritten
    data.mkdir(parents=True, mode=0o755)
    r = _run(fake_root, home, data)
    assert r.returncode == 0, r.stderr
    assert "TOKEYE_RESULTS" in (cfg / "settings").read_text()
    assert _mode(cfg) == _mode(data) == 0o700
    assert all((data / s).is_dir() for s in ("cache", "runs", "gradio"))


def test_half_swept_data_dir_is_completed(fake_root, dirs):
    home, _cfg, data = dirs
    (data / "gradio").mkdir(parents=True)
    (data / "cache").write_text("stray file where a folder belongs")
    r = _run(fake_root, home, data)
    assert r.returncode == 0, r.stderr
    assert (data / "cache").is_dir() and (data / "runs").is_dir()


def test_user_settings_are_applied_and_kept(fake_root, dirs):
    home, cfg, data = dirs
    cfg.mkdir()
    (cfg / "settings").write_text(
        "TOKEYE_RESULTS=~/my_runs\n"
        'TOKEYE_FS_KHZ="native"\n'
        "TOKEYE_THREADS=4   \n"
        "TOKEYE_EVIL=$(touch /tmp/x)\n"  # unknown key: ignored, never evaluated
        "#TOKEYE_BACKEND=torch\n"
    )
    r = _run(fake_root, home, data)
    assert r.returncode == 0, r.stderr
    assert r.vars["TOKEYE_RESULTS"] == f"{home}/my_runs"
    assert r.vars["TOKEYE_FS_KHZ"] == "native"
    assert r.vars["TOKEYE_THREADS"] == "4"
    assert "TOKEYE_BACKEND" not in r.vars
    assert "TOKEYE_EVIL" not in r.stdout
    assert (cfg / "settings").read_text().startswith("TOKEYE_RESULTS=~/my_runs")  # not overwritten


def test_environment_overrides_settings_file(fake_root, dirs):
    home, cfg, data = dirs
    cfg.mkdir()
    (cfg / "settings").write_text("TOKEYE_THREADS=4\n")
    r = _run(fake_root, home, data, TOKEYE_THREADS="2")
    assert r.vars["TOKEYE_THREADS"] == "2"


def test_data_dir_from_settings_file(fake_root, dirs, tmp_path):
    home, cfg, _data = dirs
    cfg.mkdir()
    target = tmp_path / "elsewhere" / ".tokeye"
    (cfg / "settings").write_text(f"TOKEYE_DATA_DIR={target}\n")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "USER": os.environ.get("USER", "t")}
    r = subprocess.run(
        ["bash", str(fake_root / "bin" / "tokeye")], env=env,
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr
    assert f"TOKEYE_CACHE={target}/cache" in r.stdout
    assert (target / "cache").is_dir()


def test_symlinked_data_dir_is_refused(fake_root, dirs, tmp_path):
    home, _cfg, data = dirs
    (tmp_path / "other").mkdir()
    data.parent.mkdir(parents=True)
    data.symlink_to(tmp_path / "other")
    r = _run(fake_root, home, data)
    assert r.returncode == 1
    assert "refusing" in r.stderr


def test_missing_env_says_how_to_rebuild(fake_root, dirs):
    home, _cfg, data = dirs
    shutil.rmtree(fake_root / "env")
    r = _run(fake_root, home, data)
    assert r.returncode == 1
    assert "install.sh" in r.stderr


def test_fs_setting_reaches_python(monkeypatch):
    from tokeye.transforms import DEFAULT_TARGET_FS_HZ, default_target_fs_hz

    assert default_target_fs_hz() == DEFAULT_TARGET_FS_HZ
    monkeypatch.setenv("TOKEYE_FS_KHZ", "250")
    assert default_target_fs_hz() == 250e3
    monkeypatch.setenv("TOKEYE_FS_KHZ", "native")
    assert default_target_fs_hz() == 0.0
    monkeypatch.setenv("TOKEYE_FS_KHZ", "garbage")
    assert default_target_fs_hz() == DEFAULT_TARGET_FS_HZ
