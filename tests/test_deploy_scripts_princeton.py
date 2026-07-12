"""Behavioural tests for the stellar deploy kit (``deploy/princeton/``).

``setup.sh`` runs under real bash against a fake repo root, a fake ``uv`` on
``$PATH`` and tmp module/HF roots — no cluster, no network, no ``module``
command needed. Asserts on the rendered modulefile and the script's
observable behaviour.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KIT = REPO / "deploy" / "princeton"
SETUP = KIT / "setup.sh"
APP_SH = KIT / "tokeye-app.sh"
TCL = KIT / "modulefiles" / "tokeye.tcl"


def test_scripts_parse_under_bash():
    for script in (SETUP, APP_SH):
        subprocess.run(["bash", "-n", str(script)], check=True)
        assert os.access(script, os.X_OK), f"{script.name} must be executable"


def test_tcl_template_shape():
    text = TCL.read_text()
    lines = text.splitlines()
    assert lines[0] == "#%Module1.0"  # magic cookie must be the very first line
    # The env contract the rest of the branch relies on:
    for needle in (
        'setenv TOKEYE_SOURCE          "foundation"',
        "setenv TOKEYE_FOUNDATION_DIR",
        "setenv HF_HOME",
        "setenv TOKEYE_MODULE_DIR",
        'setenv TOKEYE_SLURM_PARTITION "gpu"',
        'setenv TOKEYE_SLURM_GRES      "gpu:a100:1"',
        "setenv QT_QPA_PLATFORM",
        "unsetenv PYTHONPATH",
        'prepend-path PATH "$root/.venv/bin"',
        "/scratch/gpfs/$env(USER)/tokeye/runs",
    ):
        assert needle in text, f"template lost: {needle}"
    # Substitution tokens present in the template (setup.sh fills them).
    for token in ("@ROOT@", "@MODULES_ROOT@", "@FOUNDATION_DIR@", "@HF_HOME@"):
        assert token in text


def _write_exec(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _fake_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    """A sandbox: fake repo root, fake uv logging its calls, tmp roots."""
    root = tmp_path / "tokeye"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='tokeye'\n")

    bin_dir = tmp_path / "bin"
    uv_log = tmp_path / "uv.log"
    _write_exec(
        bin_dir / "uv",
        "#!/bin/bash\n"
        f'echo "uv $*" >> "{uv_log}"\n'
        'if [ "$1" = "run" ] && [ "$2" = "tokeye" ] && [ "$3" = "--version" ]; then\n'
        "  echo 9.9.9\n"
        "fi\n"
        "exit 0\n",
    )

    modules_root = tmp_path / "modules"
    hf_home = tmp_path / "hf"
    home = tmp_path / "home"
    home.mkdir()
    env = {
        **{k: v for k, v in os.environ.items() if k != "TOKEYE_DIR"},
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(home),
        "TOKEYE_DIR": str(root),
        "TOKEYE_MODULES_ROOT": str(modules_root),
        "TOKEYE_HF_HOME": str(hf_home),
        "TOKEYE_FOUNDATION_DIR": str(tmp_path),  # exists -> no warning noise
    }
    return env, root, modules_root, uv_log


def test_setup_renders_modulefile_and_calls_uv(tmp_path):
    env, root, modules_root, uv_log = _fake_env(tmp_path)

    result = subprocess.run(
        ["bash", str(SETUP), "--skip-download"],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr

    module_file = modules_root / "tokeye" / "9.9.9"
    assert module_file.is_file(), result.stdout
    # stellar's environment-modules needs an explicit default version.
    version_file = modules_root / "tokeye" / ".version"
    assert 'set ModulesVersion "9.9.9"' in version_file.read_text()
    assert version_file.read_text().startswith("#%Module1.0")
    rendered = module_file.read_text()
    assert rendered.splitlines()[0] == "#%Module1.0"
    assert f'set root "{root}"' in rendered
    assert f"setenv TOKEYE_MODULE_DIR      \"{modules_root}\"" in rendered
    assert str(tmp_path) in rendered  # foundation dir substituted
    assert "@" + "ROOT" + "@" not in rendered  # no token survives

    calls = uv_log.read_text()
    assert "uv sync --extra app --extra gui --extra princeton" in calls
    assert "uv run tokeye download" not in calls  # --skip-download honored
    assert "import h5py, tokeye" in calls  # self-check ran

    # Non-princeton checkout: warn, don't fail.
    assert "not 'princeton'" in result.stderr or "warning" in result.stderr


def test_setup_downloads_weights_by_default(tmp_path):
    env, _root, _modules_root, uv_log = _fake_env(tmp_path)
    result = subprocess.run(
        ["bash", str(SETUP)], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "uv run tokeye download" in uv_log.read_text()


def test_setup_is_idempotent(tmp_path):
    env, _root, modules_root, _uv_log = _fake_env(tmp_path)
    for _ in range(2):
        result = subprocess.run(
            ["bash", str(SETUP), "--skip-download"],
            env=env, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
    assert (modules_root / "tokeye" / "9.9.9").is_file()


def test_setup_bashrc_appends_exactly_once(tmp_path):
    env, _root, modules_root, _uv_log = _fake_env(tmp_path)
    bashrc = Path(env["HOME"]) / ".bashrc"
    bashrc.write_text("# existing\n")

    for _ in range(2):
        subprocess.run(
            ["bash", str(SETUP), "--skip-download", "--bashrc"],
            env=env, capture_output=True, text=True, check=True,
        )

    line = f"module use --append {modules_root}"
    assert bashrc.read_text().count(line) == 1


def test_setup_rejects_bad_root_and_bad_flag(tmp_path):
    env, _root, _modules_root, _uv_log = _fake_env(tmp_path)

    bad = dict(env, TOKEYE_DIR=str(tmp_path / "nowhere"))
    result = subprocess.run(
        ["bash", str(SETUP), "--skip-download"],
        env=bad, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert "pyproject.toml" in result.stderr

    result = subprocess.run(
        ["bash", str(SETUP), "--frobnicate"],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "usage:" in result.stderr
