"""Behavioural tests for the Omega deploy scripts (``deploy/omega/``).

These drive the launcher shim and ``ensure-env.sh`` under real bash — no
cluster, no network, no mamba, no ``module``, no torch/gradio. Each test builds
a fake ``$TOKEYE_DIR`` (and, where needed, a fake ``env-<arch>`` with stub
``bin/python`` / ``bin/tokeye``) in ``tmp_path`` and asserts on the script's
observable behaviour. The stubs stand in for the real conda env so the tests
run anywhere the dev suite runs.
"""

from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
OMEGA = REPO / "deploy" / "omega"
SHIM = OMEGA / "bin" / "tokeye"
ENSURE = OMEGA / "ensure-env.sh"
INSTALL = OMEGA / "install-home.sh"
LUA = OMEGA / "modulefiles" / "tokeye.lua"


def _env_name() -> str:
    """``env-<arch>`` dir name — mirrors tokeye.lua and the scripts' mapping."""
    return "env-aarch64" if platform.machine() == "aarch64" else "env-x86_64"


def _write_exec(path: Path, body: str) -> None:
    """Write ``body`` to ``path`` (creating parents) and make it executable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _env_without(*drop: str, **overrides: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(overrides)
    return env


# --- 1. syntax --------------------------------------------------------------


@pytest.mark.parametrize("script", [SHIM, ENSURE, INSTALL], ids=lambda p: p.name)
def test_bash_syntax_ok(script: Path):
    result = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# --- 2/3. the shim ----------------------------------------------------------


def test_shim_fast_path_execs_real_tokeye(tmp_path: Path):
    """Healthy env → shim execs the real tokeye by absolute path, no self-heal."""
    envbin = tmp_path / _env_name() / "bin"
    _write_exec(envbin / "tokeye", '#!/usr/bin/env bash\necho "REAL $@"\n')

    result = subprocess.run(
        [str(SHIM), "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=_env_without(TOKEYE_DIR=str(tmp_path)),
    )

    assert result.returncode == 0, result.stderr
    assert "REAL --help" in result.stdout
    assert "self-heal" not in (result.stdout + result.stderr)


def test_shim_missing_env_runs_selfheal_and_propagates_failure(tmp_path: Path):
    """No env binary → shim announces self-heal, runs sibling ensure-env.sh,
    and propagates its non-zero exit."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shutil.copy(SHIM, bindir / "tokeye")
    (bindir / "tokeye").chmod(0o755)
    _write_exec(
        bindir / "ensure-env.sh",
        '#!/usr/bin/env bash\necho "STUB ensure ran" >&2\nexit 1\n',
    )
    empty_root = tmp_path / "envroot"  # no env-<arch>/bin/tokeye under here
    empty_root.mkdir()

    result = subprocess.run(
        [str(bindir / "tokeye"), "gui"],
        capture_output=True,
        text=True,
        check=False,
        env=_env_without(TOKEYE_DIR=str(empty_root)),
    )

    assert result.returncode != 0
    assert "self-heal" in result.stderr


# --- 4/5. ensure-env.sh health + rebuild gating -----------------------------


def test_ensure_env_healthy_no_rebuild(tmp_path: Path):
    """Stub python that imports cleanly + existing launcher → OK, no prompt."""
    envbin = tmp_path / _env_name() / "bin"
    _write_exec(envbin / "python", "#!/usr/bin/env bash\nexit 0\n")
    _write_exec(envbin / "tokeye-app", "#!/usr/bin/env bash\nexit 0\n")

    result = subprocess.run(
        [str(ENSURE)],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        env=_env_without(TOKEYE_DIR=str(tmp_path), TOKEYE_REPO=str(REPO)),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    combined = (result.stdout + result.stderr).lower()
    assert "rebuild env" not in combined
    assert "rebuilding" not in combined


def test_ensure_env_noninteractive_refusal(tmp_path: Path):
    """Unhealthy env, no tty, no --yes → exit 1, print the rebuild command,
    and DO NOT invoke mamba."""
    envbin = tmp_path / _env_name() / "bin"
    _write_exec(envbin / "python", "#!/usr/bin/env bash\nexit 1\n")  # imports fail

    # A fake mamba on PATH that leaves a sentinel iff (wrongly) invoked.
    fakebin = tmp_path / "fakebin"
    sentinel = tmp_path / "mamba-was-run"
    _write_exec(fakebin / "mamba", f'#!/usr/bin/env bash\ntouch "{sentinel}"\n')

    result = subprocess.run(
        [str(ENSURE)],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        env=_env_without(
            TOKEYE_DIR=str(tmp_path),
            TOKEYE_REPO=str(REPO),
            PATH=f"{fakebin}:{os.environ['PATH']}",
        ),
    )

    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "mamba env create" in combined  # the exact rebuild command is shown
    assert not sentinel.exists()  # …but mamba was never actually run


# --- 6. repo resolution -----------------------------------------------------


def test_repo_resolution_via_sidecar(tmp_path: Path):
    """Published-style copy (script + .tokeye-repo sidecar), no TOKEYE_REPO →
    resolves REPO from the sidecar and reports healthy."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shutil.copy(ENSURE, bindir / "ensure-env.sh")
    (bindir / "ensure-env.sh").chmod(0o755)
    (bindir / ".tokeye-repo").write_text(f"{REPO}\n")

    envbin = tmp_path / _env_name() / "bin"
    _write_exec(envbin / "python", "#!/usr/bin/env bash\nexit 0\n")
    _write_exec(envbin / "tokeye-app", "#!/usr/bin/env bash\nexit 0\n")

    result = subprocess.run(
        [str(bindir / "ensure-env.sh")],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        env=_env_without("TOKEYE_REPO", TOKEYE_DIR=str(tmp_path)),
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_repo_resolution_failure_names_env_var(tmp_path: Path):
    """No sidecar, no TOKEYE_REPO, copied outside any repo → exit 1 naming
    TOKEYE_REPO as a remedy."""
    lonely = tmp_path / "nowhere"
    lonely.mkdir()
    shutil.copy(ENSURE, lonely / "ensure-env.sh")
    (lonely / "ensure-env.sh").chmod(0o755)

    result = subprocess.run(
        [str(lonely / "ensure-env.sh")],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        env=_env_without("TOKEYE_REPO", TOKEYE_DIR=str(tmp_path)),
    )

    assert result.returncode == 1
    assert "TOKEYE_REPO" in (result.stdout + result.stderr)


# --- 7. lua sanity (text-level, no Lmod) ------------------------------------


def test_lua_has_module_dir_guard_and_shim_prepend():
    text = LUA.read_text()
    assert "TOKEYE_MODULE_DIR" in text
    assert "isDir" in text
    env_bin = text.index('pathJoin(root, env, "bin")')
    shim_bin = text.index('pathJoin(root, "bin")')
    assert env_bin < shim_bin, "shim-dir prepend must come after the env-bin prepend"
