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


def _git(cwd: Path, *args: str) -> None:
    """Run a git command in ``cwd`` (identity forced so commits work anywhere)."""
    subprocess.run(
        ["git", "-c", "user.email=t@t.invalid", "-c", "user.name=test", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


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


def test_ensure_env_sanitizes_inherited_env(tmp_path: Path):
    """A HEALTHY env whose caller's login shell carries the two known poisons the
    modulefile normally strips — a py3.7 MDSplus on ``PYTHONPATH`` and a system
    libstdc++ ahead on ``LD_LIBRARY_PATH`` — must still pass the health check.
    ensure-env.sh has to sanitize its own environment (mirror tokeye.lua: unset
    ``PYTHONPATH``, prepend the env's own ``lib``) before running the check, or a
    healthy env false-negatives into a rebuild loop. The stub python enforces
    both invariants: it exits 1 if ``PYTHONPATH`` is set at all, or if
    ``LD_LIBRARY_PATH`` does not lead with the env's own ``lib``.
    """
    env_lib = tmp_path / _env_name() / "lib"
    envbin = tmp_path / _env_name() / "bin"
    _write_exec(
        envbin / "python",
        "#!/usr/bin/env bash\n"
        # A leftover py3.7-MDSplus PYTHONPATH must be gone entirely.
        'if [[ -n "${PYTHONPATH:-}" ]]; then\n'
        '  echo "poison: PYTHONPATH=$PYTHONPATH" >&2; exit 1\n'
        "fi\n"
        # The env's own lib must LEAD LD_LIBRARY_PATH (system /lib64 must not win).
        'case "${LD_LIBRARY_PATH:-}" in\n'
        f'  "{env_lib}"*) ;;\n'
        '  *) echo "poison: LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" >&2; exit 1 ;;\n'
        "esac\n"
        "exit 0\n",
    )
    _write_exec(envbin / "tokeye-app", "#!/usr/bin/env bash\nexit 0\n")

    decoy = tmp_path / "decoy-lib64"  # stands in for the login /lib64 poison
    result = subprocess.run(
        [str(ENSURE)],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        env=_env_without(
            TOKEYE_DIR=str(tmp_path),
            TOKEYE_REPO=str(REPO),
            PYTHONPATH=str(tmp_path / "py37-mdsplus"),
            LD_LIBRARY_PATH=str(decoy),
        ),
    )

    assert result.returncode == 0, result.stdout + result.stderr


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


def test_ensure_env_rebuild_never_passes_dash_y_to_mamba(tmp_path: Path):
    """Unhealthy env + ``--yes`` → full rebuild flow against a fake mamba.

    Old conda/mamba's ``env create`` subcommand has NO ``-y/--yes`` flag (it
    never prompts — non-interactive by design) and hard-errors on it — observed
    live on omega14 via the 2022-mambaforge absolute fallback:
    ``mamba: error: unrecognized arguments: -y``. The fake mamba mimics that
    old behaviour: it exits 2 on any ``-y``/``--yes`` argument, and otherwise
    "creates" the env by dropping a healthy stub ``python`` and ``pip`` into
    ``env-<arch>/bin`` so the script's subsequent ``pip install`` and
    post-rebuild health check run against them.
    """
    envbin = tmp_path / _env_name() / "bin"
    _write_exec(envbin / "python", "#!/usr/bin/env bash\nexit 1\n")  # unhealthy

    fakebin = tmp_path / "fakebin"
    _write_exec(
        fakebin / "mamba",
        "#!/usr/bin/env bash\n"
        'for a in "$@"; do\n'
        '  if [[ "$a" == "-y" || "$a" == "--yes" ]]; then\n'
        '    echo "mamba: error: unrecognized arguments: -y" >&2\n'
        "    exit 2\n"
        "  fi\n"
        "done\n"
        f'mkdir -p "{envbin}"\n'
        f"printf '#!/usr/bin/env bash\\nexit 0\\n' > \"{envbin}/python\"\n"
        f"printf '#!/usr/bin/env bash\\nexit 0\\n' > \"{envbin}/pip\"\n"
        f'chmod +x "{envbin}/python" "{envbin}/pip"\n',
    )

    result = subprocess.run(
        [str(ENSURE), "--yes"],
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

    assert result.returncode == 0, result.stdout + result.stderr
    assert "rebuild OK" in (result.stdout + result.stderr)


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


# --- 8. install-home.sh publish resilience ----------------------------------


def test_install_home_survives_unreachable_origin(tmp_path: Path):
    """A durable checkout whose origin is gone (swept storage) must not abort the
    publish: the pull fails, install-home.sh warns and keeps the existing
    checkout, then still publishes the launchers/modulefile and (healthy) env.

    The dest ``.git`` branch is taken first, so the clone-URL logic never touches
    the network — no ls-remote, no GitHub.
    """
    dest = tmp_path / "durable"
    dest.mkdir()
    # A real checkout: pyproject.toml (ensure-env.sh walks up to it) + the deploy/
    # tree the publish + ensure-env steps read.
    shutil.copytree(OMEGA.parent, dest / "deploy")
    shutil.copy(REPO / "pyproject.toml", dest / "pyproject.toml")
    _git(dest, "init", "-q")
    _git(dest, "checkout", "-q", "-b", "diiid")
    _git(dest, "add", "-A")
    _git(dest, "commit", "-q", "-m", "seed")
    # origin points at a path that does not exist → `git pull --ff-only` fails.
    _git(dest, "remote", "add", "origin", str(tmp_path / "gone-origin.git"))

    # A HEALTHY fake env under the publish/env root so ensure-env.sh reports OK.
    tokeye_dir = tmp_path / "envroot"
    envbin = tokeye_dir / _env_name() / "bin"
    _write_exec(envbin / "python", "#!/usr/bin/env bash\nexit 0\n")
    _write_exec(envbin / "tokeye-app", "#!/usr/bin/env bash\nexit 0\n")

    result = subprocess.run(
        [str(INSTALL), str(dest)],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        env=_env_without("TOKEYE_REPO", TOKEYE_DIR=str(tokeye_dir)),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARNING: origin unreachable" in result.stderr
    # Publish artifacts landed under the tmp TOKEYE_DIR.
    assert (tokeye_dir / "bin" / "tokeye").exists()
    assert (tokeye_dir / "modulefiles" / "tokeye.lua").exists()
    sidecar = tokeye_dir / "bin" / ".tokeye-repo"
    assert sidecar.exists()
    assert sidecar.read_text().strip() == str(dest)


# --- 7. lua sanity (text-level, no Lmod) ------------------------------------


def test_lua_has_module_dir_guard_and_shim_prepend():
    text = LUA.read_text()
    assert "TOKEYE_MODULE_DIR" in text
    assert "isDir" in text
    env_bin = text.index('pathJoin(root, env, "bin")')
    shim_bin = text.index('pathJoin(root, "bin")')
    assert env_bin < shim_bin, "shim-dir prepend must come after the env-bin prepend"
