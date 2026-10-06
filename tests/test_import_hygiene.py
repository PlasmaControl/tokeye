"""HPC-safe import guarantees.

``tokeye.batch`` must stay importable without pulling in gradio (it needs to
run on HPC login/compute nodes that may not have a display or gradio
installed). ``tokeye.cli`` must stay importable without pulling in gradio
*or* torch, so plain ``tokeye --help`` returns instantly.

These checks run in a subprocess: importing the modules in-process (as the
rest of the test suite does throughout the run) would leave the relevant
modules in ``sys.modules`` regardless of what any single import statement
pulls in, masking a regression.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_batch_import_does_not_pull_in_gradio():
    code = (
        "import tokeye.batch, sys; "
        "assert 'gradio' not in sys.modules; "
        "assert 'matplotlib.pyplot' not in sys.modules; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_package_import_does_not_pull_in_torch():
    """``from tokeye import TokEye`` is lazy (PEP 562): the bare package
    import (and ``tokeye.__version__``) must stay free of torch, scipy and
    numpy or ``tokeye --help`` slows to a crawl."""
    code = (
        "import sys, tokeye; "
        "bad = [m for m in ('torch', 'scipy', 'numpy') if m in sys.modules]; "
        "assert not bad, bad; "
        "print(tokeye.__version__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


HEAVY = ("torch", "gradio", "scipy")


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["run"],
        ["download"],
        ["example"],
        ["elmspec"],
        ["alfvenspec"],
        ["app"],
        ["info"],
    ],
    ids=lambda argv: " ".join(argv) or "tokeye",
)
def test_cli_help_does_not_pull_in_heavy_modules(argv):
    """``tokeye [COMMAND] --help`` must stay instant on a laptop."""
    code = (
        "import sys\n"
        "from tokeye.cli import main\n"
        "try:\n"
        f"    main({[*argv, '--help']!r})\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 0, exc.code\n"
        f"bad = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_python_m_tokeye_help_does_not_pull_in_heavy_modules():
    """``python -m tokeye --help`` is ``tokeye --help``, and as light."""
    code = (
        "import runpy, sys\n"
        "sys.argv = ['tokeye', '--help']\n"
        "try:\n"
        "    runpy.run_module('tokeye', run_name='__main__', alter_sys=True)\n"
        "except SystemExit as exc:\n"
        "    assert exc.code == 0, exc.code\n"
        "else:\n"
        "    raise AssertionError('no SystemExit')\n"
        f"bad = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert "usage: tokeye" in result.stdout
    assert result.stdout.strip().endswith("ok")


def test_python_m_tokeye_exits_with_the_cli_code():
    import tokeye

    version = subprocess.run(
        [sys.executable, "-m", "tokeye", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    no_command = subprocess.run(
        [sys.executable, "-m", "tokeye"], capture_output=True, text=True, check=False
    )

    assert version.returncode == 0, version.stderr
    assert version.stdout.strip() == f"tokeye {tokeye.__version__}"
    assert no_command.returncode == 2
    assert "usage: tokeye" in no_command.stderr
