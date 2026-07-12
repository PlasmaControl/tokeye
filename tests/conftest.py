"""Shared pytest fixtures.

Forces Qt's offscreen platform (before any ``QApplication`` is created) so the
GUI tests never need a display — a harmless no-op for the non-Qt suite.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(autouse=True)
def _hermetic_tokeye_env(monkeypatch):
    """Strip ambient TOKEYE_* config so tests see the branch defaults.

    A shell with the deployed module loaded exports TOKEYE_SOURCE,
    TOKEYE_MODULE_DIR, TOKEYE_SLURM_*, …; without this the suite's behaviour
    would depend on whether the developer ran ``module load tokeye`` first.
    Tests that care about a specific value set it via ``monkeypatch.setenv``.
    """
    for var in [k for k in os.environ if k.startswith("TOKEYE_")]:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="session")
def qapp():
    """A single process-wide ``QApplication`` for GUI tests (offscreen)."""
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app
