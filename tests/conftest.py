"""Shared pytest fixtures.

Forces Qt's offscreen platform (before any ``QApplication`` is created) so the
GUI tests never need a display — a harmless no-op for the non-Qt suite.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session")
def qapp():
    """A single process-wide ``QApplication`` for GUI tests (offscreen)."""
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app
