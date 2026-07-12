"""Application entry point: build the QApplication, theme, and MainWindow.

``run`` is the single lazy boundary for the GUI: it imports PySide6, pyqtgraph
and the rest of ``tokeye.gui`` only when actually called, so importing
``tokeye``/``tokeye.cli`` stays free of Qt (see ``tests/test_import_hygiene``).
"""

from __future__ import annotations

import sys


def run(view: str | None = None, self_test: bool = False) -> int:
    """Launch the GUI event loop and return the Qt exit code.

    Parameters
    ----------
    view:
        Optional view key to open on (``"spectrogram"`` or ``"modespec"``).
    self_test:
        If true, build and show the window, run a single event-loop tick, then
        quit — used by ``tokeye gui --self-test`` for headless/offscreen CI.
    """
    from PySide6 import QtCore, QtWidgets

    from tokeye.gui.main_window import MainWindow
    from tokeye.gui.theme import apply_theme, configure_pyqtgraph
    from tokeye.sources.factory import source_label

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1] or ["tokeye"])
    app.setApplicationName("TokEye")
    app.setApplicationDisplayName(f"TokEye — {source_label()}")
    app.setOrganizationName("PlasmaControl")

    configure_pyqtgraph()
    apply_theme(app)

    window = MainWindow(initial_view=view)
    window.show()

    if self_test:
        # One event-loop tick then quit: exercises show()/paint wiring without
        # blocking. Returns 0 on a clean quit.
        QtCore.QTimer.singleShot(0, app.quit)

    return int(app.exec())
