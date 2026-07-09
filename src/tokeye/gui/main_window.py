"""The main application window: top-bar nav, stacked views, status bar.

Chrome only. The two real views (spectrogram, modespec) are self-contained
widgets swapped in a ``QStackedWidget``; this window owns the shared
``QThreadPool`` and the status-bar readout/progress that views drive.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from tokeye.gui.model_service import ModelService
from tokeye.gui.widgets.modespec_view import ModespecView
from tokeye.gui.widgets.spectrogram_view import SpectrogramView

_VIEW_ORDER = ("spectrogram", "modespec")
_VIEW_TITLES = {"spectrogram": "Spectrogram", "modespec": "Modespec"}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        initial_view: str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("MainWindow")
        self.setWindowTitle("TokEye — DIII-D")
        self.resize(1360, 860)
        self.setMinimumSize(980, 640)

        # Shared worker pool + lazy model cache (views submit QRunnables here).
        self.pool = QtCore.QThreadPool.globalInstance()
        self.model_service = ModelService()

        # Stack must exist before the nav (its segments switch the stack).
        self._stack = QtWidgets.QStackedWidget()
        self._stack.setObjectName("Stack")

        root = QtWidgets.QWidget()
        root.setObjectName("Root")
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_topbar())
        outer.addWidget(self._stack, 1)
        self.setCentralWidget(root)

        self._build_statusbar()

        self._views: dict[str, int] = {}
        self._add_views()

        key = initial_view if initial_view in self._views else _VIEW_ORDER[0]
        self.show_view(key)

    # ------------------------------------------------------------------ build
    def _build_topbar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QFrame()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(52)
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(14)

        brand = QtWidgets.QLabel('<span style="color:#45b8cb;">Tok</span>Eye')
        brand.setObjectName("Brand")
        lay.addWidget(brand)

        lay.addSpacing(6)
        lay.addWidget(self._build_segnav())
        lay.addStretch(1)

        self._shot_status = QtWidgets.QLabel("no shot loaded")
        self._shot_status.setObjectName("ShotStatus")
        lay.addWidget(self._shot_status)
        return bar

    def _build_segnav(self) -> QtWidgets.QWidget:
        seg = QtWidgets.QFrame()
        seg.setObjectName("SegNav")
        lay = QtWidgets.QHBoxLayout(seg)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(0)

        self._nav = QtWidgets.QButtonGroup(self)
        self._nav.setExclusive(True)
        for i, key in enumerate(_VIEW_ORDER):
            btn = QtWidgets.QPushButton(_VIEW_TITLES[key])
            btn.setCheckable(True)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            lay.addWidget(btn)
            self._nav.addButton(btn, i)
        self._nav.idClicked.connect(self._stack.setCurrentIndex)
        return seg

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        sb.setObjectName("StatusBar")
        sb.setSizeGripEnabled(False)

        self._readout = QtWidgets.QLabel("—")
        self._readout.setObjectName("Readout")
        sb.addWidget(self._readout, 1)

        self._status_msg = QtWidgets.QLabel("")
        self._status_msg.setObjectName("StatusMsg")
        sb.addPermanentWidget(self._status_msg)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setObjectName("Progress")
        self._progress.setFixedWidth(200)
        self._progress.setTextVisible(False)
        self._progress.hide()
        sb.addPermanentWidget(self._progress)

    def _add_views(self) -> None:
        self.spectrogram_view = SpectrogramView(self)
        self.modespec_view = ModespecView(self)
        self._register_view("spectrogram", self.spectrogram_view)
        self._register_view("modespec", self.modespec_view)

    def _register_view(self, key: str, widget: QtWidgets.QWidget) -> None:
        self._views[key] = self._stack.addWidget(widget)

    # ------------------------------------------------------------------- API
    def show_view(self, key: str) -> None:
        """Switch to a view by key and sync the nav segment."""
        if key not in self._views:
            return
        idx = self._views[key]
        self._stack.setCurrentIndex(idx)
        btn = self._nav.button(idx)
        if btn is not None:
            btn.setChecked(True)

    def set_readout(self, text: str) -> None:
        """Set the monospace crosshair readout (left of the status bar)."""
        self._readout.setText(text or "—")

    def set_shot_status(self, text: str) -> None:
        """Set the shot label at the right of the top bar."""
        self._shot_status.setText(text)

    def begin_busy(self, message: str, *, determinate: bool = False) -> None:
        """Show a busy message + progress bar (indeterminate unless stated)."""
        self._status_msg.setText(message)
        if determinate:
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
        else:
            self._progress.setRange(0, 0)  # animated "busy" barber-pole
        self._progress.show()

    def set_progress(self, fraction: float) -> None:
        """Update determinate progress in ``[0, 1]``."""
        self._progress.setRange(0, 100)
        self._progress.setValue(max(0, min(100, round(fraction * 100))))

    def end_busy(self, message: str = "") -> None:
        """Hide the progress bar and set a final (optional) status message."""
        self._status_msg.setText(message)
        self._progress.hide()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
