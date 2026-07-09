"""View A — single-probe spectrogram + TokEye overlay.

A control rail (model / shot / diagnostic / STFT / view + Load & Analyze) sits
left of a plot pane = [toolbar | :class:`SpectrogramCanvas`]. Blocking work
(MDSplus fetch, torch inference) runs in ``QRunnable`` workers on the window's
thread pool; each carries a monotonic request id so a re-click drops the stale
in-flight result. View/threshold/band tweaks re-render locally from cached state
(no recompute), preserving the current zoom.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from tokeye.gui.model_service import ModelService
from tokeye.gui.render import (
    freq_crop_rows,
    is_rgb_view,
    spectrogram_rect,
    view_display_array,
)
from tokeye.gui.widgets.controls import (
    DiagnosticProbe,
    ModelSelector,
    ShotField,
    StftControls,
    ViewControls,
)
from tokeye.gui.widgets.plot_items import SpectrogramCanvas
from tokeye.gui.workers import (
    AnalyzeWorker,
    BoundsWorker,
    FetchSpectrogramWorker,
    LatestShotWorker,
)


class SpectrogramView(QtWidgets.QWidget):
    def __init__(
        self,
        window: QtWidgets.QMainWindow | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._window = window
        self._pool = getattr(window, "pool", None) or QtCore.QThreadPool.globalInstance()
        self._service = getattr(window, "model_service", None) or ModelService()

        # cached state
        self._spec: np.ndarray | None = None
        self._stft_meta: dict | None = None
        self._infer: np.ndarray | None = None

        # request bookkeeping (stale-drop)
        self._counter = 0
        self._fetch_id = -1
        self._analyze_id = -1
        self._bounds_id = -1
        self._latest_id = -1
        self._workers: dict[int, object] = {}

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_rail())
        splitter.addWidget(self._build_plot_pane())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1000])

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(splitter)

        self.canvas.cursorMoved.connect(self._on_cursor)
        self._wire_controls()
        self._update_analyze_enabled()

    # ------------------------------------------------------------- build UI
    def _build_rail(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("ControlScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(460)

        rail = QtWidgets.QWidget()
        rail.setObjectName("ControlRail")
        v = QtWidgets.QVBoxLayout(rail)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(12)

        self.model_sel = ModelSelector()
        self.shot_field = ShotField()
        self.diag_probe = DiagnosticProbe()
        self.stft = StftControls()
        self.view_controls = ViewControls()

        self.load_btn = QtWidgets.QPushButton("Load shot")
        self.load_btn.setObjectName("PrimaryButton")
        self.load_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.analyze_btn = QtWidgets.QPushButton("Analyze")
        self.analyze_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        v.addWidget(self.model_sel)
        v.addWidget(self.shot_field)
        v.addWidget(self.diag_probe)
        v.addWidget(self.stft)
        v.addWidget(self.load_btn)
        v.addWidget(self.view_controls)
        v.addWidget(self.analyze_btn)
        v.addStretch(1)

        scroll.setWidget(rail)
        return scroll

    def _build_plot_pane(self) -> QtWidgets.QWidget:
        pane = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(pane)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.canvas = SpectrogramCanvas()  # first: toolbar handlers reference it
        v.addWidget(self._build_plot_toolbar())
        v.addWidget(self.canvas, 1)
        return pane

    def _build_plot_toolbar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QFrame()
        bar.setObjectName("PlotToolbar")
        bar.setFixedHeight(40)
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(12, 4, 12, 4)
        lay.setSpacing(6)

        self._pan_btn = QtWidgets.QPushButton("Pan")
        self._zoom_btn = QtWidgets.QPushButton("Zoom box")
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for b in (self._pan_btn, self._zoom_btn):
            b.setCheckable(True)
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            group.addButton(b)
            lay.addWidget(b)
        self._pan_btn.setChecked(True)
        self._pan_btn.toggled.connect(self._on_mode_toggled)

        reset = QtWidgets.QPushButton("Reset view")
        reset.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        reset.clicked.connect(lambda: self.canvas.reset_view())
        lay.addSpacing(8)
        lay.addWidget(reset)
        lay.addStretch(1)

        self._raw_btn = QtWidgets.QPushButton("Raw signal")
        self._raw_btn.setCheckable(True)
        self._raw_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._raw_btn.setEnabled(False)
        self._raw_btn.toggled.connect(self.canvas.set_raw_visible)
        lay.addWidget(self._raw_btn)
        return bar

    def _wire_controls(self) -> None:
        self.load_btn.clicked.connect(self._load)
        self.analyze_btn.clicked.connect(self._analyze)
        self.shot_field.shotChanged.connect(self._autofill_bounds)
        self.shot_field.latestRequested.connect(self._fetch_latest_shot)
        self.diag_probe.probeChanged.connect(lambda *_: self._autofill_bounds())
        self.stft.bandChanged.connect(lambda: self._render_current(reset_view=True))
        self.view_controls.changed.connect(lambda: self._render_current(reset_view=False))

    def _on_mode_toggled(self, pan_checked: bool) -> None:
        self.canvas.set_mouse_mode(rect_mode=not pan_checked)

    # ------------------------------------------------------------- requests
    def _next_id(self) -> int:
        self._counter += 1
        return self._counter

    def _submit(self, worker) -> None:
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(self._on_error)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_finished)
        self._workers[worker.request_id] = worker
        self._pool.start(worker)

    def _load(self) -> None:
        shot = self.shot_field.shot()
        pointname = self.diag_probe.pointname()
        if not shot or not pointname:
            self._notify("Enter a shot number and pick a diagnostic / probe first.")
            return
        rid = self._next_id()
        self._fetch_id = rid
        self.load_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self._begin_busy(f"Fetching {shot} / {pointname} …")
        self._submit(
            FetchSpectrogramWorker(
                rid, shot, pointname, self.shot_field.tlim(),
                self.stft.decimation(), self.stft.params(),
            )
        )

    def _analyze(self) -> None:
        if self._spec is None:
            self._notify("Load a shot first.")
            return
        rid = self._next_id()
        self._analyze_id = rid
        self.analyze_btn.setEnabled(False)
        self._begin_busy("Running model …")
        self._submit(AnalyzeWorker(rid, self._service, self.model_sel.name(), self._spec))

    def _autofill_bounds(self) -> None:
        shot = self.shot_field.shot()
        pointname = self.diag_probe.pointname()
        if not shot or not pointname:
            return
        rid = self._next_id()
        self._bounds_id = rid
        self._submit(BoundsWorker(rid, shot, pointname))

    def _fetch_latest_shot(self) -> None:
        rid = self._next_id()
        self._latest_id = rid
        self._begin_busy("Finding latest shot …")
        self._submit(LatestShotWorker(rid))

    # -------------------------------------------------------------- results
    def _on_result(self, rid: int, payload) -> None:
        if rid == self._fetch_id:
            self._apply_fetch(payload)
        elif rid == self._analyze_id:
            self._apply_analyze(payload)
        elif rid == self._bounds_id and payload:
            self.shot_field.set_time_bounds(payload[0], payload[1])
        elif rid == self._latest_id and payload:
            self.shot_field.set_shot(int(payload))
            self._autofill_bounds()
        # anything else is stale -> ignore

    def _apply_fetch(self, payload: dict) -> None:
        self._spec = payload["spec"]
        self._stft_meta = payload["meta"]
        self._infer = None
        t, x = payload.get("t"), payload.get("x")
        if t is not None and x is not None and np.size(t):
            self.set_raw_signal(t, x)
        self._render_current(reset_view=True)
        self._update_analyze_enabled()
        shot = self.shot_field.shot()
        self._set_shot_status(f"{shot} / {self.diag_probe.pointname()}")

    def _apply_analyze(self, out) -> None:
        self._infer = np.asarray(out) if out is not None else None
        if self._infer is None:
            self._notify("Inference returned no result.")
            return
        self._render_current(reset_view=False)

    def _on_error(self, rid: int, message: str) -> None:
        if rid in (self._fetch_id, self._analyze_id, self._latest_id):
            self._notify(message)

    def _on_progress(self, rid: int, fraction: float, message: str) -> None:
        if rid in (self._fetch_id, self._analyze_id) and self._window is not None:
            self._window.begin_busy(message, determinate=True)
            self._window.set_progress(fraction)

    def _on_finished(self, rid: int) -> None:
        self._workers.pop(rid, None)
        if rid == self._fetch_id or rid == self._latest_id:
            self.load_btn.setEnabled(True)
            self._end_busy()
        if rid == self._analyze_id:
            self._end_busy()
        self._update_analyze_enabled()

    # -------------------------------------------------------------- render
    def _meta_with_band(self) -> dict | None:
        if not self._stft_meta:
            return self._stft_meta
        meta = dict(self._stft_meta)
        fmin, fmax = self.stft.band()
        meta["fmin_khz"] = fmin
        meta["fmax_khz"] = fmax
        return meta

    def _render_current(self, reset_view: bool = False) -> None:
        if self._spec is None:
            return
        meta = self._meta_with_band()
        r_lo, r_hi = freq_crop_rows(self._spec.shape[0], meta)
        arr = self._spec[r_lo:r_hi]
        rect = spectrogram_rect(meta, arr.shape[0], arr.shape[1], r_lo)

        mode = self.view_controls.mode()
        extract = self._infer[:, r_lo:r_hi] if self._infer is not None else None
        disp = view_display_array(
            mode, arr, extract,
            self.view_controls.ch0(), self.view_controls.ch1(),
            self.view_controls.vmin(), self.view_controls.vmax(),
            self.view_controls.threshold(),
        )
        if disp is None:  # e.g. an overlay mode before Analyze -> plain spectrogram
            self.canvas.show_scalar(arr, rect, reset_view=reset_view)
            return
        if is_rgb_view(mode):
            self.canvas.show_rgb(disp, rect, sample_arr=arr, reset_view=reset_view)
        else:
            self.canvas.show_scalar(disp, rect, reset_view=reset_view)

    # ------------------------------------------------------------------ API
    def set_spectrogram(self, arr2d: np.ndarray, stft_meta: dict | None = None) -> None:
        """Set a spectrogram directly (used by tests / programmatic loads)."""
        self._spec = np.asarray(arr2d, dtype=float)
        self._stft_meta = stft_meta
        self._infer = None
        self._render_current(reset_view=True)
        self._update_analyze_enabled()

    def set_raw_signal(self, t_ms: np.ndarray, x: np.ndarray) -> None:
        self.canvas.set_raw(t_ms, x)
        self._raw_btn.setEnabled(True)
        self.canvas.set_raw_visible(self._raw_btn.isChecked())

    # ---------------------------------------------------------- misc/helpers
    def _update_analyze_enabled(self) -> None:
        self.analyze_btn.setEnabled(self._spec is not None and self._analyze_id
                                    not in self._workers)

    def _on_cursor(self, info: object) -> None:
        if self._window is None:
            return
        if info is None:
            self._window.set_readout("")
            return
        t, f, value = info
        text = f"t {t:9.1f} ms   f {f:8.1f} kHz"
        if value is not None:
            text += f"   x {value:+.3g}"
        self._window.set_readout(text)

    def _begin_busy(self, message: str) -> None:
        if self._window is not None:
            self._window.begin_busy(message)

    def _end_busy(self) -> None:
        if self._window is not None:
            self._window.end_busy()

    def _set_shot_status(self, text: str) -> None:
        if self._window is not None:
            self._window.set_shot_status(text)

    def _notify(self, message: str) -> None:
        self._end_busy()
        if self._window is not None:
            self._window.set_readout(message)
