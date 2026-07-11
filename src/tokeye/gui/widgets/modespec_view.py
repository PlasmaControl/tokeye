"""View B — toroidal mode-number map with optional band-matched TokEye gating.

The classic modespec analysis: fetch the toroidal Mirnov array, compute the
dominant toroidal mode ``n`` vs time/frequency, and draw it as a discrete-``n``
map (RdBu_r, integer colour bar, transparent where suppressed). The optional
gate runs TokEye on the *same* array (average or a reference probe) and keeps
only confirmed modes. The coherence slider re-gates locally from the cached
result — no recompute.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from tokeye import export
from tokeye.gui.model_service import ModelService
from tokeye.gui.render import (
    axis_rect,
    discrete_mode_image,
    mode_color_table,
    mode_ticks,
    nd_masked_for_display,
)
from tokeye.gui.widgets.controls import (
    GateControls,
    ModelSelector,
    ModespecSettings,
    ReferenceProbe,
    ShotField,
)
from tokeye.gui.widgets.plot_items import _CROSSHAIR_PEN, _configure_axes
from tokeye.gui.workers import BoundsWorker, ModespecWorker


class ModespecCanvas(pg.GraphicsLayoutWidget):
    """A discrete mode-``n`` RGBA image + integer colour bar + crosshair (reads n)."""

    cursorMoved = QtCore.Signal(object)  # (t_ms, f_kHz, n_or_None) or None

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(360)
        self._plot = self.addPlot(row=0, col=0)
        _configure_axes(self._plot, "Time (ms)", "Frequency (kHz)")
        self._img = pg.ImageItem()
        self._plot.addItem(self._img)

        self._cbar: pg.ColorBarItem | None = None
        self._n_range: tuple[int, int] | None = None

        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=_CROSSHAIR_PEN)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=_CROSSHAIR_PEN)
        for line in (self._vline, self._hline):
            line.setVisible(False)
            self._plot.addItem(line, ignoreBounds=True)

        self._rect: tuple[float, float, float, float] | None = None
        self._z: np.ndarray | None = None  # (n_freq, n_win) mode values for readout

        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    def show_modes(
        self,
        rgba: np.ndarray,
        rect: tuple[float, float, float, float],
        n_lo: int,
        n_hi: int,
        z_grid: np.ndarray,
        reset_view: bool = True,
    ) -> None:
        self._img.setImage(np.asarray(rgba), autoLevels=False)
        self._img.setRect(QtCore.QRectF(*rect))
        self._rect = tuple(float(v) for v in rect)
        self._z = np.asarray(z_grid, dtype=float)
        self._build_colorbar(int(n_lo), int(n_hi))
        if reset_view:
            self._plot.autoRange()

    def _build_colorbar(self, n_lo: int, n_hi: int) -> None:
        if self._n_range == (n_lo, n_hi) and self._cbar is not None:
            return
        table = mode_color_table(n_lo, n_hi)  # (ncolors, 3)
        ncolors = table.shape[0]
        rgba = np.concatenate(
            [table, np.full((ncolors, 1), 255, np.uint8)], axis=1
        ).astype(np.uint8)
        # step colour map: each mode owns a flat band (crisp, not a gradient)
        edges = np.linspace(0.0, 1.0, ncolors + 1)
        pos, cols = [], []
        for i in range(ncolors):
            pos += [edges[i], edges[i + 1]]
            cols += [rgba[i], rgba[i]]
        cmap = pg.ColorMap(np.array(pos), np.array(cols))

        if self._cbar is None:
            self._cbar = pg.ColorBarItem(
                interactive=False, colorMap=cmap, values=(n_lo - 0.5, n_hi + 0.5)
            )
            self.addItem(self._cbar, row=0, col=1)
        else:
            self._cbar.setColorMap(cmap)
            self._cbar.setLevels((n_lo - 0.5, n_hi + 0.5))
        axis = getattr(self._cbar, "axis", None)
        if axis is not None:
            axis.setTicks([mode_ticks(n_lo, n_hi)])
        self._n_range = (n_lo, n_hi)

    def set_mouse_mode(self, rect_mode: bool) -> None:
        vb = self._plot.getViewBox()
        vb.setMouseMode(pg.ViewBox.RectMode if rect_mode else pg.ViewBox.PanMode)

    def reset_view(self) -> None:
        self._plot.autoRange()

    def image_item(self) -> pg.ImageItem:
        return self._img

    def colorbar(self) -> pg.ColorBarItem | None:
        return self._cbar

    def _on_mouse_moved(self, pos) -> None:
        if self._rect is None or not self._plot.sceneBoundingRect().contains(pos):
            self._show_crosshair(False)
            self.cursorMoved.emit(None)
            return
        p = self._plot.getViewBox().mapSceneToView(pos)
        t, f = p.x(), p.y()
        x0, y0, w, h = self._rect
        if not (x0 <= t <= x0 + w and y0 <= f <= y0 + h):
            self._show_crosshair(False)
            self.cursorMoved.emit(None)
            return
        self._vline.setPos(t)
        self._hline.setPos(f)
        self._show_crosshair(True)
        n = None
        if self._z is not None and w > 0 and h > 0:
            n_rows, n_cols = self._z.shape
            col = min(max(int((t - x0) / w * n_cols), 0), n_cols - 1)
            row = min(max(int((f - y0) / h * n_rows), 0), n_rows - 1)
            n = self._z[row, col]
        self.cursorMoved.emit((t, f, n))

    def _on_mouse_clicked(self, ev) -> None:
        if ev.double():
            self.reset_view()

    def _show_crosshair(self, visible: bool) -> None:
        self._vline.setVisible(visible)
        self._hline.setVisible(visible)


class ModespecView(QtWidgets.QWidget):
    def __init__(
        self,
        window: QtWidgets.QMainWindow | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._window = window
        self._pool = getattr(window, "pool", None) or QtCore.QThreadPool.globalInstance()
        self._service = getattr(window, "model_service", None) or ModelService()

        self._result: dict | None = None
        self._tok_mask: np.ndarray | None = None
        self._gate_meta: dict | None = None

        self._counter = 0
        self._analyze_id = -1
        self._bounds_id = -1
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
        self._wire()

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
        self.ref_probe = ReferenceProbe()
        self.settings = ModespecSettings()
        self.gate_controls = GateControls()
        self.analyze_btn = QtWidgets.QPushButton("Analyze")
        self.analyze_btn.setObjectName("PrimaryButton")
        self.analyze_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        for w in (self.model_sel, self.shot_field, self.ref_probe, self.settings,
                  self.gate_controls, self.analyze_btn):
            v.addWidget(w)
        v.addStretch(1)

        scroll.setWidget(rail)
        return scroll

    def _build_plot_pane(self) -> QtWidgets.QWidget:
        pane = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(pane)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.canvas = ModespecCanvas()
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
        self._pan_btn.toggled.connect(
            lambda pan: self.canvas.set_mouse_mode(rect_mode=not pan)
        )
        reset = QtWidgets.QPushButton("Reset view")
        reset.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        reset.clicked.connect(lambda: self.canvas.reset_view())
        lay.addSpacing(8)
        lay.addWidget(reset)
        lay.addStretch(1)

        self._save_btn = QtWidgets.QPushButton("Save…")
        self._save_btn.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton))
        self._save_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_dialog)
        lay.addWidget(self._save_btn)
        return bar

    def _wire(self) -> None:
        self.analyze_btn.clicked.connect(self._analyze)
        self.shot_field.shotChanged.connect(self._autofill_bounds)
        self.ref_probe.changed.connect(self._autofill_bounds)
        self.gate_controls.cohChanged.connect(lambda: self._render_modes(reset_view=False))
        self.gate_controls.gateChanged.connect(lambda: self._render_modes(reset_view=False))

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

    def _analyze(self) -> None:
        shot = self.shot_field.shot()
        if not shot:
            self._notify("Enter a shot number first.")
            return
        rid = self._next_id()
        self._analyze_id = rid
        self.analyze_btn.setEnabled(False)
        self._begin_busy("Fetching Mirnov array …")
        params = {
            "f_min": self.settings.f_min(),
            "f_max": self.settings.f_max(),
            "n_min": self.settings.n_min(),
            "n_max": self.settings.n_max(),
            "decimation": self.settings.decimation(),
        }
        gate_cfg = {
            "gate": self.gate_controls.gate(),
            "source": self.gate_controls.source(),
            "coh": self.gate_controls.coh(),
        }
        self._submit(
            ModespecWorker(
                rid, shot, self.ref_probe.pointname(), self.shot_field.tlim(),
                params, gate_cfg, self._service, self.model_sel.name(),
            )
        )

    def _autofill_bounds(self) -> None:
        shot = self.shot_field.shot()
        probe = self.ref_probe.pointname()
        if not shot or not probe:
            return
        rid = self._next_id()
        self._bounds_id = rid
        self._submit(BoundsWorker(rid, shot, probe))

    # -------------------------------------------------------------- results
    def _on_result(self, rid: int, payload) -> None:
        if rid == self._analyze_id and isinstance(payload, dict):
            self._result = payload["result"]
            self._tok_mask = payload["tok_mask"]
            self._gate_meta = payload["gate_meta"]
            self._render_modes(reset_view=True)
            self._save_btn.setEnabled(self._result is not None)
            self._set_shot_status(f"{self.shot_field.shot()}  toroidal modespec")
        elif rid == self._bounds_id and payload:
            self.shot_field.set_time_bounds(payload[0], payload[1])

    def _on_error(self, rid: int, message: str) -> None:
        if rid == self._analyze_id:
            self._notify(message)

    def _on_progress(self, rid: int, fraction: float, message: str) -> None:
        if rid == self._analyze_id and self._window is not None:
            self._window.begin_busy(message, determinate=True)
            self._window.set_progress(fraction)

    def _on_finished(self, rid: int) -> None:
        self._workers.pop(rid, None)
        if rid == self._analyze_id:
            self.analyze_btn.setEnabled(True)
            self._end_busy()

    # -------------------------------------------------------------- render
    def _gated_nd(self, coh: float) -> np.ndarray | None:
        """Dominant-mode array gated by the cached TokEye mask, or None (ungated).

        Shared by ``_render_modes`` and ``export_npz`` so display and export
        never diverge. Falls back to ungated (returns None) on any gating
        failure — callers are expected to just render/export ungated in that
        case, same as before this was factored out.
        """
        if not (
            self.gate_controls.gate()
            and self._tok_mask is not None
            and self._gate_meta is not None
        ):
            return None
        from tokeye.sources.mirnov import gate_dominant_mask

        try:
            return gate_dominant_mask(
                self._result, self._tok_mask, self._gate_meta, coh_thresh=coh
            )
        except Exception as exc:  # noqa: BLE001 - fall back to ungated on any failure
            self._notify(f"TokEye gate failed (showing ungated): {exc}")
            return None

    def _render_modes(self, reset_view: bool = False) -> None:
        if self._result is None:
            return
        coh = self.gate_controls.coh()
        nd = self._gated_nd(coh)
        nd_masked = nd_masked_for_display(self._result, nd, coh)
        n_lo, n_hi = (int(v) for v in self._result["n_range"])
        rgba = discrete_mode_image(nd_masked, n_lo, n_hi)
        rect = axis_rect(self._result["t_win_ms"], self._result["freq_khz"])
        self.canvas.show_modes(rgba, rect, n_lo, n_hi, nd_masked.T, reset_view=reset_view)

    # -------------------------------------------------------------- exporting
    def export_png(self, path: str | Path) -> Path:
        """Save a PNG of the exact current canvas (colour bar, current zoom)."""
        if not self.canvas.grab().save(str(path)):
            raise RuntimeError(f"Could not write PNG to {path}")
        return Path(path)

    def export_npz(self, path: str | Path) -> Path:
        """Save the cached mode-spectrogram result as a tokeye-modespec npz."""
        if self._result is None:
            raise ValueError("Nothing to save — run Analyze first.")
        coh = self.gate_controls.coh()
        nd = self._gated_nd(coh)

        params = {
            "shot": self.shot_field.shot(),
            "ref_probe": self.ref_probe.pointname(),
            "model": self.model_sel.name(),
            "f_min": self.settings.f_min(),
            "f_max": self.settings.f_max(),
            "n_min": self.settings.n_min(),
            "n_max": self.settings.n_max(),
            "decimation": self.settings.decimation(),
            "gate": self.gate_controls.gate(),
            "gate_source": self.gate_controls.source(),
            "coh": coh,
        }
        bundle = export.modespec_bundle(
            result=self._result,
            nd=nd,
            tok_mask=self._tok_mask,
            coh_thresh=float(coh),
            params=params,
            source="gui-modespec",
        )
        return export.save_npz(path, bundle)

    def export_csv(self, path: str | Path) -> Path:
        """Save the detected modes as a vendored-compatible ``_modes.csv``."""
        if self._result is None:
            raise ValueError("Nothing to save — run Analyze first.")
        text = export.modes_csv_text(
            self._result,
            array="toroidal",
            f_min=float(self.settings.f_min()),
            f_max=float(self.settings.f_max()),
        )
        Path(path).write_text(text)
        return Path(path)

    def _save_dialog(self) -> None:
        default = f"{self.shot_field.shot() or 'tokeye'}_modespec"
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save modespec", default, "PNG + NPZ + CSV (*.png *.npz *.csv)"
        )
        if not fname:
            return
        base = Path(fname).with_suffix("")
        try:
            png = self.export_png(base.with_suffix(".png"))
            npz = self.export_npz(base.with_suffix(".npz"))
        except Exception as exc:  # noqa: BLE001 - never lose work to a silent crash
            self._notify(f"Save failed: {exc}")
            return
        msg = f"Saved {png.name} + {npz.name}"
        try:
            csv_path = self.export_csv(base.with_name(f"{base.name}_modes.csv"))
            msg += f" + {csv_path.name}"
        except Exception as exc:  # noqa: BLE001 - CSV failure must not kill png/npz
            msg += f" (CSV failed: {exc})"
        self._notify(msg)

    # --------------------------------------------------------------- helpers
    def _on_cursor(self, info: object) -> None:
        if self._window is None:
            return
        if info is None:
            self._window.set_readout("")
            return
        t, f, n = info
        text = f"t {t:9.1f} ms   f {f:8.1f} kHz"
        text += f"   n {int(n):+d}" if n is not None and np.isfinite(n) else "   n —"
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
