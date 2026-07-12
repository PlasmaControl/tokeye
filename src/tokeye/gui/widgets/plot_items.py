"""Reusable pyqtgraph pieces for the spectrogram canvas.

``SpectrogramCanvas`` wraps a pyqtgraph ``GraphicsLayoutWidget`` with an image
on a real time-ms / frequency-kHz grid, an interactive colour bar, a linked raw
strip (hidden until toggled), and a crosshair readout. Pan / wheel-zoom /
rubber-band come from pyqtgraph's ``ViewBox`` for free; the extra affordances
(mode toggle, double-click reset, crosshair) are wired here.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

_CROSSHAIR_PEN = pg.mkPen(color="#45b8cb", width=1)
_RAW_PEN = pg.mkPen(color="#8b93a1", width=1)


def _configure_axes(plot: pg.PlotItem, x_label: str, y_label: str) -> None:
    """Real ms/kHz axes: no SI-prefix rescaling, aspect free, no grid."""
    plot.setLabel("bottom", x_label)
    plot.setLabel("left", y_label)
    plot.getAxis("bottom").enableAutoSIPrefix(False)
    plot.getAxis("left").enableAutoSIPrefix(False)
    plot.setAspectLocked(False)
    plot.showGrid(x=False, y=False)
    plot.getViewBox().setDefaultPadding(0.01)


class SpectrogramCanvas(pg.GraphicsLayoutWidget):
    """An image + colour bar + optional raw strip on shared real axes."""

    #: Emits ``(t_ms, f_kHz, value_or_None)`` while the cursor is over the image,
    #: or ``None`` when it leaves.
    cursorMoved = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(360)

        # Row 0 reserved for the (initially hidden) raw-signal strip; the
        # spectrogram lives on row 1 with the colour bar inserted to its right.
        self._raw_plot: pg.PlotItem | None = None
        self._spec_plot = self.addPlot(row=1, col=0)
        _configure_axes(self._spec_plot, "Time (ms)", "Frequency (kHz)")

        self._img = pg.ImageItem()
        self._spec_plot.addItem(self._img)

        self._cbar = pg.ColorBarItem(
            interactive=True,
            colorMap=pg.colormap.getFromMatplotlib("gist_heat"),
        )
        self._cbar.setImageItem(self._img, insert_in=self._spec_plot)

        # Crosshair (ignoreBounds so it never affects auto-range).
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=_CROSSHAIR_PEN)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=_CROSSHAIR_PEN)
        for line in (self._vline, self._hline):
            line.setVisible(False)
            self._spec_plot.addItem(line, ignoreBounds=True)

        self._rect: tuple[float, float, float, float] | None = None
        self._sample_arr: np.ndarray | None = None  # 2-D scalar for readout
        self._equal_pixels = False  # square data pixels (long strip) when True

        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    # ------------------------------------------------------------- rendering
    def show_scalar(
        self,
        arr2d: np.ndarray,
        rect: tuple[float, float, float, float],
        levels: tuple[float, float] | None = None,
        reset_view: bool = True,
    ) -> None:
        """Show a scalar image with the gist_heat colour bar."""
        from tokeye.gui.render import auto_levels

        arr2d = np.asarray(arr2d, dtype=float)
        self._img.setImage(arr2d, autoLevels=False)
        self._img.setRect(QtCore.QRectF(*rect))
        lo, hi = levels if levels is not None else auto_levels(arr2d)
        self._cbar.setLevels((lo, hi))
        self._cbar.setVisible(True)
        self._rect = tuple(float(v) for v in rect)
        self._sample_arr = arr2d
        if self._equal_pixels:
            self._apply_equal_pixels(frame=reset_view)
        elif reset_view:
            self._spec_plot.autoRange()

    def show_rgb(
        self,
        arr3d: np.ndarray,
        rect: tuple[float, float, float, float],
        sample_arr: np.ndarray | None = None,
        reset_view: bool = True,
    ) -> None:
        """Show an RGB overlay (colour bar hidden; no LUT)."""
        arr3d = np.asarray(arr3d, dtype=float)
        self._cbar.setVisible(False)
        self._img.setImage(arr3d, autoLevels=False, levels=(0.0, 1.0))
        self._img.setRect(QtCore.QRectF(*rect))
        self._rect = tuple(float(v) for v in rect)
        self._sample_arr = (
            np.asarray(sample_arr, dtype=float) if sample_arr is not None else None
        )
        if self._equal_pixels:
            self._apply_equal_pixels(frame=reset_view)
        elif reset_view:
            self._spec_plot.autoRange()

    def set_raw(self, t_ms: np.ndarray, x: np.ndarray) -> None:
        """Populate (and reveal) the raw-signal strip, x-linked to the image."""
        if self._raw_plot is None:
            self._raw_plot = self.addPlot(row=0, col=0)
            self._raw_plot.setLabel("left", "raw")
            self._raw_plot.getAxis("left").enableAutoSIPrefix(False)
            self._raw_plot.setXLink(self._spec_plot)  # shared time axis
            self._raw_plot.hideAxis("bottom")  # spectrogram below carries the ms axis
            self._raw_plot.setMaximumHeight(120)
            self._raw_plot.showGrid(x=False, y=False)
        self._raw_plot.clear()
        self._raw_plot.plot(np.asarray(t_ms), np.asarray(x), pen=_RAW_PEN)
        self._raw_plot.setVisible(True)

    def set_raw_visible(self, visible: bool) -> None:
        if self._raw_plot is not None:
            self._raw_plot.setVisible(visible)

    def has_raw(self) -> bool:
        return self._raw_plot is not None

    # --------------------------------------------------------- interactions
    def set_mouse_mode(self, rect_mode: bool) -> None:
        """Toggle left-drag between rubber-band zoom (True) and pan (False)."""
        vb = self._spec_plot.getViewBox()
        vb.setMouseMode(pg.ViewBox.RectMode if rect_mode else pg.ViewBox.PanMode)

    def set_equal_pixels(self, enabled: bool) -> None:
        """Square data pixels: one STFT column as wide as one bin is tall.

        A long shot then renders as a long strip — full frequency band tall,
        panned/zoomed along time — instead of being squeezed to the window
        width. Off restores the fit-to-window autorange.
        """
        self._equal_pixels = bool(enabled)
        if self._equal_pixels:
            self._apply_equal_pixels(frame=True)
        else:
            self._spec_plot.getViewBox().setAspectLocked(False)
            self._spec_plot.autoRange()

    def _apply_equal_pixels(self, frame: bool = False) -> None:
        """Lock screen-square data pixels; optionally frame the strip start.

        pyqtgraph's aspect ``ratio`` is xScale/yScale (screen px per x-unit
        over screen px per y-unit); squares need ``sx*dx == sy*dy`` for data
        pixels ``dx`` ms wide and ``dy`` kHz tall, i.e. ``ratio = dy/dx``.
        """
        img = self._img.image
        if img is None or self._rect is None:
            return
        n_rows, n_cols = int(img.shape[0]), int(img.shape[1])
        x0, y0, w, h = self._rect
        if not (n_rows and n_cols and w > 0 and h > 0):
            return
        dx = w / n_cols
        dy = h / n_rows
        vb = self._spec_plot.getViewBox()
        vb.setAspectLocked(True, ratio=dy / dx)
        if frame:
            # Full band tall; the lock derives the visible time span. Anchor
            # the window at the start of the data (setXRange keeps the span
            # consistent with the lock, so y stays the full band).
            vb.setYRange(y0, y0 + h, padding=0)
            span = vb.viewRange()[0]
            vb.setXRange(x0, x0 + (span[1] - span[0]), padding=0)

    def equal_pixels(self) -> bool:
        return self._equal_pixels

    def reset_view(self) -> None:
        if self._equal_pixels:
            self._apply_equal_pixels(frame=True)
        else:
            self._spec_plot.autoRange()
        if self._raw_plot is not None:
            self._raw_plot.autoRange()

    # image accessors (used by tests / views)
    def image_item(self) -> pg.ImageItem:
        return self._img

    def spectrogram_plot(self) -> pg.PlotItem:
        return self._spec_plot

    # --------------------------------------------------------------- cursor
    def _on_mouse_moved(self, pos) -> None:
        if self._rect is None or not self._spec_plot.sceneBoundingRect().contains(pos):
            self._show_crosshair(False)
            self.cursorMoved.emit(None)
            return
        p = self._spec_plot.getViewBox().mapSceneToView(pos)
        t, f = p.x(), p.y()
        x0, y0, w, h = self._rect
        if not (x0 <= t <= x0 + w and y0 <= f <= y0 + h):
            self._show_crosshair(False)
            self.cursorMoved.emit(None)
            return
        self._vline.setPos(t)
        self._hline.setPos(f)
        self._show_crosshair(True)
        value = None
        if self._sample_arr is not None and w > 0 and h > 0:
            n_rows, n_cols = self._sample_arr.shape[:2]
            col = min(max(int((t - x0) / w * n_cols), 0), n_cols - 1)
            row = min(max(int((f - y0) / h * n_rows), 0), n_rows - 1)
            value = float(self._sample_arr[row, col])
        self.cursorMoved.emit((t, f, value))

    def _on_mouse_clicked(self, ev) -> None:
        if ev.double():
            self.reset_view()

    def _show_crosshair(self, visible: bool) -> None:
        self._vline.setVisible(visible)
        self._hline.setVisible(visible)
