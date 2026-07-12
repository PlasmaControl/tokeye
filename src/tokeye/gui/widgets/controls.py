"""Reusable control-rail widgets (model / shot / diagnostic / STFT / view).

Each is a self-contained ``QGroupBox`` exposing plain accessors and, where the
change should drive a live re-render, a Qt signal. Kept free of torch: the model
list comes from the (torch-free) registry and a local ``model/`` glob, so the
window opens without importing torch.
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from tokeye.sources.factory import (
    active_diagnostics,
    active_dropdown_choices,
    default_diag_key,
)
from tokeye.sources.presets import MIRNOV_TOROIDAL

VIEW_MODES = ["Enhanced", "Mask", "Amplitude", "Original"]
GATE_AVERAGE = "Array average"
GATE_REFERENCE = "Reference probe"

# Model names mirror tokeye.hub.MODEL_REGISTRY, hardcoded to keep the GUI
# torch-free at window-open: importing tokeye.hub eagerly pulls torch, and the
# real load is deferred to the first Analyze (in a worker). The combo is
# editable, so any other registry name or a local model path can be typed.
DEFAULT_MODEL = "big_tf_unet"
_REGISTRY_MODELS = ("big_tf_unet", "ae_tf_maskrcnn")


def _find_local_models() -> list[str]:
    """Local ``model/*.pt[2]`` paths (CWD-relative, as the app does today)."""
    d = Path("model")
    if not d.exists():
        return []
    out: list[str] = []
    for ext in (".pt", ".pt2"):
        out += [str(p) for p in d.glob(f"*{ext}")]
    return sorted(out)


class ModelSelector(QtWidgets.QGroupBox):
    def __init__(self) -> None:
        super().__init__("Model")
        form = QtWidgets.QFormLayout(self)
        self._combo = QtWidgets.QComboBox()
        self._combo.setEditable(True)  # allow a custom local path
        self._combo.addItems(list(_REGISTRY_MODELS) + _find_local_models())
        self._combo.setCurrentText(DEFAULT_MODEL)
        self._combo.setToolTip(
            "Built-in models download from Hugging Face on first Analyze "
            "(~30 MB, cached). Local model/*.pt files are also listed."
        )
        form.addRow("Analysis model", self._combo)

    def name(self) -> str:
        return self._combo.currentText().strip()


class ShotField(QtWidgets.QGroupBox):
    shotChanged = QtCore.Signal()
    latestRequested = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__("Shot")
        form = QtWidgets.QFormLayout(self)

        row = QtWidgets.QHBoxLayout()
        self._shot = QtWidgets.QSpinBox()
        self._shot.setRange(0, 9_999_999)
        self._shot.setSpecialValueText("—")  # 0 shows as em dash (no shot yet)
        self._shot.setValue(0)
        self._latest = QtWidgets.QPushButton("Latest")
        self._latest.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        row.addWidget(self._shot, 1)
        row.addWidget(self._latest)
        form.addRow("Shot", row)

        self._tmin = QtWidgets.QDoubleSpinBox()
        self._tmax = QtWidgets.QDoubleSpinBox()
        for sb, val in ((self._tmin, 0.0), (self._tmax, 0.0)):
            sb.setRange(-1e7, 1e7)
            sb.setDecimals(2)
            sb.setValue(val)
            sb.setSuffix(" ms")
        form.addRow("t min", self._tmin)
        form.addRow("t max", self._tmax)

        self._shot.editingFinished.connect(self.shotChanged)
        self._latest.clicked.connect(self.latestRequested)

    def shot(self) -> int | None:
        v = int(self._shot.value())
        return v or None

    def set_shot(self, value: int) -> None:
        self._shot.setValue(int(value))

    def tlim(self) -> tuple[float, float] | None:
        lo, hi = float(self._tmin.value()), float(self._tmax.value())
        return (lo, hi) if hi > lo else None

    def set_time_bounds(self, t0: float, t1: float) -> None:
        self._tmin.setValue(round(float(t0), 2))
        self._tmax.setValue(round(float(t1), 2))


class DiagnosticProbe(QtWidgets.QGroupBox):
    probeChanged = QtCore.Signal(str, str)  # diag_key, pointname

    def __init__(self, default_diag: str | None = None) -> None:
        super().__init__("Diagnostic")
        form = QtWidgets.QFormLayout(self)
        default_diag = default_diag or default_diag_key()

        self._diag = QtWidgets.QComboBox()
        for label, key in active_dropdown_choices():
            self._diag.addItem(label, key)
        self._probe = QtWidgets.QComboBox()
        self._probe.setEditable(True)  # custom pointnames allowed
        form.addRow("Diagnostic", self._diag)
        form.addRow("Probe", self._probe)

        idx = self._diag.findData(default_diag)
        self._diag.setCurrentIndex(idx if idx >= 0 else 0)
        self._repopulate_probes()

        self._diag.currentIndexChanged.connect(self._on_diag_changed)
        self._probe.currentTextChanged.connect(self._emit_changed)

    def _repopulate_probes(self) -> None:
        diag = active_diagnostics().get(self.diag_key())
        self._probe.blockSignals(True)
        self._probe.clear()
        if diag is not None:
            self._probe.addItems(list(diag.pointnames))
            self._probe.setCurrentText(diag.default)
        self._probe.blockSignals(False)

    def _on_diag_changed(self) -> None:
        self._repopulate_probes()
        self._emit_changed()

    def _emit_changed(self) -> None:
        self.probeChanged.emit(self.diag_key(), self.pointname())

    def diag_key(self) -> str:
        return self._diag.currentData() or default_diag_key()

    def pointname(self) -> str:
        return self._probe.currentText().strip()


class _SliderRow(QtWidgets.QWidget):
    """A labelled integer slider with a live value read-out."""

    valueChanged = QtCore.Signal(int)

    def __init__(self, lo: int, hi: int, value: int, fmt=str) -> None:
        super().__init__()
        self._fmt = fmt
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setRange(lo, hi)
        self._slider.setValue(value)
        self._label = QtWidgets.QLabel(fmt(value))
        self._label.setMinimumWidth(40)
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        lay.addWidget(self._slider, 1)
        lay.addWidget(self._label)
        self._slider.valueChanged.connect(self._on_change)

    def _on_change(self, v: int) -> None:
        self._label.setText(self._fmt(v))
        self.valueChanged.emit(v)

    def value(self) -> int:
        return self._slider.value()


class StftControls(QtWidgets.QGroupBox):
    bandChanged = QtCore.Signal()  # fmin/fmax edited (display-only re-crop)

    def __init__(self) -> None:
        super().__init__("STFT settings")
        form = QtWidgets.QFormLayout(self)

        self._n_fft = QtWidgets.QComboBox()
        self._n_fft.addItems(["256", "512", "1024", "2048"])
        self._n_fft.setCurrentText("1024")
        self._hop = QtWidgets.QComboBox()
        self._hop.addItems(["64", "128", "256", "512"])
        self._hop.setCurrentText("256")
        self._hop.setToolTip("UI default 256; the released model was trained at 128.")
        self._clip_dc = QtWidgets.QCheckBox("Remove DC bin")
        self._clip_dc.setChecked(True)

        self._clip_low = QtWidgets.QDoubleSpinBox()
        self._clip_low.setRange(0.0, 100.0)
        self._clip_low.setValue(1.0)
        self._clip_high = QtWidgets.QDoubleSpinBox()
        self._clip_high.setRange(0.0, 100.0)
        self._clip_high.setValue(99.0)

        self._fmin = QtWidgets.QDoubleSpinBox()
        self._fmax = QtWidgets.QDoubleSpinBox()
        for sb, val in ((self._fmin, 5.0), (self._fmax, 250.0)):
            sb.setRange(0.0, 1e6)
            sb.setSuffix(" kHz")
            sb.setValue(val)

        self._decim = QtWidgets.QSpinBox()
        self._decim.setRange(1, 100)
        self._decim.setValue(1)
        self._decim.setToolTip("Anti-aliased decimation applied on Load (1 = off).")

        form.addRow("FFT bins", self._n_fft)
        form.addRow("Hop", self._hop)
        form.addRow("f min", self._fmin)
        form.addRow("f max", self._fmax)
        form.addRow("Clip low %", self._clip_low)
        form.addRow("Clip high %", self._clip_high)
        form.addRow("Decimation", self._decim)
        form.addRow("", self._clip_dc)

        self._fmin.valueChanged.connect(self.bandChanged)
        self._fmax.valueChanged.connect(self.bandChanged)

    def params(self) -> dict:
        return {
            "n_fft": int(self._n_fft.currentText()),
            "hop": int(self._hop.currentText()),
            "clip_dc": self._clip_dc.isChecked(),
            "clip_low": self._clip_low.value(),
            "clip_high": self._clip_high.value(),
        }

    def decimation(self) -> int:
        return int(self._decim.value())

    def band(self) -> tuple[float | None, float | None]:
        fmin = self._fmin.value() or None
        fmax = self._fmax.value() or None
        return fmin, fmax


class ViewControls(QtWidgets.QGroupBox):
    changed = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__("View")
        form = QtWidgets.QFormLayout(self)

        self._mode = QtWidgets.QComboBox()
        self._mode.addItems(VIEW_MODES)
        form.addRow("Mode", self._mode)

        self._ch0 = QtWidgets.QCheckBox("Coherent")
        self._ch0.setChecked(True)
        self._ch1 = QtWidgets.QCheckBox("Transient")
        self._ch1.setChecked(True)
        ch_row = QtWidgets.QHBoxLayout()
        ch_row.setContentsMargins(0, 0, 0, 0)
        ch_row.addWidget(self._ch0)
        ch_row.addWidget(self._ch1)
        form.addRow("Channels", ch_row)

        self._vmin = _SliderRow(0, 100, 0, fmt=lambda v: f"{v}%")
        self._vmax = _SliderRow(0, 100, 100, fmt=lambda v: f"{v}%")
        self._thr = _SliderRow(0, 100, 50, fmt=lambda v: f"{v / 100:.2f}")
        form.addRow("Min clip", self._vmin)
        form.addRow("Max clip", self._vmax)
        form.addRow("Threshold", self._thr)

        self._mode.currentTextChanged.connect(lambda _=None: self.changed.emit())
        for chk in (self._ch0, self._ch1):
            chk.toggled.connect(lambda _=None: self.changed.emit())
        for sld in (self._vmin, self._vmax, self._thr):
            sld.valueChanged.connect(lambda _=None: self.changed.emit())

    def mode(self) -> str:
        return self._mode.currentText()

    def ch0(self) -> bool:
        return self._ch0.isChecked()

    def ch1(self) -> bool:
        return self._ch1.isChecked()

    def vmin(self) -> float:
        return float(self._vmin.value())

    def vmax(self) -> float:
        return float(self._vmax.value())

    def threshold(self) -> float:
        return self._thr.value() / 100.0


class ReferenceProbe(QtWidgets.QGroupBox):
    """Toroidal-array reference probe (time-window autofill + gate reference)."""

    changed = QtCore.Signal()

    def __init__(self, default: str = "MPI66M067D") -> None:
        super().__init__("Reference probe")
        form = QtWidgets.QFormLayout(self)
        self._combo = QtWidgets.QComboBox()
        self._combo.setEditable(True)
        self._combo.addItems(list(MIRNOV_TOROIDAL))
        self._combo.setCurrentText(default)
        self._combo.setToolTip("Autofills the time window; used as the gate reference.")
        form.addRow("Probe", self._combo)
        self._combo.currentTextChanged.connect(lambda _=None: self.changed.emit())

    def pointname(self) -> str:
        return self._combo.currentText().strip()


class ModespecSettings(QtWidgets.QGroupBox):
    def __init__(self) -> None:
        super().__init__("Modespec settings")
        form = QtWidgets.QFormLayout(self)

        self._fmin = QtWidgets.QDoubleSpinBox()
        self._fmax = QtWidgets.QDoubleSpinBox()
        for sb, val in ((self._fmin, 5.0), (self._fmax, 200.0)):
            sb.setRange(0.0, 1e6)
            sb.setSuffix(" kHz")
            sb.setValue(val)
        self._nmin = QtWidgets.QSpinBox()
        self._nmin.setRange(-20, 20)
        self._nmin.setValue(-5)
        self._nmax = QtWidgets.QSpinBox()
        self._nmax.setRange(-20, 20)
        self._nmax.setValue(5)
        self._decim = QtWidgets.QSpinBox()
        self._decim.setRange(1, 200)
        self._decim.setValue(1)
        self._decim.setToolTip("Speeds up modespec; auto-raised to an f-max-safe value.")

        form.addRow("f min", self._fmin)
        form.addRow("f max", self._fmax)
        form.addRow("n min", self._nmin)
        form.addRow("n max", self._nmax)
        form.addRow("Decimation", self._decim)

    def f_min(self) -> float:
        return self._fmin.value()

    def f_max(self) -> float:
        return self._fmax.value()

    def n_min(self) -> int:
        return int(self._nmin.value())

    def n_max(self) -> int:
        return int(self._nmax.value())

    def decimation(self) -> int:
        return int(self._decim.value())


class GateControls(QtWidgets.QGroupBox):
    cohChanged = QtCore.Signal()  # coherence slider moved (local re-render)
    gateChanged = QtCore.Signal()  # gate on/off or source changed (local re-render)

    def __init__(self) -> None:
        super().__init__("Gate")
        form = QtWidgets.QFormLayout(self)

        self._gate = QtWidgets.QCheckBox("Gate with TokEye")
        self._gate.setChecked(True)
        form.addRow(self._gate)

        self._avg = QtWidgets.QRadioButton(GATE_AVERAGE)
        self._ref = QtWidgets.QRadioButton(GATE_REFERENCE)
        self._avg.setChecked(True)
        self._avg.setToolTip("Averages all probes; cancels single-probe artifacts.")
        src_row = QtWidgets.QVBoxLayout()
        src_row.setContentsMargins(0, 0, 0, 0)
        src_row.addWidget(self._avg)
        src_row.addWidget(self._ref)
        form.addRow("Source", src_row)

        self._coh = _SliderRow(0, 100, 50, fmt=lambda v: f"{v / 100:.2f}")
        form.addRow("Coherence", self._coh)

        self._coh.valueChanged.connect(lambda _=None: self.cohChanged.emit())
        self._gate.toggled.connect(lambda _=None: self.gateChanged.emit())
        self._avg.toggled.connect(lambda _=None: self.gateChanged.emit())

    def gate(self) -> bool:
        return self._gate.isChecked()

    def source(self) -> str:
        return "average" if self._avg.isChecked() else "reference"

    def coh(self) -> float:
        return self._coh.value() / 100.0
