"""Theme: Fusion base + dark palette + flat-dark QSS, and pyqtgraph config.

Chrome stays cool and quiet so the warm spectrogram data is the visual hero.
Colours live in :data:`COLORS`; the polished stylesheet is
``resources/dark.qss`` (with a compact inline fallback here so the app still
renders dark if the data file is missing from a wheel).
"""

from __future__ import annotations

import importlib.resources as _res

# Cool "control-room" chrome. The accent is a muted diagnostic cyan, chosen to
# sit away from both data colormaps (warm gist_heat, diverging RdBu_r).
COLORS = {
    "bg": "#13151a",  # window
    "panel": "#1b1e26",  # control surfaces / cards
    "panel2": "#22262f",  # raised / inputs base
    "plot": "#0c0d11",  # pyqtgraph canvas (darkest)
    "line": "#2a2f3a",  # hairline borders
    "text": "#e9ecf1",  # primary text
    "muted": "#8b93a1",  # labels / secondary
    "accent": "#45b8cb",  # interactive / selection
    "accentHi": "#63d0e2",  # hover accent
    "accentInk": "#08222a",  # text on an accent fill
}

FONT_STACK = ["Inter", "Segoe UI", "Cantarell", "DejaVu Sans"]
MONO_STACK = ["DejaVu Sans Mono", "Cascadia Mono", "Consolas", "monospace"]


def configure_pyqtgraph() -> None:
    """Global pyqtgraph config: row-major images, dark canvas, soft axes."""
    import pyqtgraph as pg

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    pg.setConfigOption("background", COLORS["plot"])
    pg.setConfigOption("foreground", "#aab2c0")


def apply_theme(app) -> None:
    """Apply the Fusion style, dark palette, base font, and QSS to ``app``."""
    from PySide6 import QtGui

    app.setStyle("Fusion")
    app.setPalette(_build_palette())

    font = QtGui.QFont()
    font.setFamilies(FONT_STACK)
    font.setPointSize(10)
    app.setFont(font)

    app.setStyleSheet(load_qss())


def load_qss() -> str:
    """Read the packaged ``dark.qss`` (fall back to the inline stylesheet)."""
    try:
        return (
            _res.files("tokeye.gui")
            .joinpath("resources", "dark.qss")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, ModuleNotFoundError):
        return _INLINE_QSS


def _build_palette():
    from PySide6.QtGui import QColor, QPalette

    c = COLORS
    role = QPalette.ColorRole
    p = QPalette()
    p.setColor(role.Window, QColor(c["bg"]))
    p.setColor(role.WindowText, QColor(c["text"]))
    p.setColor(role.Base, QColor(c["panel2"]))
    p.setColor(role.AlternateBase, QColor(c["panel"]))
    p.setColor(role.Text, QColor(c["text"]))
    p.setColor(role.Button, QColor(c["panel"]))
    p.setColor(role.ButtonText, QColor(c["text"]))
    p.setColor(role.BrightText, QColor("#ffffff"))
    p.setColor(role.ToolTipBase, QColor(c["panel2"]))
    p.setColor(role.ToolTipText, QColor(c["text"]))
    p.setColor(role.PlaceholderText, QColor(c["muted"]))
    p.setColor(role.Highlight, QColor(c["accent"]))
    p.setColor(role.HighlightedText, QColor(c["accentInk"]))
    p.setColor(role.Link, QColor(c["accent"]))

    dis = QPalette.ColorGroup.Disabled
    p.setColor(dis, role.Text, QColor(c["muted"]))
    p.setColor(dis, role.ButtonText, QColor("#5b626e"))
    p.setColor(dis, role.WindowText, QColor(c["muted"]))
    return p


# Compact safety net if resources/dark.qss is unavailable (e.g. a wheel that
# dropped package data). The full, polished stylesheet lives in that file.
_INLINE_QSS = """
QWidget { color: #e9ecf1; font-size: 13px; }
#TopBar, #StatusBar { background: #171a21; }
#TopBar { border-bottom: 1px solid #2a2f3a; }
#StatusBar { border-top: 1px solid #2a2f3a; }
QLineEdit, QComboBox, QAbstractSpinBox {
    background: #0f1115; border: 1px solid #2a2f3a; border-radius: 6px; padding: 5px 8px;
}
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus { border-color: #45b8cb; }
QPushButton {
    background: #22262f; border: 1px solid #2f3542; border-radius: 6px; padding: 6px 14px;
}
QPushButton:hover { background: #2a2f3a; }
QPushButton#PrimaryButton { background: #45b8cb; border-color: #45b8cb; color: #08222a; }
QProgressBar { background: #0f1115; border: 1px solid #2a2f3a; border-radius: 6px; }
QProgressBar::chunk { background: #45b8cb; border-radius: 5px; }
"""
