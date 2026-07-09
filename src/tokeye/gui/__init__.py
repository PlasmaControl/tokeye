"""TokEye native desktop GUI (PySide6 + pyqtgraph) for DIII-D.

A self-contained desktop front end for the DIII-D spectrogram + toroidal
modespec analyses, meant to run directly on the Omega cluster over
NoMachine / X11 (unlike the tunnelled Gradio web app).

Import-hygiene contract: this package must never be imported at
``tokeye``/``tokeye.cli`` module scope, and heavy GUI/torch imports stay local
to the functions that need them (``app.run`` defers PySide6; the model service
and workers defer torch). Launch via ``tokeye gui`` or ``python -m tokeye.gui``.
"""

from __future__ import annotations

__all__: list[str] = []
