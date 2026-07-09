"""``python -m tokeye.gui`` — launch the native desktop GUI."""

from __future__ import annotations

import sys

from tokeye.gui.app import run

if __name__ == "__main__":
    sys.exit(run())
