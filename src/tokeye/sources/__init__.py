"""Reusable signal-source layer (shot → 1-D signal) for the app and CLI.

Importing this package does not import MDSplus: :class:`MDSSource` defers the
(import-guarded) MDS fetchers until :meth:`MDSSource.fetch` is called.
"""

from __future__ import annotations

from .base import SignalSource
from .mds import (
    DEFAULT_ATLAS,
    DEFAULT_CACHE_ROOT,
    MDSSource,
    cache_root,
    latest_shot,
    time_bounds,
)
from .presets import DIAGNOSTICS, Diagnostic, diagnostic_dropdown_choices

__all__ = [
    "DEFAULT_ATLAS",
    "DEFAULT_CACHE_ROOT",
    "DIAGNOSTICS",
    "Diagnostic",
    "MDSSource",
    "SignalSource",
    "cache_root",
    "diagnostic_dropdown_choices",
    "latest_shot",
    "time_bounds",
]
