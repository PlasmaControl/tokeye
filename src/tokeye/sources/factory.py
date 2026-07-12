"""Site-agnostic source selection (``$TOKEYE_SOURCE``).

The GUI, web tabs, and CLIs fetch signals through this factory instead of
constructing a concrete source, so a site branch (or a user, via the env var)
can swap the backend without touching consumers. Kinds:

- ``mds`` (aliases ``diiid``, ``d3d``) — DIII-D MDSplus via atlas.gat.com.
- ``foundation`` (aliases ``princeton``, ``h5``) — local
  ``{shot}_processed.h5`` files (Princeton ``foundation_model`` set).

``DEFAULT_SOURCE_KIND`` is the one intentional divergence point between site
branches (``mds`` here). Everything stays lazily imported: resolving a kind
touches only :mod:`os`, and the concrete source module loads on first use.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokeye.sources.base import SignalSource
    from tokeye.sources.presets import Diagnostic

DEFAULT_SOURCE_KIND = "mds"

_KIND_ALIASES = {
    "mds": "mds",
    "diiid": "mds",
    "d3d": "mds",
    "foundation": "foundation",
    "princeton": "foundation",
    "h5": "foundation",
}

_LABELS = {"mds": "DIII-D (MDSplus)", "foundation": "Princeton foundation_model"}


def _canonical(kind: str) -> str:
    canon = _KIND_ALIASES.get(kind.strip().lower())
    if canon is None:
        raise ValueError(
            f"unknown TOKEYE_SOURCE {kind!r} "
            f"(choices: {', '.join(sorted(_KIND_ALIASES))})"
        )
    return canon


def source_kind() -> str:
    """Canonical active source kind (``mds`` or ``foundation``)."""
    return _canonical(os.environ.get("TOKEYE_SOURCE", DEFAULT_SOURCE_KIND))


def source_label() -> str:
    """Human-readable name of the active source (window titles, help text)."""
    return _LABELS[source_kind()]


def get_source_class(kind: str | None = None) -> type[SignalSource]:
    """The :class:`SignalSource` class for ``kind`` (default: active kind)."""
    canon = _canonical(kind) if kind is not None else source_kind()
    if canon == "mds":
        from tokeye.sources.mds import MDSSource

        return MDSSource
    try:
        from tokeye.sources.foundation import FoundationSource
    except ImportError as exc:
        raise ImportError(
            "the 'foundation' source is not available on this branch/install "
            "(tokeye.sources.foundation missing — it ships on the princeton "
            "branch); unset TOKEYE_SOURCE or set it to 'mds'"
        ) from exc
    return FoundationSource


def get_source(kind: str | None = None, **kwargs) -> SignalSource:
    """A ready-to-``fetch`` source instance for ``kind`` (default: active)."""
    return get_source_class(kind)(**kwargs)


def latest_shot() -> int | None:
    """Most recent shot number from the active source, or ``None``."""
    return get_source_class().latest_shot()


def time_bounds(shot: int, pointname: str) -> tuple[float, float] | None:
    """``(t0_ms, t1_ms)`` for a shot+pointname from the active source."""
    return get_source_class().time_bounds(int(shot), str(pointname))


def active_diagnostics() -> dict[str, Diagnostic]:
    """Diagnostic presets matching the active source."""
    if source_kind() == "foundation":
        try:
            from tokeye.sources.foundation_presets import FOUNDATION_DIAGNOSTICS
        except ImportError as exc:
            raise ImportError(
                "the 'foundation' source is not available on this "
                "branch/install (tokeye.sources.foundation_presets missing — "
                "it ships on the princeton branch); unset TOKEYE_SOURCE or "
                "set it to 'mds'"
            ) from exc

        return FOUNDATION_DIAGNOSTICS
    from tokeye.sources.presets import DIAGNOSTICS

    return DIAGNOSTICS


def active_dropdown_choices() -> list[tuple[str, str]]:
    """``(label, key)`` dropdown pairs for the active source's diagnostics."""
    return [(diag.label, diag.key) for diag in active_diagnostics().values()]


def default_diag_key() -> str:
    """Diagnostic preselected in the UI (first preset of the active source)."""
    return next(iter(active_diagnostics()))
