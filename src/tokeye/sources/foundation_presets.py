"""Diagnostic presets for the Princeton ``foundation_model`` archive.

Pointnames are ``group/index`` (channel row in the shot file's ``ydata``).
The exporter stored rows in sorted-original-name order but did not record the
names, so **probe identity is unknown** — index pointnames are honest about
that, and the mode-number analyses that need probe geometry (modespec) are not
available from this source.

Channel counts below are the archive-wide common shapes (verified on a random
sample of shots); availability still varies per shot, so the app refreshes the
dropdown from :func:`signals_for_shot` when a shot is selected. Kept in a
separate module from the DIII-D ``presets`` so facility-branch merges never
conflict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tokeye.sources.foundation import list_signals
from tokeye.sources.presets import Diagnostic

if TYPE_CHECKING:
    import os

_IDENTITY_NOTE = "channel = row index; probe identity not recorded in the archive"


def _indexed(group: str, n: int) -> tuple[str, ...]:
    width = max(2, len(str(n - 1)))
    return tuple(f"{group}/{i:0{width}d}" for i in range(n))


def _diag(
    key: str,
    label: str,
    n: int,
    *,
    default: str | None = None,
    verified: bool = False,
    note: str = _IDENTITY_NOTE,
) -> Diagnostic:
    names = _indexed(key, n)
    return Diagnostic(
        key=key,
        label=label,
        pointnames=names,
        default=default or names[0],
        verified=verified,
        note=note,
    )


FOUNDATION_DIAGNOSTICS: dict[str, Diagnostic] = {
    d.key: d
    for d in (
        _diag("mirnov", "Mirnov (B-dot, 500 kHz)", 29, verified=True),
        _diag("mhr", "Magnetics high-res", 8),
        _diag("ece", "ECE radiometer", 48),
        _diag(
            "co2",
            "CO2 interferometer",
            4,
            note=f"{_IDENTITY_NOTE}; present only for shots ≳197965",
        ),
        _diag("sxr", "Soft X-ray (10 kHz)", 320),
        _diag("filterscopes", "Filterscopes", 104),
        _diag("bes", "BES", 64, note=f"{_IDENTITY_NOTE}; rare — most shots lack it"),
    )
}


def foundation_dropdown_choices() -> list[tuple[str, str]]:
    """``(label, key)`` pairs for the diagnostic dropdown."""
    return [(d.label, d.key) for d in FOUNDATION_DIAGNOSTICS.values()]


def signals_for_shot(
    shot: int, data_dir: str | os.PathLike[str] | None = None
) -> dict[str, Diagnostic]:
    """Per-shot diagnostics built from the shot file's actual groups.

    Channel counts (and thus pointname lists) come from the file, so the
    dropdown matches what the shot really carries — including groups with no
    static preset. Falls back to :data:`FOUNDATION_DIAGNOSTICS` when the shot
    file is missing or unreadable.
    """
    signals = list_signals(shot, data_dir)
    if not signals:
        return FOUNDATION_DIAGNOSTICS
    out: dict[str, Diagnostic] = {}
    for group, (n_ch, _n) in sorted(signals.items()):
        base = FOUNDATION_DIAGNOSTICS.get(group)
        names = _indexed(group, n_ch)
        default = base.default if base and base.default in names else names[0]
        out[group] = Diagnostic(
            key=group,
            label=base.label if base else group,
            pointnames=names,
            default=default,
            verified=bool(base and base.verified),
            note=base.note if base else _IDENTITY_NOTE,
        )
    return out
