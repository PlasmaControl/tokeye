"""DIII-D diagnostic presets: named pointname sets for the app + CLI.

Each :class:`Diagnostic` maps a short key to a list of PTDATA pointnames and a
sensible default channel. ``mag`` (the toroidal Mirnov array) is the verified,
end-to-end case — it is the MHD-mode signal the segmentation model targets.
``ece`` / ``co2`` / ``bes`` are scaffolds (``verified=False``): the pointname
sets need confirming against a live shot before they're trusted.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Diagnostic:
    """A named DIII-D diagnostic and its selectable pointnames."""

    key: str
    label: str
    pointnames: tuple[str, ...]
    default: str
    verified: bool = False
    note: str = ""


# Toroidal Mirnov array (Bp_probes_R0), B-dot signals (~200 kHz). Names copied
# from tokeye.modespec.classic.modespec.TOR_PROBES (geometry from
# /fusion/projects/diagnostics/magnetics/data/coords/all_mag); duplicated here
# so tokeye.sources stays decoupled from the classic path.
MIRNOV_TOROIDAL: tuple[str, ...] = (
    "MPI66M020D",
    "MPI66M067D",
    "MPI66M097D",
    "MPI66M127D",
    "MPI66M132D",
    "MPI66M137D",
    "MPI66M157D",
    "MPI66M200D",
    "MPI66M247D",
    "MPI66M277D",
    "MPI66M307D",
    "MPI66M312D",
    "MPI66M322D",
    "MPI66M340D",
)

DIAGNOSTICS: dict[str, Diagnostic] = {
    "mag": Diagnostic(
        key="mag",
        label="Fast Magnetics / Mirnov (toroidal B-dot, ~200 kHz)",
        pointnames=MIRNOV_TOROIDAL,
        default="MPI66M067D",
        verified=True,
        note="DIII-D toroidal Mirnov array; the MHD-mode case the U-Net targets.",
    ),
    "ece": Diagnostic(
        key="ece",
        label="Electron Cyclotron Emission (scaffold)",
        pointnames=("TECEF01", "TECEF02", "TECEF03", "TECEF04"),
        default="TECEF01",
        verified=False,
        note="Scaffold — representative TECEF channels; confirm names + live fetch.",
    ),
    "co2": Diagnostic(
        key="co2",
        label="CO2 Interferometer (scaffold)",
        pointnames=("DENV1UF", "DENV2UF", "DENV3UF", "DENR0UF"),
        default="DENV2UF",
        verified=False,
        note="Scaffold — CO2 density chords; confirm names + live fetch.",
    ),
    "bes": Diagnostic(
        key="bes",
        label="Beam Emission Spectroscopy (scaffold)",
        pointnames=("BESFU01", "BESFU02", "BESFU03", "BESFU04"),
        default="BESFU01",
        verified=False,
        note="Scaffold — BES fast channels; confirm names + live fetch.",
    ),
}


def diagnostic_dropdown_choices() -> list[tuple[str, str]]:
    """``(label, key)`` pairs for a Gradio diagnostic dropdown."""
    return [(d.label, d.key) for d in DIAGNOSTICS.values()]
