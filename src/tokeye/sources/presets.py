"""DIII-D diagnostic presets: named pointname sets for the app + CLI.

Each :class:`Diagnostic` maps a short key to a list of PTDATA pointnames and a
sensible default channel. ``mag`` (the toroidal Mirnov array) is the verified,
end-to-end case — it is the MHD-mode signal the segmentation model targets, and
its 14 probes + toroidal angles feed the classic mode-number analysis.

The others mirror pyspecview's diagnostic menu:

* ``mag_pol`` — the 31-probe 322° poloidal Mirnov array (poloidal ``m`` analysis).
* ``mhr``     — the 8 high-resolution magnetics probes ``B1``..``B8`` (2 MHz).
* ``ece``     — electron-cyclotron-emission channels ``TECEF01``..``TECEF40``.
* ``bes``     — beam-emission-spectroscopy channels ``BESFU01``..``BESFU40``.
* ``co2``     — CO2/BCI interferometer density chords.

All are verified against a live shot (PTDATA on the ``D3D`` tree) except ``ece``:
``TECEF`` channels live in a separate ``ece`` MDSplus tree, not PTDATA, so the
generic fetch can't reach them yet (they are listed for selection; wiring the
tree fetch is a follow-up). The probe dropdown allows custom entries too.
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


# ── Toroidal Mirnov array (Bp_probes_R0), B-dot signals (~200 kHz) ───────────────
# Names + toroidal angles [deg] copied from
# tokeye.modespec.classic.modespec.TOR_PROBES (geometry from
# /fusion/projects/diagnostics/magnetics/data/coords/all_mag), kept index-aligned
# and duplicated here so tokeye.sources stays decoupled from the classic path.
# phi is re-centred exactly as the classic code (raw phi > 315° -> phi - 360°);
# the matched filter uses exp(-i n phi), so only the name<->angle pairing matters.
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
MIRNOV_TOROIDAL_ANGLES: tuple[float, ...] = (
    19.5,
    67.5,
    97.4,
    127.9,
    132.5,
    137.4,
    157.6,
    199.7,
    246.4,
    277.5,
    307.0,
    312.4,
    -42.6,   # MPI66M322D: raw 317.4° -> -42.6°
    -20.3,   # MPI66M340D: raw 339.7° -> -20.3°
)

# ── Poloidal Mirnov array at phi ≈ 322° (Bp_probes_322), 31 probes ───────────────
# Names copied from tokeye.modespec.classic.modespec.POL_PROBES_RAW. Poloidal
# angles are geometry-derived (atan2(Z, R-R0)); not duplicated here because the
# poloidal (m-number) mode analysis is a later follow-up — these are exposed now
# only for single-probe spectrogram viewing.
MIRNOV_POLOIDAL: tuple[str, ...] = (
    "MPI11M322D",
    "MPI1A322D",
    "MPI2A322D",
    "MPI3A322D",
    "MPI4A322D",
    "MPI5A322D",
    "MPI8A322D",
    "MPI89A322D",
    "MPI9A322D",
    "MPI79FA322D",
    "MPI79NA322D",
    "MPI7FA322D",
    "MPI7NA322D",
    "MPI67A322D",
    "MPI6FA322D",
    "MPI6NA322D",
    "MPI66M322D",
    "MPI1B322D",
    "MPI2B322D",
    "MPI3B322D",
    "MPI4B322D",
    "MPI5B322D",
    "MPI8B322D",
    "MPI89B322D",
    "MPI9B322D",
    "MPI79B322D",
    "MPI7FB322D",
    "MPI7NB322D",
    "MPI67B322D",
    "MPI6FB322D",
    "MPI6NB322D",
)

# ── High-resolution magnetics ("b1-b8"), PTDATA B1..B8 (~2 MHz) ──────────────────
# NOT Mirnov/MPI: the 8 fast magnetics probes (provenance:
# training/big_tf_unet_ablation/preprocess/preserve_raw_fast.py "mhr" modality).
MHR_PROBES: tuple[str, ...] = tuple(f"B{i}" for i in range(1, 9))

# ── ECE (electron cyclotron emission), TECEF01..TECEF40 (~500 kHz) ───────────────
ECE_CHANNELS: tuple[str, ...] = tuple(f"TECEF{i:02d}" for i in range(1, 41))

# ── BES (beam emission spectroscopy), BESFU01..BESFU40 (~1 MHz) ──────────────────
BES_CHANNELS: tuple[str, ...] = tuple(f"BESFU{i:02d}" for i in range(1, 41))

# ── CO2 / BCI interferometer density chords (~2 MHz) ─────────────────────────────
CO2_CHORDS: tuple[str, ...] = ("DENV1UF", "DENV2UF", "DENV3UF", "DENR0UF")


DIAGNOSTICS: dict[str, Diagnostic] = {
    "mag": Diagnostic(
        key="mag",
        label="Fast Magnetics / Mirnov (toroidal B-dot, ~200 kHz)",
        pointnames=MIRNOV_TOROIDAL,
        default="MPI66M067D",
        verified=True,
        note="DIII-D toroidal Mirnov array; the MHD-mode case the U-Net targets.",
    ),
    "mag_pol": Diagnostic(
        key="mag_pol",
        label="Poloidal Mirnov (322° B-dot array)",
        pointnames=MIRNOV_POLOIDAL,
        default="MPI66M322D",
        verified=True,
        note="31-probe poloidal array at phi≈322°; single-probe viewing (PTDATA).",
    ),
    "mhr": Diagnostic(
        key="mhr",
        label="High-Res Magnetics (B1–B8, ~2 MHz)",
        pointnames=MHR_PROBES,
        default="B1",
        verified=True,
        note="Fast magnetics probes B1..B8 (not Mirnov/MPI); PTDATA, verified live.",
    ),
    "ece": Diagnostic(
        key="ece",
        label="Electron Cyclotron Emission (TECEF, ~500 kHz)",
        pointnames=ECE_CHANNELS,
        default="TECEF20",
        verified=False,
        note=(
            "TECEF channels live in the 'ece' MDSplus tree (\\TECEFnn), NOT PTDATA — "
            "the generic fetch can't reach them yet (see modespec.fetch_ece). Listed "
            "for selection; fetch is a follow-up."
        ),
    ),
    "co2": Diagnostic(
        key="co2",
        label="CO2 Interferometer (BCI density chords)",
        pointnames=CO2_CHORDS,
        default="DENV2UF",
        verified=True,
        note="CO2/BCI density chords (PTDATA); verified live.",
    ),
    "bes": Diagnostic(
        key="bes",
        label="Beam Emission Spectroscopy (BESFU, ~1 MHz)",
        pointnames=BES_CHANNELS,
        default="BESFU20",
        verified=True,
        note="BESFU01..BESFU40 fast channels (PTDATA); verified live (availability "
        "varies by shot).",
    ),
}


def diagnostic_dropdown_choices() -> list[tuple[str, str]]:
    """``(label, key)`` pairs for a Gradio diagnostic dropdown."""
    return [(d.label, d.key) for d in DIAGNOSTICS.values()]
