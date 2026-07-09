"""Real DIII-D fast CO2/BCI interferometer fetch.

The obvious ``PTDATA("DENV1UF", shot)`` pointnames resolve to a correctly-sized,
correct-timebase array that is **all zeros** for every shot — so the ``co2``
diagnostic showed nothing. The live fast-CO2 data lives elsewhere (confirmed in
the neighboring ``FusionAIHub`` repo's ``co2_check`` post-mortem):

* **Recent shots** — the D3D-tree node
  ``\\D3D::TOP.ELECTRONS.BCI.DPD.{V1,V2,V3,R0}:DENUF`` (~2 MHz, float64).
* **Older shots** — segmented ``BCI``-tree signals ``den{v1,v2,v3,r0}_uf_0..9``,
  discovered via ``findsig`` and concatenated in time order (~1.667 MHz). The DPD
  node is absent (``%TREE-W-NNF``) on those shots.

The source is chosen per chord by a **non-zero gate** (DPD first, then BCI). This
mirrors ``FusionAIHub/scripts/data_fetching_omega/co2_check/fetch_co2_any.py``.

``fetch_co2_chord`` returns the same ``(data, time_ms)`` 2-tuple as
``data_utils.fetch_ptdata``, so :meth:`tokeye.sources.mds.MDSSource.fetch` can
route CO2 pointnames here transparently through the existing pickle cache. As
with the rest of ``tokeye.sources``, ``import MDSplus`` is deferred so importing
this module stays MDSplus-free.
"""

from __future__ import annotations

import contextlib

import numpy as np

from tokeye.sources.mds import DEFAULT_ATLAS

# Chord pointname -> (DPD subnode, BCI findsig base). Keys are the user-facing
# pointnames listed in presets.CO2_CHORDS.
CO2_CHORD_MAP: dict[str, tuple[str, str]] = {
    "DENV1_UF": ("V1", "denv1_uf"),
    "DENV2_UF": ("V2", "denv2_uf"),
    "DENV3_UF": ("V3", "denv3_uf"),
    "DENR0_UF": ("R0", "denr0_uf"),
}
CO2_CHORDS: tuple[str, ...] = tuple(CO2_CHORD_MAP)

_NSEG = 10  # BCI segments denX_uf_0 .. denX_uf_9


def is_co2_chord(pointname: str) -> bool:
    """True if ``pointname`` is one of the CO2 chords handled by this module."""
    return str(pointname).upper() in CO2_CHORD_MAP


def dpd_node(chord: str) -> str:
    """The D3D-tree DPD node for a CO2 chord (recent-shot source)."""
    sub, _base = CO2_CHORD_MAP[str(chord).upper()]
    return rf"\D3D::TOP.ELECTRONS.BCI.DPD.{sub}:DENUF"


def _sstr(v) -> str:
    v = np.atleast_1d(v)[0]
    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)


def _to_ms(t: np.ndarray) -> np.ndarray:
    """Match the repo convention: a time axis in seconds (|t| < 100) -> ms."""
    t = np.asarray(t, dtype=float)
    if t.size and np.max(np.abs(t)) < 100:
        t = t * 1e3
    return t


def _get_dpd(conn, shot: int, sub: str):
    """``(t, z)`` from the D3D-tree DPD node if present and non-zero, else None."""
    node = rf"\D3D::TOP.ELECTRONS.BCI.DPD.{sub}:DENUF"
    try:
        conn.openTree("D3D", shot)
        z = np.asarray(conn.get(node).data(), dtype=float)
        if z.size <= 1 or not np.count_nonzero(z):
            return None
        t = np.asarray(conn.get(f"dim_of({node})").data(), dtype=float)
        return t, z
    except Exception:  # noqa: BLE001 - node absent (%TREE-W-NNF) on old shots
        return None


def _get_bci(conn, shot: int, base: str):
    """``(t, z)`` concatenated from non-empty BCI segments via findsig, else None."""
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    for n in range(_NSEG):
        sig = f"{base}_{n}"
        try:
            tag = _sstr(conn.get(f'findsig("{sig}",_fstree)').data())
            fstree = _sstr(conn.get("_fstree").data())
            conn.openTree(fstree, shot)
            # dim_of(tag) directly returns %TREE-E-NODATA; assign _s first.
            z = np.asarray(conn.get("_s = " + tag).data(), dtype=float)
            if z.size <= 1:
                continue
            t = np.asarray(conn.get("dim_of(_s)").data(), dtype=float)
            segs.append((t, z))
        except Exception:  # noqa: BLE001 - skip a missing/bad segment
            continue
    if not segs:
        return None
    segs.sort(key=lambda r: float(r[0][0]))  # time order
    t = np.concatenate([s[0] for s in segs])
    z = np.concatenate([s[1] for s in segs])
    if not np.count_nonzero(z):
        return None
    return t, z


def fetch_co2_chord(
    shot: int, chord: str, atlas: str = DEFAULT_ATLAS
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch one CO2 chord, auto-selecting the non-zero source.

    Returns ``(data, time_ms)`` — the same contract as ``fetch_ptdata`` — so it
    slots straight into the ``fetch_or_load`` pickle cache. Raises ``RuntimeError``
    if neither the DPD node nor the BCI segments have non-zero data (never returns
    the all-zeros PTDATA trap).
    """
    import MDSplus as mds

    key = str(chord).upper()
    if key not in CO2_CHORD_MAP:
        raise KeyError(f"unknown CO2 chord {chord!r}; expected one of {CO2_CHORDS}")
    sub, base = CO2_CHORD_MAP[key]

    conn = mds.Connection(atlas)
    try:
        r = _get_dpd(conn, int(shot), sub) or _get_bci(conn, int(shot), base)
    finally:
        with contextlib.suppress(Exception):
            conn.closeAllTrees()

    if r is None:
        raise RuntimeError(
            f"CO2 chord {chord} for shot {shot}: no non-zero source "
            "(DPD node absent/zero and BCI segments empty/zero)."
        )
    t, z = r
    return z, _to_ms(t)


def time_bounds_co2(conn, shot: int, chord: str) -> tuple[float, float] | None:
    """``(t0_ms, t1_ms)`` from the DPD node endpoints in one round-trip, or None.

    Best-effort: only tries the recent-shot DPD node. Building the time base is the
    expensive server-side step, so it is evaluated **once** (assigned to ``_t``) and
    both endpoints returned in a single call. Returns None on old shots without the
    node so the caller leaves the fields blank rather than pay a full segmented
    fetch just for bounds.
    """
    node = dpd_node(chord)
    try:
        conn.openTree("D3D", int(shot))
        r = np.asarray(
            conn.get(f"[ (_t = dim_of({node}))[0], _t[size(_t) - 1] ]").data(),
            dtype=float,
        )
    except Exception:  # noqa: BLE001 - node absent / unreachable
        return None
    if r.size < 2:
        return None
    t0, t1 = float(r[0]), float(r[-1])
    if abs(t1) < 100:  # seconds -> ms
        t0, t1 = t0 * 1e3, t1 * 1e3
    return (t0, t1) if t1 > t0 else None
