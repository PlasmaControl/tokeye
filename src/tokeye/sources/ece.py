"""Real DIII-D fast ECE (electron-cyclotron-emission) fetch.

The fast ECE ``TECEF`` channels are **not** plain PTDATA (the generic
``PTDATA("TECEF20", shot)`` fetch can't reach them) and the vendored
``modespec.fetch_ece`` reads them from a separate ``ece`` MDSplus tree. The
neighboring ``FusionAIHub`` repo (``co2_check/config_ece.yaml`` +
``ece_probe.py``) confirmed the simplest working source is a **D3D-tree node** —
exactly the CO2 fix pattern:

* ``\\D3D::TOP.ELECTRONS.ECE.TECEF:TECEF01`` … ``TECEF48`` (~500 kHz, fast Te).

So this module fetches each channel from the ``D3D`` tree node, chosen by a
**non-zero gate** (raise if a channel is absent/all-zero), mirroring
``tokeye.sources.co2``. ``fetch_ece_channel`` returns the same ``(data, time_ms)``
2-tuple as ``data_utils.fetch_ptdata``, so :meth:`tokeye.sources.mds.MDSSource.fetch`
can route ``TECEF`` pointnames here transparently through the existing pickle
cache. As with the rest of ``tokeye.sources``, ``import MDSplus`` is deferred so
importing this module stays MDSplus-free.
"""

from __future__ import annotations

import contextlib

import numpy as np

from tokeye.sources.mds import DEFAULT_ATLAS

# DIII-D fast ECE has up to 48 fixed-frequency channels (recent shots); older
# shots have fewer — a missing channel just fails the non-zero gate.
N_ECE = 48
ECE_CHANNELS: tuple[str, ...] = tuple(f"TECEF{i:02d}" for i in range(1, N_ECE + 1))
_ECE_SET = frozenset(ECE_CHANNELS)


def is_ece_channel(pointname: str) -> bool:
    """True if ``pointname`` is one of the fast ECE channels handled here."""
    return str(pointname).upper() in _ECE_SET


def ece_node(channel: str) -> str:
    """The D3D-tree node for a fast ECE channel (e.g. ``TECEF20``)."""
    return rf"\D3D::TOP.ELECTRONS.ECE.TECEF:{str(channel).upper()}"


def _to_ms(t: np.ndarray) -> np.ndarray:
    """Match the repo convention: a time axis in seconds (|t| < 100) -> ms."""
    t = np.asarray(t, dtype=float)
    if t.size and np.max(np.abs(t)) < 100:
        t = t * 1e3
    return t


def fetch_ece_channel(
    shot: int, channel: str, atlas: str = DEFAULT_ATLAS
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch one fast ECE channel from the D3D-tree TECEF node.

    Returns ``(data, time_ms)`` — the same contract as ``fetch_ptdata`` — so it
    slots straight into the ``fetch_or_load`` pickle cache. Raises ``RuntimeError``
    if the channel is absent or all-zero for this shot.
    """
    import MDSplus as mds

    key = str(channel).upper()
    if key not in _ECE_SET:
        raise KeyError(f"unknown ECE channel {channel!r}; expected TECEF01..TECEF{N_ECE}")
    node = ece_node(key)

    conn = mds.Connection(atlas)
    try:
        conn.openTree("D3D", int(shot))
        z = np.asarray(conn.get(node).data(), dtype=float)
        if z.size <= 1 or not np.count_nonzero(z):
            raise RuntimeError(
                f"ECE channel {key} for shot {shot}: absent or all-zero "
                f"(node {node})."
            )
        t = np.asarray(conn.get(f"dim_of({node})").data(), dtype=float)
    finally:
        with contextlib.suppress(Exception):
            conn.closeAllTrees()

    return z, _to_ms(t)


def time_bounds_ece(conn, shot: int, channel: str) -> tuple[float, float] | None:
    """``(t0_ms, t1_ms)`` from the TECEF node endpoints in one round-trip, or None.

    Evaluates the time base **once** (assigned to ``_t``) and returns both endpoints
    in a single call. Returns None if the channel/tree is unreachable so the caller
    leaves the time-window fields blank. Mirrors :func:`tokeye.sources.co2.time_bounds_co2`.
    """
    node = ece_node(channel)
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
