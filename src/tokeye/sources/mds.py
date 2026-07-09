"""DIII-D MDSplus signal source.

Thin wrapper over the vendored thin-client fetchers in
``tokeye.modespec.classic.data_utils`` (``fetch_ptdata`` + the pickle
``fetch_or_load`` cache). Those keep MDSplus behind an import guard, and this
module defers importing them until :meth:`MDSSource.fetch` is actually called —
so ``import tokeye.sources`` never imports MDSplus (import-hygiene, matching the
modespec-classic guarantee).

Access is the atlas.gat.com thin client, reachable from the login / somega
nodes but often not from batch compute nodes → fetches are cached to disk
(default ``/cscratch/share/tokeye/cache``, overridable via ``$TOKEYE_CACHE``)
so a shot fetched on a reachable node can be reused anywhere.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np

# Scratch cache is cluster-wide and shared; note it is not backed up and files
# older than ~32 days are swept. The Lmod modulefile sets $TOKEYE_CACHE.
DEFAULT_CACHE_ROOT = "/cscratch/share/tokeye/cache"

# MDSplus thin-client server for DIII-D (reachable from login / somega).
DEFAULT_ATLAS = "atlas.gat.com"


def cache_root() -> str:
    """Directory for the on-disk shot cache (``$TOKEYE_CACHE`` or the default)."""
    return os.environ.get("TOKEYE_CACHE", DEFAULT_CACHE_ROOT)


def latest_shot(atlas: str = DEFAULT_ATLAS) -> int | None:
    """Most recent DIII-D shot number from MDSplus, or ``None`` if unavailable.

    Uses the atlas thin client (``current_shot("d3d")``). Returns ``None`` on any
    failure (MDSplus missing, atlas unreachable, off-cluster) so callers can fall
    back gracefully — MDSplus stays deferred, keeping ``import`` MDSplus-free.
    """
    try:
        import MDSplus as mds

        conn = mds.Connection(atlas)
        return int(conn.get('current_shot("d3d")'))
    except Exception:  # noqa: BLE001 - any failure -> no latest shot
        return None


# Bounds are immutable per (shot, pointname), so cache successes in-process — the
# DIM_OF time-base build is seconds server-side, so re-selecting a probe is instant.
_BOUNDS_CACHE: dict[tuple[int, str], tuple[float, float]] = {}


def time_bounds(
    shot: int, pointname: str, atlas: str = DEFAULT_ATLAS
) -> tuple[float, float] | None:
    """``(t0_ms, t1_ms)`` data window for a shot+pointname, or ``None``.

    Evaluates the time base **once** server-side (assigned to ``_t``) and returns
    both endpoints in a single round-trip — used to auto-fill the time-window
    fields on a shot/probe selection. Results are cached in-process. Best-effort:
    returns ``None`` on any failure (MDSplus missing, atlas unreachable, empty
    signal) so the UI just leaves the fields blank. MDSplus stays deferred.
    """
    shot = int(shot)
    pointname = str(pointname)
    ck = (shot, pointname)
    if ck in _BOUNDS_CACHE:
        return _BOUNDS_CACHE[ck]

    bounds: tuple[float, float] | None = None
    try:
        import MDSplus as mds

        conn = mds.Connection(atlas)
        try:
            from tokeye.sources.co2 import is_co2_chord, time_bounds_co2
            from tokeye.sources.ece import is_ece_channel, time_bounds_ece

            if is_co2_chord(pointname):
                bounds = time_bounds_co2(conn, shot, pointname)
            elif is_ece_channel(pointname):
                bounds = time_bounds_ece(conn, shot, pointname)
            else:
                conn.openTree("D3D", shot)
                sig = f'PTDATA("{pointname}", {shot})'
                r = np.asarray(
                    conn.get(f"[ (_t = DIM_OF({sig}))[0], _t[SIZE(_t) - 1] ]").data(),
                    dtype=float,
                )
                if r.size >= 2:
                    t0, t1 = float(r[0]), float(r[-1])
                    if abs(t1) < 100:  # seconds -> ms (repo convention)
                        t0, t1 = t0 * 1e3, t1 * 1e3
                    bounds = (t0, t1) if t1 > t0 else None
        finally:
            with contextlib.suppress(Exception):
                conn.closeAllTrees()
    except Exception:  # noqa: BLE001 - any failure -> no bounds
        return None

    if bounds is not None:
        _BOUNDS_CACHE[ck] = bounds
    return bounds


def _fs_from_time_ms(t_ms: np.ndarray) -> float:
    """Sampling rate [Hz] from a millisecond time axis (0.0 if undeterminable)."""
    if t_ms.size < 2:
        return 0.0
    dt_ms = float(np.median(np.diff(t_ms)))
    if dt_ms <= 0:
        return 0.0
    return 1.0e3 / dt_ms


class MDSSource:
    """Fetch DIII-D signals from MDSplus (atlas.gat.com), cached to disk."""

    def __init__(self, data_dir: str | os.PathLike[str] | None = None) -> None:
        self.data_dir = str(data_dir) if data_dir is not None else cache_root()

    @staticmethod
    def latest_shot(atlas: str = DEFAULT_ATLAS) -> int | None:
        """Most recent DIII-D shot number, or ``None`` if unavailable."""
        return latest_shot(atlas)

    @staticmethod
    def time_bounds(
        shot: int, pointname: str, atlas: str = DEFAULT_ATLAS
    ) -> tuple[float, float] | None:
        """Cheap ``(t0_ms, t1_ms)`` data window, or ``None`` if unavailable."""
        return time_bounds(shot, pointname, atlas)

    def fetch(
        self,
        shot: int,
        pointname: str,
        tlim: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        # Deferred so importing this module does not import MDSplus.
        from tokeye.modespec.classic.data_utils import fetch_or_load, fetch_ptdata
        from tokeye.sources.co2 import fetch_co2_chord, is_co2_chord
        from tokeye.sources.ece import fetch_ece_channel, is_ece_channel

        shot = int(shot)
        pointname = str(pointname)
        # CO2 chords and fast ECE channels are NOT plain PTDATA (that source is
        # all-zeros / unreachable); route them to their real D3D-tree fetchers.
        # Everything else is PTDATA. The pickle cache (keyed by shot+pointname) is
        # shared by all paths.
        if is_co2_chord(pointname):
            fetch_fn = lambda: fetch_co2_chord(shot, pointname)  # noqa: E731
        elif is_ece_channel(pointname):
            fetch_fn = lambda: fetch_ece_channel(shot, pointname)  # noqa: E731
        else:
            fetch_fn = lambda: fetch_ptdata(shot, pointname)  # noqa: E731
        data, t_ms = fetch_or_load(shot, pointname, fetch_fn, self.data_dir)
        x = np.asarray(data, dtype=float).ravel()
        t = np.asarray(t_ms, dtype=float).ravel()

        if tlim is not None and t.size:
            lo, hi = tlim
            keep = (t >= lo) & (t <= hi)
            t, x = t[keep], x[keep]

        return t, x, _fs_from_time_ms(t)
