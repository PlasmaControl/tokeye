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

    def fetch(
        self,
        shot: int,
        pointname: str,
        tlim: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        # Deferred so importing this module does not import MDSplus.
        from tokeye.modespec.classic.data_utils import fetch_or_load, fetch_ptdata

        shot = int(shot)
        pointname = str(pointname)
        data, t_ms = fetch_or_load(
            shot,
            pointname,
            lambda: fetch_ptdata(shot, pointname),
            self.data_dir,
        )
        x = np.asarray(data, dtype=float).ravel()
        t = np.asarray(t_ms, dtype=float).ravel()

        if tlim is not None and t.size:
            lo, hi = tlim
            keep = (t >= lo) & (t <= hi)
            t, x = t[keep], x[keep]

        return t, x, _fs_from_time_ms(t)
