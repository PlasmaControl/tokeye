"""Princeton ``foundation_model`` HDF5 signal source.

Reads the shared DIII-D shot archive on stellar's GPFS
(``/scratch/gpfs/EKOLEMEN/foundation_model``): one ``{shot}_processed.h5`` per
shot, one HDF5 group per signal, each holding ``xdata`` float32 ``(N,)`` — the
time base in **seconds** — and ``ydata`` float32 ``(C, N)`` — channels ×
samples. Groups carry no attributes; a signal absent for a shot is stored
degenerate with ``xdata`` of shape ``(1,)``. Channel rows follow the exporter's
sorted-original-name order, but the original names are not recorded, so
pointnames here are ``group/index`` (e.g. ``mirnov/07``) and probe identity is
unknown.

Precision trap (measured on the real files): the float32 time base quantizes to
~0.24 ms steps at ``|t| ≈ 4 s``, so a median-of-diffs sampling rate comes out
524288 Hz for a true 500 kHz signal. :meth:`FoundationSource.fetch` therefore
derives ``dt`` from the float64 endpoints, ``(x[-1] - x[0]) / (N - 1)``, and
synthesizes the returned ms time axis — it never returns the raw float32 axis.

h5py is imported inside the functions that need it, keeping
``import tokeye.sources`` h5py-free (import hygiene, matching the MDSplus
guarantee). No on-disk cache: the data is already local GPFS.
"""

from __future__ import annotations

import os
import re
import time
import warnings
from pathlib import Path

import numpy as np

# Read-only shared archive on stellar (group `kolemen`); ~17k shots, one
# {shot}_processed.h5 each. The Tcl modulefile sets $TOKEYE_FOUNDATION_DIR.
DEFAULT_FOUNDATION_DIR = "/scratch/gpfs/EKOLEMEN/foundation_model"

_SHOT_FILE_RE = re.compile(r"^(\d+)_processed\.h5$")

# How far the spot-checked midpoint may drift from the uniform grid (in units
# of dt) before fetch falls back to reading the full time base.
_UNIFORMITY_TOL = 0.5


def foundation_dir() -> str:
    """Shot-archive directory (``$TOKEYE_FOUNDATION_DIR`` or the default)."""
    return os.environ.get("TOKEYE_FOUNDATION_DIR", DEFAULT_FOUNDATION_DIR)


def shot_path(shot: int, data_dir: str | os.PathLike[str] | None = None) -> Path:
    """Path of the ``{shot}_processed.h5`` file for ``shot``."""
    return Path(data_dir if data_dir is not None else foundation_dir()) / (
        f"{int(shot)}_processed.h5"
    )


def parse_pointname(pointname: str) -> tuple[str, int]:
    """``"mirnov/07"`` → ``("mirnov", 7)``; a bare group means channel 0.

    The index may be unpadded (``mirnov/7``). Malformed names (empty group,
    non-integer or negative index) raise ``ValueError``.
    """
    name = str(pointname).strip()
    if "/" not in name:
        if not name:
            raise ValueError("empty pointname (expected 'group/index')")
        return name, 0
    group, _, idx = name.rpartition("/")
    group = group.strip()
    idx = idx.strip()
    if not group or not idx.isdigit():
        raise ValueError(
            f"malformed pointname {pointname!r} (expected 'group/index', "
            "e.g. 'mirnov/07')"
        )
    return group, int(idx)


def pointname_slug(pointname: str) -> str:
    """Filename-safe pointname (``mirnov/07`` → ``mirnov-07``)."""
    return str(pointname).strip().replace("/", "-")


def list_shots(data_dir: str | os.PathLike[str] | None = None) -> list[int]:
    """Sorted shot numbers present in the archive (empty list if unreadable).

    Matches only ``{shot}_processed.h5`` entries, so the archive's ``data/``
    and ``models/`` subdirectories don't pollute the listing.
    """
    root = Path(data_dir if data_dir is not None else foundation_dir())
    try:
        entries = list(root.iterdir())
    except OSError:
        return []
    shots = []
    for entry in entries:
        m = _SHOT_FILE_RE.match(entry.name)
        if m:
            shots.append(int(m.group(1)))
    return sorted(shots)


# The archive is append-only and huge (~17k files), so cache the newest-shot
# scan briefly instead of hitting GPFS on every GUI poll.
_LATEST_TTL_S = 300.0
_LATEST_CACHE: dict[str, tuple[float, int | None]] = {}


def latest_shot(data_dir: str | os.PathLike[str] | None = None) -> int | None:
    """Newest shot number in the archive, or ``None`` if none are visible."""
    root = str(data_dir if data_dir is not None else foundation_dir())
    now = time.monotonic()
    hit = _LATEST_CACHE.get(root)
    if hit is not None and now - hit[0] < _LATEST_TTL_S:
        return hit[1]
    shots = list_shots(root)
    latest = shots[-1] if shots else None
    _LATEST_CACHE[root] = (now, latest)
    return latest


def list_signals(
    shot: int, data_dir: str | os.PathLike[str] | None = None
) -> dict[str, tuple[int, int]] | None:
    """``{group: (n_channels, n_samples)}`` for a shot, skipping empty groups.

    Returns ``None`` if the shot file is missing/unreadable (callers fall back
    to the static presets).
    """
    path = shot_path(shot, data_dir)
    try:
        import h5py

        with h5py.File(path, "r") as f:
            out: dict[str, tuple[int, int]] = {}
            for group in f:
                node = f[group]
                if not isinstance(node, h5py.Group) or "ydata" not in node:
                    continue
                if node["ydata"].ndim != 2:  # e.g. tangtv camera video (4-D)
                    continue
                c, n = node["ydata"].shape
                if n < 2:  # degenerate placeholder (signal absent this shot)
                    continue
                out[group] = (int(c), int(n))
            return out
    except OSError:
        return None


# Bounds are immutable per (dir, shot, group) — the archive is read-only.
_BOUNDS_CACHE: dict[tuple[str, int, str], tuple[float, float]] = {}


def time_bounds(
    shot: int,
    pointname: str,
    data_dir: str | os.PathLike[str] | None = None,
) -> tuple[float, float] | None:
    """``(t0_ms, t1_ms)`` for a shot's signal group, or ``None``.

    Reads only the two endpoint samples of ``xdata`` — used to auto-fill the
    time-window fields. Best-effort: any failure (missing file/group, empty
    signal) returns ``None`` so the UI just leaves the fields blank.
    """
    shot = int(shot)
    try:
        group, _ = parse_pointname(pointname)
    except ValueError:
        return None
    root = str(data_dir if data_dir is not None else foundation_dir())
    ck = (root, shot, group)
    if ck in _BOUNDS_CACHE:
        return _BOUNDS_CACHE[ck]

    try:
        import h5py

        with h5py.File(shot_path(shot, root), "r") as f:
            xdata = f[group]["xdata"]
            n = xdata.shape[0]
            if n < 2:
                return None
            t0 = float(xdata[0]) * 1e3
            t1 = float(xdata[n - 1]) * 1e3
    except (OSError, KeyError, IndexError):
        return None

    if t1 <= t0:
        return None
    _BOUNDS_CACHE[ck] = (t0, t1)
    return (t0, t1)


class FoundationSource:
    """Fetch signals from the local ``foundation_model`` HDF5 archive."""

    def __init__(self, data_dir: str | os.PathLike[str] | None = None) -> None:
        self.data_dir = str(data_dir) if data_dir is not None else foundation_dir()

    @staticmethod
    def latest_shot() -> int | None:
        """Newest shot in the archive, or ``None`` if unavailable."""
        return latest_shot()

    @staticmethod
    def time_bounds(shot: int, pointname: str) -> tuple[float, float] | None:
        """Cheap ``(t0_ms, t1_ms)`` data window, or ``None`` if unavailable."""
        return time_bounds(shot, pointname)

    def fetch(
        self,
        shot: int,
        pointname: str,
        tlim: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """One channel of one signal group as ``(t_ms, x, fs_hz)``.

        ``tlim`` is an inclusive ms window (matching the MDS source's crop
        semantics); a window outside the data returns empty arrays, as does a
        signal stored degenerate for this shot.
        """
        import h5py

        shot = int(shot)
        group, channel = parse_pointname(pointname)
        path = shot_path(shot, self.data_dir)
        if not path.is_file():
            shots = list_shots(self.data_dir)
            span = f" (archive holds {shots[0]}–{shots[-1]})" if shots else ""
            raise ValueError(f"no foundation_model file for shot {shot}: {path}{span}")

        empty = (np.array([], dtype=float), np.array([], dtype=float), 0.0)
        with h5py.File(path, "r") as f:
            if group not in f or "ydata" not in f[group]:
                available = ", ".join(sorted(k for k in f if "ydata" in f[k]))
                raise ValueError(
                    f"shot {shot} has no signal group {group!r} "
                    f"(available: {available})"
                )
            node = f[group]
            ydata = node["ydata"]
            xdata = node["xdata"]
            if ydata.ndim != 2:
                raise ValueError(
                    f"{group!r} is not a channels×samples signal "
                    f"(ydata shape {ydata.shape}; e.g. tangtv is camera video)"
                )
            n_ch, n = ydata.shape
            if channel >= n_ch:
                raise ValueError(
                    f"channel {channel} out of range for {group!r} "
                    f"(shot {shot} has channels 0–{n_ch - 1})"
                )
            if n < 2 or xdata.shape[0] != n:
                return empty

            # float64 endpoint time base: immune to the float32 quantization
            # that skews per-sample diffs (see module docstring).
            x0 = float(xdata[0])
            x1 = float(xdata[n - 1])
            dt = (x1 - x0) / (n - 1)
            if dt <= 0:
                raise ValueError(
                    f"non-increasing time base for shot {shot} {group!r} "
                    f"(t[0]={x0!r}, t[-1]={x1!r})"
                )

            mid = n // 2
            expected_mid = x0 + dt * mid
            if abs(float(xdata[mid]) - expected_mid) > _UNIFORMITY_TOL * dt:
                warnings.warn(
                    f"shot {shot} {group!r}: time base is not uniform; "
                    "falling back to reading it in full",
                    stacklevel=2,
                )
                t_ms = np.asarray(xdata[:], dtype=float) * 1e3
                x = np.asarray(ydata[channel, :], dtype=float)
                if tlim is not None:
                    lo, hi = tlim
                    keep = (t_ms >= lo) & (t_ms <= hi)
                    t_ms, x = t_ms[keep], x[keep]
                fs = 1.0e3 / float(np.median(np.diff(t_ms))) if t_ms.size > 1 else 0.0
                return t_ms, x, fs

            i0, i1 = 0, n
            if tlim is not None:
                lo_s, hi_s = tlim[0] / 1e3, tlim[1] / 1e3
                i0 = max(0, int(np.ceil((lo_s - x0) / dt - 1e-9)))
                i1 = min(n, int(np.floor((hi_s - x0) / dt + 1e-9)) + 1)
                if i0 >= i1:
                    return empty

            x = np.asarray(ydata[channel, i0:i1], dtype=float)

        t_ms = (x0 + dt * np.arange(i0, i1, dtype=float)) * 1e3
        return t_ms, x, 1.0 / dt
