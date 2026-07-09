"""Signal-source interface.

A ``SignalSource`` yields a 1-D time series addressed by ``(shot, pointname)``.
DIII-D's MDSplus reader is the first implementation (:class:`tokeye.sources.mds.MDSSource`);
keeping this as a protocol leaves room for other machines / file backends
without changing the app or CLI that consume it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class SignalSource(Protocol):
    """A source of 1-D signals keyed by shot + pointname."""

    def fetch(
        self,
        shot: int,
        pointname: str,
        tlim: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return ``(t_ms, x, fs_hz)`` for one signal.

        Args:
            shot: machine shot number.
            pointname: diagnostic channel name (e.g. a PTDATA pointname).
            tlim: optional ``(t_min_ms, t_max_ms)`` crop.

        Returns:
            ``t_ms`` (time axis, ms), ``x`` (1-D signal), ``fs_hz`` (sampling
            rate in Hz, derived from ``t_ms``; ``0.0`` if undeterminable).
        """
        ...
