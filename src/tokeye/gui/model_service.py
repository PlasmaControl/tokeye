"""Lazy, cached torch model loader for the GUI.

torch is imported **only here** (inside :meth:`ModelService.infer`), off the GUI
thread, so the window opens instantly and stays Qt-only until the first Analyze.
Loads and forward passes are serialized under a lock (one inference at a time).
"""

from __future__ import annotations

import threading


class ModelService:
    def __init__(self) -> None:
        self._cache: dict[str, object] = {}
        self._lock = threading.Lock()

    def infer(self, name: str, spec):
        """Load (and cache) the model ``name`` and run inference on ``spec``.

        Returns the ``(2, H, W)`` mask (channel 0 = coherent, 1 = transient).
        Serialized across worker threads.
        """
        from tokeye.inference import model_infer

        with self._lock:
            model = self._get(name)
            return model_infer(spec, model)

    def _get(self, name: str):
        model = self._cache.get(name)
        if model is None:
            from tokeye.app.analyze.load import model_load

            model = model_load(name, device="auto")
            self._cache[name] = model
        return model
