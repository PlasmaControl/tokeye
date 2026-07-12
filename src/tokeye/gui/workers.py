"""Background workers for blocking MDSplus fetch and torch inference.

Each worker is a ``QRunnable`` submitted to the window's ``QThreadPool``; it
reports back through Qt signals (queued to the GUI thread). Every result carries
the originating ``request_id`` so the view can drop stale results when a newer
request has superseded it. All heavy imports (MDSplus, scipy, torch via the
model service) stay inside ``work`` so importing this module is cheap.
"""

from __future__ import annotations

from PySide6 import QtCore


class WorkerSignals(QtCore.QObject):
    progress = QtCore.Signal(int, float, str)  # request_id, fraction, message
    result = QtCore.Signal(int, object)  # request_id, payload
    error = QtCore.Signal(int, str)  # request_id, message
    finished = QtCore.Signal(int)  # request_id


class _Worker(QtCore.QRunnable):
    def __init__(self, request_id: int) -> None:
        super().__init__()
        self.request_id = request_id
        self.signals = WorkerSignals()

    def run(self) -> None:  # QThreadPool entry point
        rid = self.request_id

        def progress(fraction: float, message: str = "") -> None:
            self.signals.progress.emit(rid, float(fraction), message)

        try:
            payload = self.work(progress)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self.signals.error.emit(rid, str(exc))
        else:
            self.signals.result.emit(rid, payload)
        finally:
            self.signals.finished.emit(rid)

    def work(self, progress):
        raise NotImplementedError


class LatestShotWorker(_Worker):
    def work(self, progress):
        from tokeye.sources.factory import latest_shot

        return latest_shot()


class BoundsWorker(_Worker):
    def __init__(self, request_id: int, shot: int, pointname: str) -> None:
        super().__init__(request_id)
        self.shot = shot
        self.pointname = pointname

    def work(self, progress):
        from tokeye.sources.factory import time_bounds

        return time_bounds(int(self.shot), str(self.pointname))


class FetchSpectrogramWorker(_Worker):
    """Fetch one probe, optionally decimate, and STFT it to a spectrogram."""

    def __init__(
        self,
        request_id: int,
        shot: int,
        pointname: str,
        tlim: tuple[float, float] | None,
        decimation: int,
        stft: dict,
    ) -> None:
        super().__init__(request_id)
        self.shot = shot
        self.pointname = pointname
        self.tlim = tlim
        self.decimation = decimation
        self.stft = stft

    def work(self, progress):
        from tokeye.sources.factory import get_source
        from tokeye.transforms import compute_stft

        progress(0.1, f"Fetching {int(self.shot)} / {self.pointname} …")
        t, x, fs = get_source().fetch(int(self.shot), str(self.pointname), self.tlim)
        if x.size == 0:
            raise ValueError(
                f"{int(self.shot)} / {self.pointname} returned no samples in that window."
            )

        d = int(self.decimation) if self.decimation else 1
        if d > 1 and x.size > 64:
            progress(0.5, f"Decimating ×{d} …")
            from scipy.signal import decimate

            x = decimate(x, d, ftype="fir")
            t = t[::d][: x.size]
            fs = fs / d

        progress(0.7, "Computing spectrogram …")
        spec = compute_stft(
            x[None, :],
            n_fft=int(self.stft["n_fft"]),
            hop=int(self.stft["hop"]),
            clip_dc=bool(self.stft["clip_dc"]),
            clip_low=float(self.stft["clip_low"]),
            clip_high=float(self.stft["clip_high"]),
        )
        meta = {
            "fs": float(fs),
            "t0_ms": float(t[0]) if t.size else 0.0,
            "n_fft": int(self.stft["n_fft"]),
            "hop": int(self.stft["hop"]),
            "clip_dc": bool(self.stft["clip_dc"]),
        }
        progress(1.0, "Done")
        return {"spec": spec, "meta": meta, "t": t, "x": x, "fs": float(fs)}


class AnalyzeWorker(_Worker):
    """Run the segmentation model on a loaded spectrogram via the model service."""

    def __init__(self, request_id: int, service, model_name: str, spec) -> None:
        super().__init__(request_id)
        self.service = service
        self.model_name = model_name
        self.spec = spec

    def work(self, progress):
        progress(0.3, "Running model …")
        out = self.service.infer(self.model_name, self.spec)
        progress(1.0, "Done")
        return out


class ModespecWorker(_Worker):
    """Toroidal mode-spectrogram + optional band-matched TokEye gate.

    Returns ``{result, tok_mask, gate_meta, nd}``; the view caches these so the
    coherence slider re-gates locally without recompute.
    """

    def __init__(
        self,
        request_id: int,
        shot: int,
        ref_probe: str,
        tlim: tuple[float, float] | None,
        params: dict,
        gate_cfg: dict,
        service,
        model_name: str,
    ) -> None:
        super().__init__(request_id)
        self.shot = shot
        self.ref_probe = ref_probe
        self.tlim = tlim
        self.params = params
        self.gate_cfg = gate_cfg
        self.service = service
        self.model_name = model_name

    def work(self, progress):
        from tokeye.sources.mirnov import (
            array_gate_mask,
            gate_dominant_mask,
            run_mode_spectrogram,
        )

        p = self.params
        dec = int(p["decimation"]) if p.get("decimation") else None
        progress(0.1, "Fetching Mirnov array + mode spectrogram …")
        result = run_mode_spectrogram(
            int(self.shot),
            "toroidal",
            self.tlim,
            decimation=dec,
            n_range=(int(p["n_min"]), int(p["n_max"])),
            f_min_khz=float(p["f_min"]),
            f_max_khz=float(p["f_max"]),
        )

        tok_mask = gate_meta = nd = None
        g = self.gate_cfg
        if g.get("gate"):
            from tokeye.transforms import (
                DEFAULT_CLIP_DC,
                DEFAULT_CLIP_HIGH,
                DEFAULT_CLIP_LOW,
                DEFAULT_HOP,
                DEFAULT_N_FFT,
            )

            stft_kwargs = {
                "n_fft": DEFAULT_N_FFT,
                "hop": DEFAULT_HOP,
                "clip_dc": DEFAULT_CLIP_DC,
                "clip_low": DEFAULT_CLIP_LOW,
                "clip_high": DEFAULT_CLIP_HIGH,
            }
            tok_mask, gate_meta = array_gate_mask(
                int(self.shot),
                "toroidal",
                self.tlim,
                stft_kwargs,
                lambda spec: self.service.infer(self.model_name, spec),
                threshold=0.5,
                decimation=dec,
                source=g.get("source", "average"),
                reference=str(self.ref_probe) if self.ref_probe else None,
                f_max_khz=float(p["f_max"]),
                on_progress=lambda frac, desc: progress(0.3 + 0.6 * frac, desc),
            )
            nd = gate_dominant_mask(
                result, tok_mask, gate_meta, coh_thresh=float(g.get("coh", 0.5))
            )

        progress(1.0, "Rendering …")
        return {"result": result, "tok_mask": tok_mask, "gate_meta": gate_meta, "nd": nd}
