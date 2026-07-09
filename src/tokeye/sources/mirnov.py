"""Cached Mirnov-array fetch + classic mode-spectrogram + TokEye gating.

The vendored ``modespec.classic.fetch_mirnov`` opens a fresh MDSplus connection
every call and never caches — unusable for an interactive app and impossible on a
no-internet compute node. This module is the cached analogue: it pulls each probe
through the same per-probe pickle cache the single-probe path uses
(:class:`tokeye.sources.MDSSource`), so a shot fetched once on somega replays
instantly and works offline in a batch job. The vendored ``modespec`` file is left
untouched.

MDSplus stays deferred (only ``MDSSource.fetch`` touches it), and the heavy
``mode_spectrogram`` import is deferred too, so importing this module is cheap and
MDSplus-free.
"""

from __future__ import annotations

import numpy as np

from .mds import MDSSource
from .presets import MIRNOV_TOROIDAL, MIRNOV_TOROIDAL_ANGLES

# Named toroidal/poloidal arrays available to the mode analysis. Poloidal
# (m-number) needs geometric angles that we do not duplicate yet, so only the
# toroidal array is wired for mode_spectrogram.
_ARRAYS = {
    "toroidal": (MIRNOV_TOROIDAL, MIRNOV_TOROIDAL_ANGLES),
}

MIN_PROBES = 4  # matched-filter mode fit is meaningless with too few probes


def fetch_mirnov_cached(
    shot: int,
    array: str = "toroidal",
    tlim: tuple[float, float] | None = None,
    data_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Fetch a Mirnov array (cached per probe) → ``(signals, t_ms, angles, names)``.

    ``signals`` is ``(n_probes, n_t)``; probes that fail to fetch are dropped
    (along with their angle). Reuses the on-disk pickle cache, so this works on a
    compute node once the shot has been prefetched on somega.
    """
    if array not in _ARRAYS:
        raise ValueError(f"unknown/unsupported array '{array}' (have {list(_ARRAYS)})")
    names, angles = _ARRAYS[array]
    src = MDSSource(data_dir=data_dir)

    sigs: list[np.ndarray] = []
    good_angles: list[float] = []
    good_names: list[str] = []
    t_ref: np.ndarray | None = None
    for name, phi in zip(names, angles):
        try:
            t, x, _fs = src.fetch(int(shot), str(name), tlim)
        except Exception:  # noqa: BLE001 - drop a bad/missing probe, keep the array
            continue
        if x.size < 2:
            continue
        if t_ref is None:
            t_ref = t
        sigs.append(x)
        good_angles.append(float(phi))
        good_names.append(str(name))

    if len(sigs) < MIN_PROBES or t_ref is None:
        raise RuntimeError(
            f"only {len(sigs)} Mirnov probe(s) available for shot {shot}"
            f" (need >= {MIN_PROBES}); is it cached / is atlas reachable?"
        )

    # Probes share the 200 kHz digitizer clock; align to the shortest length.
    n = min(s.size for s in sigs)
    signals = np.stack([s[:n] for s in sigs], axis=0)
    return signals, np.asarray(t_ref[:n], dtype=float), np.asarray(good_angles), good_names


def run_mode_spectrogram(
    shot: int,
    array: str = "toroidal",
    tlim: tuple[float, float] | None = None,
    data_dir: str | None = None,
    **params,
) -> dict:
    """Cached fetch + vendored ``mode_spectrogram``. Returns its result dict."""
    from tokeye.modespec.classic.modespec import mode_spectrogram  # deferred (heavy)

    signals, t_ms, angles, _names = fetch_mirnov_cached(shot, array, tlim, data_dir)
    if "n_range" in params and params["n_range"] is not None:
        params["n_range"] = tuple(params["n_range"])
    return mode_spectrogram(signals, t_ms, angles, **params)


def _stft_axes_khz_ms(shape: tuple[int, int], stft_meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Frequency [kHz] and time [ms] vectors for a TokEye STFT image ``(H, W)``."""
    h, w = shape
    fs = float(stft_meta["fs"])
    n_fft = int(stft_meta.get("n_fft", 1024))
    hop = int(stft_meta.get("hop", 256))
    t0 = float(stft_meta.get("t0_ms", 0.0))
    clip_dc = bool(stft_meta.get("clip_dc", True))
    row0 = 1 if clip_dc else 0
    f_khz = (np.arange(h) + row0) * fs / n_fft / 1e3
    t_ms = t0 + np.arange(w) * hop / fs * 1e3
    return f_khz, t_ms


def gate_dominant(
    result: dict,
    arr_extract: np.ndarray,
    stft_meta: dict,
    mask_threshold: float = 0.5,
    coh_thresh: float | None = None,
    channel: str = "coherent",
) -> np.ndarray:
    """Gate the modespec dominant-``n`` array with the TokEye mask + coherence.

    Resamples the thresholded TokEye mask (STFT grid) onto the modespec
    ``(t_win_ms × freq_khz)`` grid by nearest neighbour, then keeps the dominant
    mode number only where the TokEye mask is on AND ``coherence > coh_thresh``.
    Returns a ``(n_win, n_freq)`` float array, NaN where suppressed — ready for
    :func:`tokeye.sources.viz.render_modespec` via its ``nd`` argument.
    """
    if arr_extract is None or stft_meta is None or float(stft_meta.get("fs", 0.0)) <= 0:
        raise ValueError("gating needs a TokEye mask and an STFT with a known fs")

    arr_extract = np.asarray(arr_extract)
    coherent = arr_extract[0] > mask_threshold
    if channel == "both" and arr_extract.shape[0] > 1:
        tok_mask = coherent | (arr_extract[1] > mask_threshold)
    else:
        tok_mask = coherent  # (H, W) on the STFT grid

    f_stft, t_stft = _stft_axes_khz_ms(tok_mask.shape, stft_meta)
    f_ms = np.asarray(result["freq_khz"])
    t_ms = np.asarray(result["t_win_ms"])

    # Nearest STFT bin for each modespec bin; flag modespec bins outside STFT range.
    row = np.abs(f_stft[:, None] - f_ms[None, :]).argmin(axis=0)   # (n_freq,)
    col = np.abs(t_stft[:, None] - t_ms[None, :]).argmin(axis=0)   # (n_win,)
    in_f = (f_ms >= f_stft.min()) & (f_ms <= f_stft.max())
    in_t = (t_ms >= t_stft.min()) & (t_ms <= t_stft.max())

    mask_on_grid = tok_mask[np.ix_(row, col)].T                    # (n_win, n_freq)
    mask_on_grid &= in_t[:, None] & in_f[None, :]

    coh = np.asarray(result["coherence"])
    c95 = float(result.get("c95", 0.0))
    thresh = coh_thresh if coh_thresh is not None else max(c95, 0.3)
    nd = np.asarray(result["n_dominant"], dtype=float)
    keep = mask_on_grid & (coh > thresh)
    return np.where(keep, nd, np.nan)
