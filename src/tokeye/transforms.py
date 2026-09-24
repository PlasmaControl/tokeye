"""STFT and log scaling: the numerical front half of the TokEye contract.

:func:`compute_stft` is the preprocessing the released model was trained
with (STFT magnitude -> ``log1p`` -> DC drop -> percentile clip).
Standardization happens later, in :func:`tokeye.inference.infer`.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .config import DEFAULT_CONFIG

DEFAULT_N_FFT = DEFAULT_CONFIG.n_fft
DEFAULT_HOP = DEFAULT_CONFIG.hop  # the released model's training hop
DEFAULT_WINDOW = DEFAULT_CONFIG.window
DEFAULT_CLIP_DC = DEFAULT_CONFIG.clip_dc
DEFAULT_CLIP_LOW = DEFAULT_CONFIG.clip_low
DEFAULT_CLIP_HIGH = DEFAULT_CONFIG.clip_high


def log_scale(arr: np.ndarray) -> np.ndarray:
    """Apply ``log1p`` to a linear-scale spectrogram.

    Rejects negative values: they indicate the input is already log/dB
    scaled, and ``log1p`` would silently produce NaNs below -1.
    """
    if arr.min() < 0:
        raise ValueError(
            "log scaling expects a non-negative (linear-scale) spectrogram; "
            "input has negative values — is it already log/dB scaled?"
        )
    return np.log1p(arr)


def compute_stft(
    arr: np.ndarray,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
    window: str = DEFAULT_WINDOW,
    clip_dc: bool = DEFAULT_CLIP_DC,
    fs: float = 1.0,
    clip_low: float = DEFAULT_CLIP_LOW,
    clip_high: float = DEFAULT_CLIP_HIGH,
) -> np.ndarray:
    """Log-magnitude STFT of a signal, or cross-power of a signal pair.

    Parameters
    ----------
    arr
        A 1D signal, a ``(1, N)`` signal, or a ``(2, N)`` pair whose
        cross-power spectrum ``X0 * conj(X1)`` is used.
    n_fft, hop, window
        STFT window length, hop and window name.
    clip_dc
        Drop the DC row.
    fs
        Sampling rate; it only scales the (unused) axes, never the values.
    clip_low, clip_high
        Percentiles the result is clipped to.

    Returns
    -------
    numpy.ndarray
        ``(n_fft // 2 + 1 - clip_dc, frames)`` float64 spectrogram.
    """
    arr = np.asarray(arr)
    if not (arr.ndim == 1 or (arr.ndim == 2 and arr.shape[0] in (1, 2))):
        raise ValueError(
            "compute_stft expects a 1D signal, a (1, N) signal or a (2, N) "
            f"signal pair; got shape {arr.shape}"
        )

    win = signal.get_window(window, n_fft)
    transform = signal.ShortTimeFFT(win=win, hop=hop, fs=fs)
    sxx = transform.stft(arr)

    if arr.ndim == 2:
        sxx = sxx[0] * np.conj(sxx[1]) if arr.shape[0] == 2 else sxx[0]

    sxx = np.abs(sxx)
    sxx = np.log1p(sxx)

    if clip_dc:
        sxx = sxx[1:, :]

    vmin, vmax = np.percentile(sxx, [clip_low, clip_high])
    return np.clip(sxx, vmin, vmax)
