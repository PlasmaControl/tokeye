from __future__ import annotations

import os

import numpy as np
from scipy import signal

DEFAULT_N_FFT = 1024
DEFAULT_HOP = 128  # matches the released model's training recipe (n_fft=1024)
DEFAULT_CLIP_DC = True
DEFAULT_CLIP_LOW = 1.0
DEFAULT_CLIP_HIGH = 99.0

# Default analysis rate. TokEye's training data were resampled to a uniform
# 500 kHz (ECE native; mhr 2 MHz, BES 1 MHz, CO2 ~1.67 MHz decimated down), and
# 500 kHz is the native rate of the current DIII-D Mirnov probes that modespec /
# pyspecview are normally run on. Faster signals are decimated to about this
# rate by default; slower ones are left at their native rate (never upsampled).
DEFAULT_TARGET_FS_HZ = 500e3


def default_target_fs_hz() -> float:
    """Analysis rate in Hz: ``$TOKEYE_FS_KHZ`` (kHz) if set, else 500 kHz.

    ``native`` / ``0`` / ``off`` mean "no decimation" (returns 0). Set per user in
    ``~/.tokeye/settings`` by the Omega launcher; an unparsable value is ignored.
    """
    raw = os.environ.get("TOKEYE_FS_KHZ", "").strip().lower().removesuffix("khz").strip()
    if not raw:
        return DEFAULT_TARGET_FS_HZ
    if raw in {"native", "0", "off", "none", "full"}:
        return 0.0
    try:
        khz = float(raw)
    except ValueError:
        return DEFAULT_TARGET_FS_HZ
    return khz * 1e3 if khz > 0 else 0.0


def auto_decimation(fs_hz: float, target_hz: float | None = DEFAULT_TARGET_FS_HZ) -> int:
    """Integer decimation bringing ``fs_hz`` down to about ``target_hz``.

    Never goes below the target (so the analysed band always reaches
    ``target/2``): 2 MHz -> 4, 1.67 MHz -> 3, 1 MHz -> 2, <= 500 kHz -> 1.
    ``target_hz`` of ``None``/``0`` means "keep the native rate" (returns 1).
    """
    if not target_hz or target_hz <= 0 or not fs_hz or fs_hz <= 0:
        return 1
    return max(1, int(np.floor(fs_hz / target_hz * (1 + 1e-6))))


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
    window: str = "hann",
    clip_dc: bool = DEFAULT_CLIP_DC,
    fs: float = 1.0,
    clip_low: float = DEFAULT_CLIP_LOW,
    clip_high: float = DEFAULT_CLIP_HIGH,
) -> np.ndarray:
    win = signal.get_window(window, n_fft)
    transform = signal.ShortTimeFFT(win=win, hop=hop, fs=fs)
    sxx = transform.stft(arr)

    if sxx.shape[0] == 2:
        sxx = sxx[0] * np.conj(sxx[1])
    elif sxx.shape[0] == 1:
        sxx = sxx[0]

    sxx = np.abs(sxx)
    sxx = np.log1p(sxx)

    # DC clipping
    if clip_dc:
        sxx = sxx[1:, :]

    # Percentile clipping
    vmin, vmax = np.percentile(
        sxx,
        [clip_low, clip_high],
    )
    return np.clip(sxx, vmin, vmax)
