"""Read signals and spectrograms from common file formats.

:func:`load_signal` returns ``(array, fs)``: the array is a 1D signal or a
2D spectrogram, ``fs`` the sampling rate in Hz when the file says so (WAV,
FLAC and OGG headers; an ``fs``-like field in ``.npz``/``.mat``/``.h5``; a
time column in ``.csv``/``.txt``; or an ``_sr<fs>`` filename suffix such as
``shot_sr500000.npy``), else ``None``.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

SIGNAL_SUFFIXES = (
    ".npy",
    ".npz",
    ".wav",
    ".flac",
    ".ogg",
    ".csv",
    ".txt",
    ".mat",
    ".h5",
    ".hdf5",
)
# Directory inputs skip text files (READMEs, logs) -- pass those explicitly.
DIRECTORY_SUFFIXES = tuple(s for s in SIGNAL_SUFFIXES if s not in (".csv", ".txt"))
DATA_KEYS = ("data", "signal", "x", "spectrogram", "values", "y")
FS_KEYS = ("fs", "sr", "sample_rate", "sampling_rate")
_SR_SUFFIX = re.compile(r"_sr(\d+(?:\.\d+)?)$")


def load_signal(path: str | Path) -> tuple[np.ndarray, float | None]:
    """Load a 1D signal or 2D spectrogram and its sampling rate.

    Parameters
    ----------
    path
        A file with one of the suffixes in :data:`SIGNAL_SUFFIXES`.

    Returns
    -------
    tuple
        ``(array, fs)``; ``(1, N)`` and ``(N, 1)`` arrays are flattened to
        1D, and ``fs`` is ``None`` when the file does not record it.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a file.
    ValueError
        For an unsupported suffix, a container whose signal array cannot be
        identified, a text file that is not a numeric table, or no numeric
        data (an empty or header-only table, an empty or 0-d array, or a
        non-numeric dtype such as strings or bool).
    ImportError
        For FLAC/OGG without ``soundfile`` or HDF5 without ``h5py``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(
            f"unsupported input format {path.suffix!r} ({path}); supported: "
            f"{', '.join(SIGNAL_SUFFIXES)}"
        )
    data, fs = loader(path)
    if fs is None:
        fs = fs_from_name(path)
    arr = _squeeze_vector(np.asarray(data))
    detail = _no_data(arr)
    if detail is not None:
        raise ValueError(f"{path}: no numeric data ({detail})")
    return arr, fs


def fs_from_name(path: str | Path) -> float | None:
    """Sampling rate from an ``_sr<fs>`` stem suffix (``x_sr200000.npy``)."""
    match = _SR_SUFFIX.search(Path(path).stem)
    if match is None:
        return None
    return _positive(float(match.group(1)))


def _positive(value: Any) -> float | None:
    try:
        fs = float(value)
    except (TypeError, ValueError):
        return None
    return fs if math.isfinite(fs) and fs > 0 else None


def _squeeze_vector(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2 and 1 in arr.shape:
        return arr.reshape(-1)
    return arr


def _is_numeric(arr: np.ndarray) -> bool:
    return arr.dtype != np.bool_ and np.issubdtype(arr.dtype, np.number)


def _no_data(arr: np.ndarray) -> str | None:
    """Why ``arr`` holds no signal, or ``None`` if it does."""
    if arr.ndim == 0:
        return "a single value, not a signal"
    if arr.size == 0:
        return "empty array"
    if not _is_numeric(arr):
        return f"dtype {arr.dtype}"
    return None


def _fs_from_mapping(arrays: Mapping[str, Any]) -> float | None:
    for key in FS_KEYS:
        if key in arrays:
            value = np.asarray(arrays[key]).squeeze()
            if value.size == 1 and _is_numeric(value):
                fs = _positive(value)
                if fs is not None:
                    return fs
    return None


def _from_mapping(
    arrays: Mapping[str, Any], path: Path
) -> tuple[np.ndarray, float | None]:
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    fs = _fs_from_mapping(arrays)
    for key in DATA_KEYS:
        if key in arrays and _is_numeric(arrays[key]) and arrays[key].size > 1:
            return arrays[key], fs
    candidates = [
        key
        for key, value in arrays.items()
        if key not in FS_KEYS and _is_numeric(value) and value.size > 1
    ]
    if len(candidates) == 1:
        return arrays[candidates[0]], fs
    raise ValueError(
        f"{path}: cannot tell which array is the signal (found "
        f"{sorted(arrays)}); name it one of {list(DATA_KEYS)}"
    )


def _load_npy(path: Path) -> tuple[np.ndarray, float | None]:
    return np.load(path, allow_pickle=False), None


def _load_npz(path: Path) -> tuple[np.ndarray, float | None]:
    with np.load(path, allow_pickle=False) as data:
        return _from_mapping({key: data[key] for key in data.files}, path)


def _load_mat(path: Path) -> tuple[np.ndarray, float | None]:
    from scipy.io import loadmat

    try:
        raw = loadmat(path)
    except NotImplementedError:  # MATLAB v7.3 files are HDF5
        return _load_h5(path)
    return _from_mapping({k: v for k, v in raw.items() if not k.startswith("__")}, path)


def _audio_to_float(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.uint8:
        return (data.astype(np.float64) - 128.0) / 128.0
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float64) / (np.iinfo(data.dtype).max + 1.0)
    return data.astype(np.float64)


def _mono(data: np.ndarray, path: Path) -> np.ndarray:
    if data.ndim == 2:
        warnings.warn(
            f"{path.name}: averaging {data.shape[1]} audio channels to mono",
            UserWarning,
            stacklevel=4,
        )
        return data.mean(axis=1)
    return data


def _load_wav(path: Path) -> tuple[np.ndarray, float | None]:
    from scipy.io import wavfile

    rate, data = wavfile.read(path)
    return _mono(_audio_to_float(data), path), float(rate)


def _load_soundfile(path: Path) -> tuple[np.ndarray, float | None]:
    try:
        import soundfile
    except ImportError as exc:
        raise ImportError(
            f"reading {path.suffix} files needs soundfile: pip install "
            "soundfile (included in 'tokeye[app]')"
        ) from exc
    data, rate = soundfile.read(path, dtype="float64", always_2d=False)
    return _mono(data, path), float(rate)


def _load_text(path: Path) -> tuple[np.ndarray, float | None]:
    delimiter = "," if path.suffix.lower() == ".csv" else None
    # An empty table is reported by load_signal, so numpy's warning is noise.
    # Not thread-safe (the filter list is global): at worst it leaks through.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"loadtxt: input contained no data", category=UserWarning
        )
        try:
            table = np.loadtxt(path, delimiter=delimiter, ndmin=1)
        except ValueError:  # a header row
            try:
                table = np.loadtxt(path, delimiter=delimiter, ndmin=1, skiprows=1)
            except ValueError as exc:
                raise ValueError(f"{path}: not a numeric table ({exc})") from exc
    if table.ndim == 2 and table.shape[1] == 2 and table.shape[0] > 2:
        dt = np.diff(table[:, 0])
        if np.all(dt > 0):  # (time, value) columns
            return table[:, 1], _positive(1.0 / float(np.median(dt)))
    return table, None


def _load_h5(path: Path) -> tuple[np.ndarray, float | None]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "reading HDF5 files needs h5py: pip install h5py (or 'tokeye[hdf5]')"
        ) from exc

    with h5py.File(path, "r") as fh:
        datasets: dict[str, Any] = {}

        def collect(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj

        fh.visititems(collect)
        numeric = [
            name
            for name, dset in datasets.items()
            if np.issubdtype(dset.dtype, np.number) and dset.size > 1
        ]
        by_basename = {name.rsplit("/", 1)[-1]: name for name in numeric}
        chosen = next((by_basename[k] for k in DATA_KEYS if k in by_basename), None)
        if chosen is None:
            if len(numeric) != 1:
                raise ValueError(
                    f"{path}: cannot tell which dataset is the signal (found "
                    f"{sorted(numeric)}); name it one of {list(DATA_KEYS)}"
                )
            chosen = numeric[0]
        dset = datasets[chosen]
        data = np.asarray(dset[()])
        fs = (
            _fs_from_mapping(dict(dset.attrs))
            or _fs_from_mapping(dict(fh.attrs))
            or _fs_from_mapping(
                {k: fh[k][()] for k in FS_KEYS if isinstance(fh.get(k), h5py.Dataset)}
            )
        )
    return data, fs


_LOADERS: dict[str, Callable[[Path], tuple[np.ndarray, float | None]]] = {
    ".npy": _load_npy,
    ".npz": _load_npz,
    ".wav": _load_wav,
    ".flac": _load_soundfile,
    ".ogg": _load_soundfile,
    ".csv": _load_text,
    ".txt": _load_text,
    ".mat": _load_mat,
    ".h5": _load_h5,
    ".hdf5": _load_h5,
}
