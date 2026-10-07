"""Read signals and spectrograms from common file formats.

:func:`load_signal` returns ``(array, fs)``: the array is a 1D signal or a
2D spectrogram, ``fs`` the sampling rate in Hz when the file says so (WAV,
FLAC and OGG headers; an ``fs``-like field in ``.npz``/``.mat``/``.h5``; a
time column in ``.csv``/``.txt``; or an ``_sr<fs>`` filename suffix such as
``shot_sr500000.npy``), else ``None``.

A container (``.npz``, ``.mat``, ``.h5``/``.hdf5``) can hold several
arrays. TokEye reads the one named ``data``, ``signal``, ``x``,
``spectrogram``, ``values`` or ``y``, else the only numeric one; when
several qualify it raises and lists them, and ``key=`` picks one.
"""

from __future__ import annotations

import math
import re
import warnings
import zipfile
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

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
# The suffixes whose files hold several named arrays, so key= applies.
CONTAINER_SUFFIXES = (".npz", ".mat", ".h5", ".hdf5")
DATA_KEYS = ("data", "signal", "x", "spectrogram", "values", "y")
FS_KEYS = ("fs", "sr", "sample_rate", "sampling_rate")
_SR_SUFFIX = re.compile(r"_sr(\d+(?:\.\d+)?)$")
_LIST_LIMIT = 20
# What reading a damaged .npz member header can raise.
_NPZ_ERRORS = (ValueError, OSError, EOFError, zipfile.BadZipFile, zlib.error)
# MATLAB classes that hold numbers; char, logical, cell, struct and objects
# such as string do not.
_MATLAB_NUMERIC = frozenset(
    ("double", "single", "int8", "uint8", "int16", "uint16")
    + ("int32", "uint32", "int64", "uint64")
)
_MATLAB_INTERNAL = ("#refs#", "#subsystem#")
# A time unit in a column name: a standalone token, no ASCII letter around it.
_TIME_UNIT = re.compile(
    r"(?<![A-Za-z])(ms|msec|us|µs|μs|usec)(?![A-Za-z])", re.IGNORECASE
)
_BRACKETED = re.compile(r"\(.*\)|\[.*\]")
# The time column's name in a '#' comment header: it starts with any Unicode
# letter or '_' (a micro sign included), continues as one token without '='
# or ':' (those are metadata, dt=1ms), then optionally a bracketed or
# parenthesised unit (time [ms], t(ms), t_ms, time).
_HEADER_NAME = re.compile(r"[^\W\d][^\s()\[\]=:]*(?:\s*(?:\([^()]*\)|\[[^\[\]]*\]))?")


def load_signal(
    path: str | Path, key: str | None = None
) -> tuple[np.ndarray, float | None]:
    """Load a 1D signal or 2D spectrogram and its sampling rate.

    Parameters
    ----------
    path
        A file with one of the suffixes in :data:`SIGNAL_SUFFIXES`.
    key
        Which array to read from a container (``.npz``, ``.mat``, ``.h5``,
        ``.hdf5``): an entry name, or an HDF5 dataset path (``/`` first for
        an exact path from the root) or its trailing whole path components,
        such as ``tree/pointname/data``. By default, the one array named
        ``data``, ``signal``, ``x``, ``spectrogram``, ``values`` or ``y``,
        else the only numeric array with more than one element.

    Returns
    -------
    tuple
        ``(array, fs)``; ``(1, N)`` and ``(N, 1)`` arrays are flattened to
        1D, and ``fs`` is ``None`` when the file does not record it. A
        2-column ``.csv``/``.txt`` is read as (time, value), with time in
        seconds unless the header names the column in ``ms`` or ``us``.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not a file.
    TypeError
        ``key`` is not a string.
    ValueError
        For an unsupported suffix; an empty ``key``, or a ``key`` on a file
        that holds one array; a container whose signal array cannot be
        identified (the message lists the candidates), or a ``key`` that
        matches no array, several arrays, or one that is not numeric; a
        text file that is not a numeric table, or a 2-column table whose
        first column is not a strictly increasing time; or no numeric data
        (an empty or header-only table, an empty or 0-d array, or a
        non-numeric dtype such as strings or bool).
    ImportError
        For FLAC/OGG without ``soundfile``, or HDF5 (including MATLAB v7.3
        ``.mat``) without ``h5py``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    if key is not None:
        _check_key(key)
    suffix = path.suffix.lower()
    if suffix in _CONTAINER_LOADERS:
        data, fs = _CONTAINER_LOADERS[suffix](path, key)
    elif suffix not in _LOADERS:
        raise ValueError(
            f"unsupported input format {path.suffix!r} ({path}); supported: "
            f"{', '.join(SIGNAL_SUFFIXES)}"
        )
    elif key is not None:
        raise ValueError(
            f"{path}: key= selects an array inside .npz, .mat, .h5 and .hdf5 "
            f"files; a {suffix} file holds one array"
        )
    else:
        data, fs = _LOADERS[suffix](path)
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


def _check_key(key: object) -> None:
    """Reject a ``key`` that is not a non-empty string."""
    if not isinstance(key, str):
        raise TypeError(f"key must be a string, got {type(key).__name__}")
    if not key.strip():
        raise ValueError("key must be a non-empty string")


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


def _is_numeric_dtype(dtype: np.dtype) -> bool:
    return dtype != np.bool_ and np.issubdtype(dtype, np.number)


def _is_numeric(arr: np.ndarray) -> bool:
    return _is_numeric_dtype(arr.dtype)


def _no_data(arr: np.ndarray) -> str | None:
    """Why ``arr`` holds no signal, or ``None`` if it does."""
    if arr.ndim == 0:
        return "a single value, not a signal"
    if arr.size == 0:
        return "empty array"
    if not _is_numeric(arr):
        return f"dtype {arr.dtype}"
    return None


def _fs_from_mapping(mapping: Mapping[str, Any]) -> float | None:
    """The first valid ``FS_KEYS`` value in ``mapping``, ignoring case.

    For each name in ``FS_KEYS`` order, the exact lower-case spelling is
    tried first, then the other spellings in mapping order. Only matching
    values are read.
    """
    names = [name for name in mapping if isinstance(name, str)]
    for fs_key in FS_KEYS:
        matching = [name for name in names if name.lower() == fs_key]
        matching.sort(key=lambda name: name != fs_key)  # stable
        for name in matching:
            value = np.asarray(mapping[name]).squeeze()
            if value.size == 1 and _is_numeric(value):
                fs = _positive(value)
                if fs is not None:
                    return fs
    return None


def _basename(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _list_names(names: Iterable[str]) -> str:
    """Sorted names joined by ``", "``: the first 20, then a count."""
    ordered = sorted(names)
    text = ", ".join(ordered[:_LIST_LIMIT])
    if len(ordered) > _LIST_LIMIT:
        text += f", … and {len(ordered) - _LIST_LIMIT} more"
    return text


def _cannot_tell(path: Path, names: list[str], *, hint: bool) -> ValueError:
    text = (
        f"{path}: cannot tell which array is the signal; {len(names)} "
        f"candidates: {_list_names(names)}. Choose one with key= (--key on the "
        "command line)."
    )
    if hint:
        text += f" Or name it one of: {', '.join(DATA_KEYS)}."
    return ValueError(text)


def _not_usable(path: Path, name: str) -> ValueError:
    return ValueError(
        f"{path}: {name!r} is not a numeric array with more than one element"
    )


def _choose(path: Path, entries: Mapping[str, bool], key: str | None) -> str:
    """The name of the entry to read: the one selection rule for containers.

    Parameters
    ----------
    path
        The file, for the messages.
    entries
        Every entry name (an npz key, a ``.mat`` variable or an HDF5
        dataset path without the leading ``/``) mapped to whether it is a
        usable array: numeric, not bool, with more than one element. Values
        are read only when needed: all of them without a key or when the
        key matches nothing (to list the candidates), else only the match's.
    key
        The caller's ``key=``, already checked by :func:`_check_key`.

    Raises
    ------
    ValueError
        No entry, or several, qualify; or ``key`` matches none, several, or
        an entry that is not usable.
    """

    def usable_names() -> list[str]:
        return [
            name
            for name, usable in entries.items()
            if usable and _basename(name).lower() not in FS_KEYS
        ]

    if key is None:
        candidates = usable_names()
        named = [name for name in candidates if _basename(name) in DATA_KEYS]
        if len(named) == 1:
            return named[0]
        if named:
            raise _cannot_tell(path, named, hint=False)
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            raise _cannot_tell(path, candidates, hint=True)
        listed = f"entries: {_list_names(entries)}" if entries else "no entries"
        raise ValueError(
            f"{path}: no numeric array with more than one element, other than "
            f"fs fields ({listed})"
        )

    if key.startswith("/"):
        matches = [key[1:]] if key[1:] in entries else []
    elif key in entries:
        matches = [key]
    else:  # a suffix of whole path components
        matches = [name for name in entries if name.endswith("/" + key)]
    if not matches:
        raise ValueError(
            f"{path}: no array named {key!r}; candidates: "
            f"{_list_names(usable_names()) or 'none'}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"{path}: key {key!r} matches {len(matches)} arrays: "
            f"{_list_names(matches)}. Give more of the path."
        )
    if not entries[matches[0]]:
        raise _not_usable(path, matches[0])
    return matches[0]


def _load_npy(path: Path) -> tuple[np.ndarray, float | None]:
    return np.load(path, allow_pickle=False), None


def _npz_headers(data: Any) -> dict[str, tuple[tuple[int, ...], np.dtype] | None]:
    """Each entry's ``(shape, dtype)`` from its ``.npy`` header, or ``None``.

    Only the headers are read, never the data.
    """
    headers: dict[str, tuple[tuple[int, ...], np.dtype] | None] = {}
    for member in data.zip.namelist():
        name = member[:-4] if member.endswith(".npy") else member
        try:
            with data.zip.open(member) as fh:
                version = np.lib.format.read_magic(fh)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(fh)
                else:
                    shape, _, dtype = np.lib.format.read_array_header_2_0(fh)
        except _NPZ_ERRORS:
            headers[name] = None
        else:
            headers[name] = (shape, dtype)
    return headers


def _load_npz(path: Path, key: str | None) -> tuple[np.ndarray, float | None]:
    with np.load(path, allow_pickle=False) as data:
        headers = _npz_headers(data)
        entries = {
            name: header is not None
            and _is_numeric_dtype(header[1])
            and math.prod(header[0]) > 1
            for name, header in headers.items()
        }
        name = _choose(path, entries, key)
        fs_fields = {}
        for field, header in headers.items():
            if field.lower() in FS_KEYS and header and math.prod(header[0]) == 1:
                try:
                    fs_fields[field] = data[field]
                except _NPZ_ERRORS:
                    continue
        return data[name], _fs_from_mapping(fs_fields)


def _load_mat(path: Path, key: str | None) -> tuple[np.ndarray, float | None]:
    from scipy.io import loadmat

    try:
        raw = loadmat(path)
    except NotImplementedError:  # MATLAB v7.3 files are HDF5, column-major
        data, fs = _load_h5(path, key, matlab=True)
        return data.T, fs
    arrays = {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}
    entries = {k: _is_numeric(v) and v.size > 1 for k, v in arrays.items()}
    if any(entries[k] and v.dtype == np.uint8 for k, v in arrays.items()):
        # Logicals load as uint8, like compact doubles; only the class tells.
        from scipy.io import whosmat

        logical = {name for name, _, cls in whosmat(path) if cls == "logical"}
        entries = {k: usable and k not in logical for k, usable in entries.items()}
    return arrays[_choose(path, entries, key)], _fs_from_mapping(arrays)


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
            'soundfile (included in "tokeye[app]")'
        ) from exc
    data, rate = soundfile.read(path, dtype="float64", always_2d=False)
    return _mono(data, path), float(rate)


def _column_names(line: str, *, csv: bool) -> list[str]:
    """A header row's column names; in a ``.txt``, ``t (ms)`` is one name."""
    if csv:
        return line.split(",")
    names: list[str] = []
    for token in line.split():
        if names and _BRACKETED.fullmatch(token):
            names[-1] += f" {token}"
        else:
            names.append(token)
    return names


def _unquote(name: str) -> str:
    """``name`` less one pair of surrounding quotes (``"`` or ``'``)."""
    if len(name) >= 2 and name[0] == name[-1] and name[0] in "\"'":
        return name[1:-1]
    return name


def _time_scale(path: Path, *, csv: bool, header_row: bool) -> float:
    """Seconds per unit of a 2-column table's time column.

    The unit comes from the first column name of the header row: the first
    line when it is not numeric (``header_row``), or a ``#`` comment first
    line with two names (``np.savetxt(header=...)`` writes one), judged by
    the time column's name alone, since the other is never read. That name,
    less one pair of surrounding quotes, starts with a letter (any Unicode
    letter) or ``_`` and is a single token without ``=`` or ``:``, plus an
    optional bracketed or parenthesised unit (``time [ms]``, ``t(ms)``,
    ``t_ms``, ``Δt [ms]``); the value column's name may be anything. Any
    other comment, such as ``# sampled at 2 ms, 1 kHz``, ``# 2ms, 1kHz`` or
    ``# dt=1ms, fs=1kHz``, is not a header. A
    standalone ``ms``/``msec`` gives 1e-3, ``us``/``µs``/``usec`` 1e-6;
    anything else is seconds.
    """
    with path.open(errors="replace") as fh:  # the default encoding, as loadtxt
        line = fh.readline().strip()
    if not header_row:
        if not line.startswith("#"):
            return 1.0
        line = line[1:].lstrip()
    names = _column_names(line, csv=csv)
    if not header_row and (
        len(names) != 2 or not _HEADER_NAME.fullmatch(_unquote(names[0].strip()))
    ):  # a comment, not a header
        return 1.0
    match = _TIME_UNIT.search(names[0]) if names else None
    if match is None:
        return 1.0
    return 1e-3 if match.group(1).lower() in ("ms", "msec") else 1e-6


def _load_text(path: Path) -> tuple[np.ndarray, float | None]:
    """A numeric table; a 2-column one is (time, value).

    The time column is in seconds unless the header names its unit: ``ms``
    or ``msec``, ``us``, ``µs`` or ``usec`` (see :func:`_time_scale`).
    """
    csv = path.suffix.lower() == ".csv"
    delimiter = "," if csv else None
    header_row = False
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
            header_row = True
    if table.ndim == 2 and table.shape[1] == 2:
        dt = np.diff(table[:, 0])
        if table.shape[0] <= 2 or not np.all(dt > 0):  # NaN fails dt > 0 too
            raise ValueError(
                f"{path}: a 2-column table is read as (time, value) and needs "
                "more than two rows with a strictly increasing first column; "
                "save a single signal column instead"
            )
        scale = _time_scale(path, csv=csv, header_row=header_row)
        return table[:, 1], _positive(1.0 / (float(np.median(dt)) * scale))
    return table, None


def _h5_usable(name: str, dset: Any) -> bool:
    """A numeric, non-bool dataset with more than one element.

    MATLAB's non-numeric classes (char, logical, cell, ...), its ``[]``
    placeholders and its ``#refs#``/``#subsystem#`` internals are not.
    """
    if name.lstrip("/").split("/", 1)[0] in _MATLAB_INTERNAL:
        return False
    if not _is_numeric_dtype(dset.dtype) or (dset.size or 0) <= 1:
        return False
    matlab_class = dset.attrs.get("MATLAB_class")
    if isinstance(matlab_class, bytes):  # np.bytes_ too
        matlab_class = matlab_class.decode("ascii", "replace")
    if matlab_class is not None and not (
        isinstance(matlab_class, str) and matlab_class in _MATLAB_NUMERIC
    ):
        return False
    empty = dset.attrs.get("MATLAB_empty")
    return empty is None or not np.any(np.asarray(empty) != 0)


class _H5Usability(Mapping):
    """HDF5 dataset name -> :func:`_h5_usable`, judged on first lookup.

    Judging reads attributes, which is slow over a large file on a network
    file system, so :func:`_choose` judges only the datasets it needs.
    """

    def __init__(self, datasets: dict[str, Any]) -> None:
        self._datasets = datasets
        self._usable: dict[str, bool] = {}

    def __getitem__(self, name: str) -> bool:
        if name not in self._usable:
            self._usable[name] = _h5_usable(name, self._datasets[name])
        return self._usable[name]

    def __contains__(self, name: object) -> bool:
        return name in self._datasets

    def __iter__(self) -> Iterator[str]:
        return iter(self._datasets)

    def __len__(self) -> int:
        return len(self._datasets)


def _load_h5(
    path: Path, key: str | None, *, matlab: bool = False
) -> tuple[np.ndarray, float | None]:
    """The selected dataset as h5py returns it (never transposed here).

    ``matlab`` is the MATLAB v7.3 ``.mat`` path: only root-level datasets are
    variables there, and a key is only matched against them.
    """
    try:
        import h5py
    except ImportError as exc:
        what = "MATLAB v7.3 .mat files" if matlab else "HDF5 files"
        raise ImportError(
            f'reading {what} needs h5py: pip install h5py (or "tokeye[hdf5]")'
        ) from exc

    with h5py.File(path, "r") as fh:
        dset = None
        if key is not None and not matlab:
            # Resolve the key itself first, without walking the file (which
            # takes seconds for a large one); visititems also lists a
            # hard-linked dataset under one name only and does not follow
            # soft links.
            name = key[1:] if key.startswith("/") else key
            obj = fh.get(name) if name else None
            if isinstance(obj, h5py.Dataset):
                if not _h5_usable(name, obj):
                    raise _not_usable(path, name)
                dset = obj
        if dset is None:
            datasets: dict[str, Any] = {}
            if matlab:
                for name, obj in fh.items():
                    if isinstance(obj, h5py.Dataset):
                        datasets[name] = obj
            else:

                def collect(name: str, obj: Any) -> None:
                    if isinstance(obj, h5py.Dataset):
                        datasets[name] = obj

                fh.visititems(collect)
            dset = datasets[_choose(path, _H5Usability(datasets), key)]
        data = np.asarray(dset[()])
        root_fs = {
            name: fh[name][()]
            for name in fh
            if name.lower() in FS_KEYS
            and isinstance(fh.get(name), h5py.Dataset)
            and fh[name].size == 1
        }
        fs = (
            _fs_from_mapping(dset.attrs)
            or _fs_from_mapping(fh.attrs)
            or _fs_from_mapping(root_fs)
        )
    return data, fs


_LOADERS: dict[str, Callable[[Path], tuple[np.ndarray, float | None]]] = {
    ".npy": _load_npy,
    ".wav": _load_wav,
    ".flac": _load_soundfile,
    ".ogg": _load_soundfile,
    ".csv": _load_text,
    ".txt": _load_text,
}
_CONTAINER_LOADERS: dict[
    str, Callable[[Path, str | None], tuple[np.ndarray, float | None]]
] = {
    ".npz": _load_npz,
    ".mat": _load_mat,
    ".h5": _load_h5,
    ".hdf5": _load_h5,
}
