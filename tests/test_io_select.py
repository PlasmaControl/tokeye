"""Which array ``load_signal`` reads from a container, and ``key=``.

One selection rule serves ``.npz``, ``.mat`` and ``.h5``: the one candidate
named ``data``/``signal``/``x``/``spectrogram``/``values``/``y``, else the
only numeric candidate; anything else raises and lists the candidates.
Paths are compared as ``str(path)``, never put in a regex.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from tokeye.io import CONTAINER_SUFFIXES, load_signal

SIG = np.sin(np.linspace(0, 20, 500))
HINT = " Or name it one of: data, signal, x, spectrogram, values, y."
CHOOSE = "Choose one with key= (--key on the command line)."
V73_HEADER = b"MATLAB 7.3 MAT-file".ljust(124, b" ") + b"\x00\x02IM"


def _message(path, key=None) -> str:
    with pytest.raises(ValueError) as excinfo:
        load_signal(path, key=key)
    return str(excinfo.value)


def _npz(tmp_path, name="x.npz", **arrays):
    path = tmp_path / name
    np.savez(path, **arrays)
    return path


def _h5(tmp_path, build, name="x.h5"):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / name
    with h5py.File(path, "w") as fh:
        build(fh)
    return path


def _write_v73(path, variables, attrs=None):
    """A MATLAB v7.3 ``.mat``: HDF5 behind a 512-byte MATLAB header.

    Each array is stored transposed, as MATLAB lays out column-major data;
    a name with ``/`` in it makes h5py create the group.
    """
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w", userblock_size=512) as fh:
        for name, value in variables.items():
            dset = fh.create_dataset(name, data=np.asarray(value).T)
            for attr, attr_value in (attrs or {}).get(name, {}).items():
                dset.attrs[attr] = attr_value
    with path.open("r+b") as fh:
        fh.write(V73_HEADER)
    return path


def test_container_suffixes():
    assert CONTAINER_SUFFIXES == (".npz", ".mat", ".h5", ".hdf5")


# ---------------------------------------------------------------------------
# .npz
# ---------------------------------------------------------------------------


def test_npz_x_and_y_raise_and_list_both(tmp_path):
    path = _npz(tmp_path, x=np.arange(500) / 1000.0, y=SIG)

    text = _message(path)

    assert text == (
        f"{path}: cannot tell which array is the signal; 2 candidates: x, y. {CHOOSE}"
    )


def test_npz_key_selects_the_array(tmp_path):
    t = np.arange(500) / 1000.0
    path = _npz(tmp_path, x=t, y=SIG)

    np.testing.assert_array_equal(load_signal(path, key="y")[0], SIG)
    np.testing.assert_array_equal(load_signal(path, key="x")[0], t)


def test_npz_candidates_leave_out_fs_strings_and_bools(tmp_path):
    path = _npz(
        tmp_path,
        a=SIG,
        b=SIG,
        fs=np.array(1000.0),
        label=np.array("abc"),
        flags=SIG > 0,
    )

    text = _message(path)

    assert "2 candidates: a, b." in text
    assert text.endswith(f"{CHOOSE}{HINT}")


def test_npz_lists_at_most_20_candidates(tmp_path):
    path = _npz(tmp_path, **{f"a{i:02d}": SIG for i in range(25)})

    text = _message(path)

    listed = ", ".join(f"a{i:02d}" for i in range(20))
    assert f"25 candidates: {listed}, … and 5 more. " in text


def test_npz_object_entry_is_never_loaded(tmp_path):
    path = _npz(tmp_path, signal=SIG, meta=np.array([{"a": 1}], dtype=object))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data, _ = load_signal(path)

    np.testing.assert_array_equal(data, SIG)
    assert caught == []
    assert _message(path, key="meta") == (
        f"{path}: 'meta' is not a numeric array with more than one element"
    )


def test_npz_reads_only_the_chosen_entry(tmp_path, monkeypatch):
    path = _npz(tmp_path, signal=SIG, other=np.ones(1000))
    calls = []
    real = np.lib.format.read_array

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(np.lib.format, "read_array", counting)

    data, _ = load_signal(path)

    assert len(calls) == 1
    np.testing.assert_array_equal(data, SIG)


def test_npz_without_a_usable_array(tmp_path):
    path = _npz(tmp_path, fs=np.array(10.0), label=np.array("abc"))

    assert _message(path) == (
        f"{path}: no numeric array with more than one element, other than fs "
        "fields (entries: fs, label)"
    )


# ---------------------------------------------------------------------------
# fs fields match case-insensitively
# ---------------------------------------------------------------------------


def test_npz_capitalized_fs(tmp_path):
    assert load_signal(_npz(tmp_path, signal=SIG, Fs=1000.0))[1] == 1000.0


def test_mat_capitalized_fs(tmp_path):
    savemat(tmp_path / "x.mat", {"x": SIG, "Fs": 250.0})
    assert load_signal(tmp_path / "x.mat")[1] == 250.0


def test_h5_uppercase_root_fs(tmp_path):
    def build(fh):
        fh.create_dataset("trace", data=SIG)
        fh.create_dataset("FS", data=5000.0)

    assert load_signal(_h5(tmp_path, build))[1] == 5000.0


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [(1.0, 2.0, 1.0), (0.0, 2.0, 2.0)],
    ids=["both", "zero"],
)
def test_lowercase_fs_is_tried_first(tmp_path, lower, upper, expected):
    path = _npz(tmp_path, signal=SIG, fs=lower, Fs=upper)
    assert load_signal(path)[1] == expected


# ---------------------------------------------------------------------------
# Key errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".npy", ".wav", ".csv"])
def test_key_on_a_single_array_file(tmp_path, suffix):
    path = tmp_path / f"a{suffix}"
    path.write_bytes(b"")

    assert _message(path, key="y") == (
        f"{path}: key= selects an array inside .npz, .mat, .h5 and .hdf5 files; "
        f"a {suffix} file holds one array"
    )


@pytest.mark.parametrize("key", ["", "  "], ids=["empty", "blank"])
def test_an_empty_key_is_rejected(tmp_path, key):
    path = _npz(tmp_path, x=SIG, y=SIG)
    with pytest.raises(ValueError, match="key must be a non-empty string"):
        load_signal(path, key=key)


def test_a_key_must_be_a_string(tmp_path):
    path = _npz(tmp_path, x=SIG, y=SIG)
    with pytest.raises(TypeError, match="key must be a string, got int"):
        load_signal(path, key=3)


def test_key_checks_run_in_order(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_signal(tmp_path / "missing.npz", key=3)
    (tmp_path / "x.json").write_text("{}")
    with pytest.raises(ValueError, match="unsupported input format"):
        load_signal(tmp_path / "x.json", key="y")


def test_an_unknown_key_lists_the_candidates(tmp_path):
    path = _npz(tmp_path, x=SIG, y=SIG)

    assert _message(path, key="nope") == (
        f"{path}: no array named 'nope'; candidates: x, y"
    )


# ---------------------------------------------------------------------------
# .h5
# ---------------------------------------------------------------------------


def test_h5_datasets_sharing_a_basename_are_ambiguous(tmp_path):
    def build(fh):
        fh.create_dataset("g1/data", data=SIG)
        fh.create_dataset("g2/data", data=SIG)

    path = _h5(tmp_path, build)
    text = _message(path)

    assert "2 candidates: g1/data, g2/data." in text
    assert HINT not in text


@pytest.fixture
def tree_h5(tmp_path):
    def build(fh):
        fh.create_dataset("a/sig/data", data=SIG)
        fh.create_dataset("b/sig/data", data=2 * SIG)
        fh.create_dataset("c/other/data", data=3 * SIG)

    return _h5(tmp_path, build)


@pytest.mark.parametrize("key", ["c/other/data", "other/data", "/c/other/data"])
def test_h5_key_by_path_or_suffix(tree_h5, key):
    np.testing.assert_array_equal(load_signal(tree_h5, key=key)[0], 3 * SIG)


@pytest.mark.parametrize(
    "key", ["/other/data", "ther/data"], ids=["absolute-is-exact", "whole-parts"]
)
def test_h5_key_matching_nothing(tree_h5, key):
    assert _message(tree_h5, key=key) == (
        f"{tree_h5}: no array named {key!r}; candidates: a/sig/data, "
        "b/sig/data, c/other/data"
    )


def test_h5_key_matching_several(tree_h5):
    assert _message(tree_h5, key="sig/data") == (
        f"{tree_h5}: key 'sig/data' matches 2 arrays: a/sig/data, b/sig/data. "
        "Give more of the path."
    )
    assert "key 'data' matches 3 arrays: " in _message(tree_h5, key="data")


def test_h5_exact_name_wins_over_a_suffix(tmp_path):
    def build(fh):
        fh.create_dataset("data", data=SIG)
        fh.create_dataset("g/data", data=2 * SIG)

    np.testing.assert_array_equal(load_signal(_h5(tmp_path, build), key="data")[0], SIG)


def test_h5_null_dataspace_is_skipped(tmp_path):
    h5py = pytest.importorskip("h5py")

    def build(fh):
        fh.create_dataset("empty", data=h5py.Empty("f"))
        fh.create_dataset("signal", data=SIG)

    np.testing.assert_array_equal(load_signal(_h5(tmp_path, build))[0], SIG)


def test_h5_candidates_are_numeric_only(tmp_path):
    h5py = pytest.importorskip("h5py")

    def build(fh):
        fh.create_dataset("a", data=SIG)
        fh.create_dataset("b", data=SIG)
        fh.create_dataset("names", data=["x", "y"], dtype=h5py.string_dtype())
        fh.create_dataset("fs", data=1000.0)

    assert "2 candidates: a, b." in _message(_h5(tmp_path, build))


def test_h5_2d_dataset_keeps_its_orientation(tmp_path):
    spec = np.arange(24.0).reshape(6, 4)

    def build(fh):
        fh.create_dataset("spec", data=spec)

    np.testing.assert_array_equal(load_signal(_h5(tmp_path, build))[0], spec)


def test_h5_reads_only_the_chosen_dataset(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")

    def build(fh):
        fh.create_dataset("signal", data=SIG)
        fh.create_dataset("other", data=np.ones(100))
        fh.create_dataset("big", data=np.zeros(10_000))

    path = _h5(tmp_path, build)
    reads = []
    getitem, array = h5py.Dataset.__getitem__, h5py.Dataset.__array__

    def spy_getitem(self, *args, **kwargs):
        reads.append(self.name)
        return getitem(self, *args, **kwargs)

    def spy_array(self, *args, **kwargs):
        reads.append(self.name)
        return array(self, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", spy_getitem)
    monkeypatch.setattr(h5py.Dataset, "__array__", spy_array)

    data, _ = load_signal(path)

    np.testing.assert_array_equal(data, SIG)
    assert set(reads) == {"/signal"}


def test_h5_key_follows_a_soft_link(tmp_path):
    h5py = pytest.importorskip("h5py")

    def build(fh):
        fh.create_dataset("signal", data=SIG)
        fh["alias"] = h5py.SoftLink("/signal")

    np.testing.assert_array_equal(
        load_signal(_h5(tmp_path, build), key="alias")[0], SIG
    )


# ---------------------------------------------------------------------------
# .mat (v5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variables",
    [{"sig": SIG, "mask": SIG > 0}, {"x": SIG > 0, "sig": SIG}],
    ids=["mask", "logical-x"],
)
def test_v5_logicals_are_not_candidates(tmp_path, variables):
    savemat(tmp_path / "x.mat", variables)
    np.testing.assert_array_equal(load_signal(tmp_path / "x.mat")[0], SIG)


# ---------------------------------------------------------------------------
# .mat (v7.3, HDF5)
# ---------------------------------------------------------------------------


def test_v73_fixture_is_not_readable_by_scipy(tmp_path):
    path = _write_v73(tmp_path / "x.mat", {"sig": SIG})
    with pytest.raises(NotImplementedError):
        loadmat(path)


def test_v73_2d_keeps_matlab_orientation(tmp_path):
    matrix = np.arange(24.0).reshape(6, 4)
    path = _write_v73(
        tmp_path / "x.mat",
        {"M": matrix},
        attrs={"M": {"MATLAB_class": np.bytes_(b"double")}},
    )

    data, _ = load_signal(path)

    assert data.shape == (6, 4)
    np.testing.assert_array_equal(data, matrix)


def test_v73_row_vector_and_root_fs(tmp_path):
    path = _write_v73(tmp_path / "x.mat", {"x": SIG[np.newaxis], "fs": [[500.0]]})

    data, fs = load_signal(path)

    np.testing.assert_array_equal(data, SIG)
    assert fs == 500.0


def test_v73_key_selects_a_variable(tmp_path):
    matrix = np.arange(24.0).reshape(6, 4)
    path = _write_v73(tmp_path / "x.mat", {"M": matrix, "sig": SIG})

    np.testing.assert_array_equal(load_signal(path, key="M")[0], matrix)


V73_NEIGHBOURS = {
    "char-bytes": (
        {"name": np.array([104, 105], dtype=np.uint16)},
        {"name": {"MATLAB_class": np.bytes_(b"char")}},
    ),
    "char-str": (
        {"name": np.array([104, 105], dtype=np.uint16)},
        {"name": {"MATLAB_class": "char"}},
    ),
    "refs": ({"#refs#/a": np.ones(5)}, {}),
    "struct": ({"info/x": np.ones(5)}, {}),
    "sparse": (
        {
            "S/data": np.ones(3),
            "S/ir": np.arange(3, dtype=np.uint64),
            "S/jc": np.arange(4, dtype=np.uint64),
        },
        {},
    ),
    "empty": (
        {"x": np.array([0, 0], dtype=np.uint64)},
        {"x": {"MATLAB_class": np.bytes_(b"double"), "MATLAB_empty": np.uint8(1)}},
    ),
}


@pytest.mark.parametrize("case", list(V73_NEIGHBOURS))
def test_v73_non_numeric_variables_are_not_candidates(tmp_path, case):
    variables, attrs = V73_NEIGHBOURS[case]
    path = _write_v73(tmp_path / "x.mat", {"sig": SIG, **variables}, attrs=attrs)

    np.testing.assert_array_equal(load_signal(path)[0], SIG)


def test_v73_without_h5py_names_matlab(tmp_path, monkeypatch):
    path = tmp_path / "x.mat"
    path.write_bytes(V73_HEADER + b"\x00" * 384)
    monkeypatch.setitem(sys.modules, "h5py", None)

    with pytest.raises(ImportError, match=r"MATLAB v7\.3 \.mat.*tokeye\[hdf5\]"):
        load_signal(path)


def test_plain_h5_without_h5py_keeps_its_hint(tmp_path, monkeypatch):
    path = tmp_path / "x.h5"
    path.write_bytes(b"not really hdf5")
    monkeypatch.setitem(sys.modules, "h5py", None)

    with pytest.raises(ImportError, match="reading HDF5 files needs h5py"):
        load_signal(path)


# ---------------------------------------------------------------------------
# Text tables: time units and 2-column rules
# ---------------------------------------------------------------------------


def _table(path, header, step, n=200, sep=","):
    t = np.arange(n) * step
    rows = [
        f"{a!r}{sep}{b!r}" for a, b in zip(t.tolist(), SIG[:n].tolist(), strict=True)
    ]
    lines = ([header] if header is not None else []) + rows
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("header", "step", "fs"),
    [
        ("time_ms,value", 0.5, 2000.0),
        ("t (ms),v", 0.5, 2000.0),
        ("Time [us],v", 2.0, 500_000.0),
        ("t (µs),v", 2.0, 500_000.0),
        ("t_us,v", 2.0, 500_000.0),
        ("Time_MS,v", 0.5, 2000.0),
        ("t[ms],v", 0.5, 2000.0),
        ("time,value", 0.001, 1000.0),
        ("items,value", 1.0, 1.0),
        ("rms,value", 1.0, 1.0),
        ("mass,value", 1.0, 1.0),
    ],
)
def test_time_unit_from_the_csv_header(tmp_path, header, step, fs):
    path = _table(tmp_path / "x.csv", header, step)

    data, got = load_signal(path)

    np.testing.assert_allclose(data, SIG[:200])
    assert got == pytest.approx(fs)


def test_time_unit_from_a_whitespace_txt_header(tmp_path):
    path = _table(tmp_path / "x.txt", "t (ms) value", 0.5, sep=" ")
    assert load_signal(path)[1] == pytest.approx(2000.0)


def test_time_unit_from_a_savetxt_comment_header(tmp_path):
    path = tmp_path / "x.csv"
    t = np.arange(200) * 0.5
    np.savetxt(
        path, np.stack([t, SIG[:200]], axis=1), delimiter=",", header="time_ms,value"
    )

    assert load_signal(path)[1] == pytest.approx(2000.0)


def test_a_comment_that_is_not_a_header_is_ignored(tmp_path):
    path = _table(tmp_path / "x.csv", "# sampled every 2 ms", 0.5)
    assert load_signal(path)[1] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "time",
    [[0, 0, 1, 1, 2, 2], [0, 1]],
    ids=["repeated-time", "two-rows"],
)
def test_a_2_column_table_needs_increasing_time(tmp_path, time):
    path = tmp_path / "x.csv"
    rows = [f"{t},{v}" for t, v in zip(time, SIG, strict=False)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    text = _message(path)

    assert text.startswith(f"{path}: ")
    assert "(time, value)" in text


def test_a_3_column_table_is_unchanged(tmp_path):
    table = np.stack([np.arange(10.0), np.ones(10), np.zeros(10)], axis=1)
    np.savetxt(tmp_path / "x.csv", table, delimiter=",")

    data, fs = load_signal(tmp_path / "x.csv")

    assert data.shape == (10, 3)
    assert fs is None
