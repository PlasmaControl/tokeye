"""Input and flag checks: colliding output stems, glob filtering, unreadable
or empty inputs, and range-checked numeric flags.

Every exit-2 case here happens before the model loads (a load spy that calls
``pytest.fail`` proves it) and leaves no output directory behind. Paths are
compared as ``str(path)``, never put in a regex: ``C:\\Users`` is not one.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn
from cli_helpers import _StubRCNN, _TransientStub, one_error_line

from tokeye import batch
from tokeye.cli import _options, build_parser, main
from tokeye.cli import alfvenspec as alfvenspec_cli
from tokeye.io import SIGNAL_SUFFIXES, load_signal

# The flags each command needs for it to write per-stem files.
PER_STEM = {"run": [], "elmspec": ["--png"], "alfvenspec": []}


def _no_load(*args, **kwargs):
    pytest.fail("the model was loaded")


@pytest.fixture
def no_load(monkeypatch):
    monkeypatch.setattr("tokeye.hub.load_model", _no_load)


@pytest.fixture
def out_dir(tmp_path):
    return tmp_path / "out"


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _spectrogram(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.random.default_rng(0).random((32, 64)).astype(np.float32))
    return path


def _collision_line(command, inputs, out_dir, capsys) -> str:
    argv = [command, *map(str, inputs), *PER_STEM[command]]
    assert main([*argv, "--output-dir", str(out_dir)]) == 2
    line = one_error_line(capsys.readouterr().err)
    assert not out_dir.exists()
    assert "would overwrite each other's outputs" in line
    assert "tokeye example" not in line
    return line


# ---------------------------------------------------------------------------
# Colliding output stems
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_load")
@pytest.mark.parametrize("command", list(PER_STEM))
class TestCollisions:
    def test_same_stem_in_one_directory(self, command, tmp_path, out_dir, capsys):
        d = tmp_path / "d"
        npy, wav = _touch(d / "shot.npy"), _touch(d / "shot.wav")

        line = _collision_line(command, [d], out_dir, capsys)

        assert f"{npy}, {wav} -> 'shot'" in line

    def test_same_name_in_two_directories(self, command, tmp_path, out_dir, capsys):
        a, b = _touch(tmp_path / "a" / "shot.npy"), _touch(tmp_path / "b" / "shot.npy")

        line = _collision_line(command, [a, b], out_dir, capsys)

        assert str(a) in line and str(b) in line

    def test_same_stem_different_suffix(self, command, tmp_path, out_dir, capsys):
        npy, npz = _touch(tmp_path / "shot.npy"), _touch(tmp_path / "shot.npz")

        line = _collision_line(command, [npy, npz], out_dir, capsys)

        assert str(npy) in line and str(npz) in line

    def test_stems_that_differ_only_in_case(self, command, tmp_path, out_dir, capsys):
        # Two directories: one name per directory, so the files are distinct
        # on case-insensitive file systems too.
        upper = _touch(tmp_path / "a" / "Shot.npy")
        lower = _touch(tmp_path / "b" / "shot.npy")

        line = _collision_line(command, [upper, lower], out_dir, capsys)

        assert f"{upper}, {lower} -> 'Shot'" in line

    def test_many_groups_are_truncated(self, command, tmp_path, out_dir, capsys):
        inputs = []
        for i in range(6):
            inputs += [
                _touch(tmp_path / f"g{i}" / f"s{i}.npy"),
                _touch(tmp_path / f"h{i}" / f"s{i}.npy"),
            ]

        line = _collision_line(command, inputs, out_dir, capsys)

        assert [f"-> 's{i}'" in line for i in range(6)] == [True] * 5 + [False]
        assert "; … and 1 more. " in line


def test_check_unique_stems_message(tmp_path):
    a, b = tmp_path / "a" / "shot.npy", tmp_path / "b" / "shot.npy"
    other = tmp_path / "other.npy"

    batch.check_unique_stems([a, other])  # distinct stems: no error
    with pytest.raises(ValueError) as excinfo:
        batch.check_unique_stems([a, other, b])

    assert str(excinfo.value) == (
        "inputs would overwrite each other's outputs (outputs are named after "
        f"the file stem, ignoring case): {a}, {b} -> 'shot'. Process files "
        "that share a stem in separate runs with different output "
        "directories, or rename them."
    )


def test_join_limited():
    assert batch._join_limited(["a", "b"], sep=", ") == "a, b"
    parts = [str(i) for i in range(7)]
    assert batch._join_limited(parts, sep="; ") == "0; 1; 2; 3; 4; … and 2 more"
    assert batch._join_limited(parts, sep=", ", limit=7) == ", ".join(parts)


def test_process_files_rejects_colliding_stems(tmp_path):
    a = _spectrogram(tmp_path / "a" / "shot.npy")
    b = _spectrogram(tmp_path / "b" / "shot.npy")
    out = tmp_path / "out"
    out.mkdir()

    with pytest.raises(ValueError, match="would overwrite each other's outputs"):
        batch.process_files([a, b], nn.Conv2d(1, 2, 1), None, out)

    assert list(out.iterdir()) == []


# ---------------------------------------------------------------------------
# Inputs that do not collide
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "model", "flags"),
    [("elmspec", _TransientStub, []), ("alfvenspec", _StubRCNN, ["--no-masks"])],
    ids=["elmspec", "alfvenspec"],
)
def test_no_per_stem_files_no_collision(
    command, model, flags, tmp_path, out_dir, monkeypatch
):
    stub = model()
    monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: stub)
    a = _spectrogram(tmp_path / "a" / "shot.npy")
    b = _spectrogram(tmp_path / "b" / "shot.npy")

    assert main([command, str(a), str(b), *flags, "--output-dir", str(out_dir)]) == 0


def test_a_file_reached_twice_is_one_input(tmp_path, monkeypatch):
    d = tmp_path / "d"
    x = _touch(d / "x.npy")
    monkeypatch.chdir(d)

    assert batch.collect_inputs([str(d), str(x)]) == [x]
    assert batch.collect_inputs(["x.npy", str(x)]) == [Path("x.npy")]


def test_the_same_file_twice_runs(tmp_path, out_dir, monkeypatch, capsys):
    model = nn.Conv2d(1, 2, kernel_size=1)
    monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: model)
    x = _spectrogram(tmp_path / "x.npy")
    monkeypatch.chdir(tmp_path)

    argv = ["run", "x.npy", str(x), "--no-png", "--output-dir", str(out_dir)]

    assert main(argv) == 0
    assert (out_dir / "x_mask.npy").exists()


def test_a_symlink_with_another_name_is_a_separate_input(tmp_path):
    d = tmp_path / "d"
    shot = _touch(d / "shot.npy")
    try:
        (d / "latest.npy").symlink_to(shot)
    except OSError as exc:
        pytest.skip(f"cannot create a symlink here: {exc}")

    assert batch.collect_inputs([str(d)]) == [d / "latest.npy", shot]


# ---------------------------------------------------------------------------
# run_batch's up-front checks
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_load")
def test_run_batch_rejects_colliding_stems_before_loading(tmp_path):
    a, b = _touch(tmp_path / "a" / "shot.npy"), _touch(tmp_path / "b" / "shot.npy")
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="would overwrite each other's outputs"):
        batch.run_batch([str(a), str(b)], out_dir=out)

    assert not out.exists()


@pytest.mark.usefixtures("no_load")
@pytest.mark.parametrize(
    ("fs", "error"),
    [("abc", TypeError), (-1.0, ValueError), (float("inf"), ValueError)],
    ids=["text", "negative", "inf"],
)
def test_run_batch_checks_fs_before_loading(tmp_path, fs, error):
    x = _touch(tmp_path / "x.npy")
    out = tmp_path / "out"

    with pytest.raises(error, match="fs must be"):
        batch.run_batch([str(x)], out_dir=out, fs=fs)

    assert not out.exists()


# ---------------------------------------------------------------------------
# Globs match supported files only
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_dir(tmp_path):
    d = tmp_path / "mixed"
    for name in ("a_sr1000.npy", "README.md", "notes.csv"):
        _touch(d / name)
    (d / "sub").mkdir()
    return d


def test_a_glob_keeps_supported_files_only(mixed_dir):
    assert batch.collect_inputs([str(mixed_dir / "*")]) == [
        mixed_dir / "a_sr1000.npy",
        mixed_dir / "notes.csv",
    ]
    assert batch.collect_inputs([str(mixed_dir / "*.csv")]) == [mixed_dir / "notes.csv"]


def test_an_explicit_unsupported_file_is_kept(mixed_dir):
    readme = mixed_dir / "README.md"

    assert batch.collect_inputs([str(readme)]) == [readme]
    with pytest.raises(ValueError, match="unsupported input format '.md'"):
        load_signal(readme)


@pytest.mark.usefixtures("no_load")
def test_a_glob_of_unsupported_files_only_exits_2(tmp_path, out_dir, capsys):
    d = tmp_path / "docs"
    _touch(d / "README.md")
    pattern = str(d / "*")

    assert main(["run", pattern, "--output-dir", str(out_dir)]) == 2
    line = one_error_line(capsys.readouterr().err)

    assert not out_dir.exists()
    assert "1 match skipped with unsupported suffixes: .md" in line
    assert f"supported: {', '.join(SIGNAL_SUFFIXES)})" in line
    assert "tokeye example" in line  # "no inputs" keeps its hint


# ---------------------------------------------------------------------------
# Unreadable or empty inputs
# ---------------------------------------------------------------------------


@pytest.fixture
def notes_txt(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("These are my notes.\nNothing numeric here.\n", encoding="utf-8")
    return path


def test_prose_is_not_a_numeric_table(notes_txt):
    with pytest.raises(ValueError) as excinfo:
        load_signal(notes_txt)

    text = str(excinfo.value)
    assert text.startswith(f"{notes_txt}: ")
    assert "not a numeric table" in text


def _write_empty_csv(path):
    path.write_text("", encoding="utf-8")


def _write_header_only_csv(path):
    path.write_text("time,value\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("name", "write", "detail"),
    [
        ("empty.csv", _write_empty_csv, "empty array"),
        ("header.csv", _write_header_only_csv, "empty array"),
        ("empty.npy", lambda p: np.save(p, np.array([])), "empty array"),
        (
            "scalar.npy",
            lambda p: np.save(p, np.float64(3.0)),
            "a single value, not a signal",
        ),
        ("text.npy", lambda p: np.save(p, np.array(["a", "b"])), "dtype <U1"),
    ],
    ids=["empty-csv", "header-only-csv", "empty-npy", "0d-npy", "string-npy"],
)
def test_no_numeric_data(tmp_path, name, write, detail):
    path = tmp_path / name
    write(path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as excinfo:
            load_signal(path)

    assert str(excinfo.value) == f"{path}: no numeric data ({detail})"
    assert caught == []


def test_run_on_prose_is_one_failure_line(notes_txt, out_dir, monkeypatch, capsys):
    model = nn.Conv2d(1, 2, kernel_size=1)
    monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: model)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exit_code = main(["run", str(notes_txt), "--output-dir", str(out_dir)])

    assert exit_code == 1
    err = capsys.readouterr().err
    errors = [line for line in err.splitlines() if line.startswith("error: ")]
    assert len(errors) == 1, err
    assert errors[0].startswith(
        f"error: failed to process {notes_txt}: ValueError: {notes_txt}: "
        "not a numeric table ("
    )
    assert "Traceback" not in err
    assert not [w for w in caught if "no data" in str(w.message)]


# ---------------------------------------------------------------------------
# Range-checked numeric flags
# ---------------------------------------------------------------------------

REJECTED = [
    *[
        (command, ["--threshold", value])
        for command in ("run", "elmspec")
        for value in ("1.5", "-0.1", "nan")
    ],
    ("elmspec", ["--activity-min", "2"]),
    ("elmspec", ["--min-gap-cols", "-1"]),
    ("elmspec", ["--min-duration-cols", "0"]),
    ("alfvenspec", ["--score-min", "1.5"]),
    ("alfvenspec", ["--mean", "nan"]),
    ("alfvenspec", ["--mean", "inf"]),
    ("alfvenspec", ["--window-cols", "-5"]),
    ("alfvenspec", ["--window-cols", "10"]),
    ("alfvenspec", ["--window-cols", "31"]),
]


@pytest.mark.usefixtures("no_load")
@pytest.mark.parametrize(
    ("command", "flag"), REJECTED, ids=[f"{c} {' '.join(f)}" for c, f in REJECTED]
)
def test_out_of_range_flags_exit_2(command, flag, tmp_path, out_dir, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([command, "x.npy", *flag, "--output-dir", str(out_dir)])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert f"argument {flag[0]}" in err
    assert not out_dir.exists()


def test_window_cols_message_shows_the_rule(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["alfvenspec", "x.npy", "--window-cols", "10"])

    assert "must be 0 (no windowing) or an integer >= 32, got 10" in (
        capsys.readouterr().err
    )


ACCEPTED = [
    ("run", ["--threshold", "0"], "threshold", 0.0),
    ("run", ["--threshold", "1"], "threshold", 1.0),
    ("elmspec", ["--threshold", "0"], "threshold", 0.0),
    ("elmspec", ["--threshold", "1"], "threshold", 1.0),
    ("alfvenspec", ["--window-cols", "0"], "window_cols", 0),
    ("alfvenspec", ["--window-cols", "32"], "window_cols", 32),
    ("alfvenspec", ["--mean", "-3.5"], "mean", -3.5),
]


@pytest.mark.parametrize(
    ("command", "flag", "dest", "value"),
    ACCEPTED,
    ids=[f"{c} {' '.join(f)}" for c, f, _, _ in ACCEPTED],
)
def test_in_range_flags_are_accepted(command, flag, dest, value):
    args = build_parser().parse_args([command, "x.npy", *flag])
    assert getattr(args, dest) == value


@pytest.mark.parametrize(
    ("module", "name"),
    [
        (_options, "unit_float"),
        (_options, "finite_float"),
        (_options, "nonnegative_int"),
        (_options, "positive_int"),
        (alfvenspec_cli, "window_cols"),
    ],
    ids=[
        "unit_float",
        "finite_float",
        "nonnegative_int",
        "positive_int",
        "window_cols",
    ],
)
def test_validators_reject_non_numbers(module, name):
    with pytest.raises(argparse.ArgumentTypeError, match="abc"):
        getattr(module, name)("abc")


def test_window_cols_floor_matches_the_library():
    from tokeye.alfvenspec import inference

    assert alfvenspec_cli.MIN_WINDOW_COLS == inference._MIN_WINDOW_COLS


class _MustNotRun(nn.Module):
    def forward(self, images):
        pytest.fail("model called")


@pytest.mark.parametrize("window_cols", [1, 10, 31])
def test_detect_windowed_rejects_narrow_windows(window_cols):
    from tokeye.alfvenspec import detect_windowed

    with pytest.raises(ValueError, match=r"window_cols must be 0 \(no windowing\)"):
        detect_windowed(np.zeros((64, 100)), _MustNotRun(), window_cols=window_cols)
