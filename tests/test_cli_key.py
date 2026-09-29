"""``--key`` on the command line and ``key=`` in the batch API.

Paths are compared as ``str(path)``, never put in a regex.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch.nn as nn
from cli_helpers import _StubRCNN, _TransientStub, one_error_line

from tokeye import batch
from tokeye.cli import main

COMMANDS = ("run", "elmspec", "alfvenspec")


def _no_load(*args, **kwargs):
    pytest.fail("the model was loaded")


@pytest.fixture
def xy_npz(tmp_path):
    """The usual (x, y) layout: a 3,000-sample ramp and a 5,000-sample signal."""
    path = tmp_path / "xy.npz"
    rng = np.random.default_rng(0)
    np.savez(path, x=np.arange(3000) / 1000.0, y=rng.standard_normal(5000))
    return path


@pytest.fixture
def stub_model(monkeypatch):
    model = nn.Conv2d(1, 2, kernel_size=1)
    monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: model)
    return model


def _error_lines(err: str) -> list[str]:
    return [line for line in err.splitlines() if line.startswith("error:")]


@pytest.mark.usefixtures("stub_model")
def test_run_with_key(xy_npz, tmp_path):
    out = tmp_path / "out"

    assert main(["run", str(xy_npz), "--key", "y", "--output-dir", str(out)]) == 0

    params = json.loads((out / "xy_params.json").read_text(encoding="utf-8"))
    assert params["key"] == "y"
    expected = batch.load_spectrogram(xy_npz, key="y").values.shape
    assert params["mask_shape"][1:] == list(expected)


@pytest.mark.usefixtures("stub_model")
def test_run_without_key_fails_that_input(xy_npz, tmp_path, capsys):
    out = tmp_path / "out"

    assert main(["run", str(xy_npz), "--no-png", "--output-dir", str(out)]) == 1

    err = capsys.readouterr().err
    lines = _error_lines(err)
    assert len(lines) == 1
    assert lines[0].startswith(f"error: failed to process {xy_npz}: ")
    assert "cannot tell which array is the signal" in lines[0]
    assert "Traceback" not in err


@pytest.mark.parametrize("command", COMMANDS)
def test_key_on_a_non_container_exits_2_before_loading(
    command, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr("tokeye.hub.load_model", _no_load)
    npy = tmp_path / "a.npy"
    np.save(npy, np.zeros(100))
    out = tmp_path / "out"

    code = main([command, str(npy), "--key", "y", "--output-dir", str(out)])

    assert code == 2
    line = one_error_line(capsys.readouterr().err)
    assert line == (
        "error: key= (--key) applies only to .npz, .mat, .h5 and .hdf5 inputs, "
        f"not: {npy}"
    )
    assert not out.exists()


def test_key_on_a_non_container_lists_at_most_5(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("tokeye.hub.load_model", _no_load)
    paths = []
    for i in range(7):
        paths.append(tmp_path / f"a{i}.npy")
        np.save(paths[-1], np.zeros(100))
    npz = tmp_path / "ok.npz"
    np.savez(npz, y=np.zeros(100))
    out = tmp_path / "out"

    argv = ["run", str(npz), *map(str, paths), "--key", "y"]
    assert main([*argv, "--output-dir", str(out)]) == 2

    line = one_error_line(capsys.readouterr().err)
    assert line.endswith(f"not: {', '.join(map(str, paths[:5]))}, … and 2 more")
    assert str(npz) not in line


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize("key", ["", "   "], ids=["empty", "blank"])
def test_an_empty_key_is_a_usage_error(command, key, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("tokeye.hub.load_model", _no_load)
    out = tmp_path / "out"

    with pytest.raises(SystemExit) as excinfo:
        main([command, "x.npz", "--key", key, "--output-dir", str(out)])

    assert excinfo.value.code == 2
    assert "argument --key" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    ("command", "stub"),
    [("elmspec", _TransientStub), ("alfvenspec", _StubRCNN)],
    ids=["elmspec", "alfvenspec"],
)
def test_suite_commands_pass_the_key(command, stub, xy_npz, tmp_path, monkeypatch):
    model = stub()
    monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: model)
    keys = []
    real = batch.load_spectrogram

    def spy(*args, **kwargs):
        keys.append(kwargs.get("key"))
        return real(*args, **kwargs)

    monkeypatch.setattr("tokeye.batch.load_spectrogram", spy)
    out = tmp_path / "out"

    assert main([command, str(xy_npz), "--key", "y", "--output-dir", str(out)]) == 0
    assert keys == ["y"]


def test_run_batch_rejects_a_key_on_a_non_container(tmp_path, monkeypatch):
    monkeypatch.setattr("tokeye.hub.load_model", _no_load)
    npy = tmp_path / "a.npy"
    np.save(npy, np.zeros(100))
    out = tmp_path / "out"

    with pytest.raises(ValueError, match=r"applies only to \.npz") as excinfo:
        batch.run_batch([str(npy)], out_dir=out, key="y")

    assert str(npy) in str(excinfo.value)
    assert not out.exists()


@pytest.mark.parametrize(
    ("key", "error"),
    [(3, TypeError), ("", ValueError), (" ", ValueError)],
    ids=["int", "empty", "blank"],
)
def test_run_batch_checks_the_key_up_front(key, error, tmp_path, monkeypatch):
    monkeypatch.setattr("tokeye.batch.collect_inputs", _no_load)
    monkeypatch.setattr("tokeye.hub.load_model", _no_load)

    with pytest.raises(error, match="key must be"):
        batch.run_batch([str(tmp_path / "missing.npz")], out_dir=tmp_path, key=key)


@pytest.mark.usefixtures("stub_model")
def test_run_batch_with_key(xy_npz, tmp_path):
    out = tmp_path / "out"

    assert batch.run_batch([str(xy_npz)], out_dir=out, save_png=False, key="y") == 0

    params = json.loads((out / "xy_params.json").read_text(encoding="utf-8"))
    assert params["key"] == "y"


@pytest.mark.usefixtures("stub_model")
def test_params_json_records_no_key_as_null(tmp_path):
    path = tmp_path / "sig.npy"
    np.save(path, np.random.default_rng(0).standard_normal(4000))
    out = tmp_path / "out"

    assert main(["run", str(path), "--no-png", "--output-dir", str(out)]) == 0

    params = json.loads((out / "sig_params.json").read_text(encoding="utf-8"))
    assert params["key"] is None


def test_check_key_inputs():
    batch.check_key_inputs(["a.npy"], None)
    batch.check_key_inputs(["a.NPZ", "b.mat", "c.h5", "d.hdf5"], "y")
    with pytest.raises(ValueError, match=r"not: a\.npy, b\.wav$"):
        batch.check_key_inputs(["a.npy", "ok.npz", "b.wav"], "y")
    with pytest.raises(TypeError, match="key must be a string, got int"):
        batch.check_key_inputs(["a.npz"], 3)
