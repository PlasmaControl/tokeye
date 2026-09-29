"""``--tile`` for ``tokeye run``/``elmspec``, and tiling in ``params.json``."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch.nn as nn

from tokeye import batch, inference
from tokeye.cli import _options, build_parser, main

COMMANDS = ["run", "elmspec"]


def _must_not_run(*args, **kwargs):
    # AssertionError, because the CLI handlers catch ValueError and OSError.
    raise AssertionError("called too early")


@pytest.fixture
def stub_model(monkeypatch):
    """A cheap Conv2d(1, 2, 1) stands in for the real model."""
    model = nn.Conv2d(1, 2, kernel_size=1)
    monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: model)
    return model


@pytest.fixture
def small_npy(tmp_path):
    path = tmp_path / "small.npy"
    np.save(path, np.random.default_rng(0).random((64, 32)).astype(np.float32))
    return path


@pytest.fixture
def wide_npy(tmp_path):
    """A 2D input wider than a 512 tile."""
    path = tmp_path / "wide.npy"
    np.save(path, np.random.default_rng(1).random((40, 1300)).astype(np.float32))
    return path


@pytest.fixture
def tiles(monkeypatch):
    """The ``tile`` each ``infer`` call receives."""
    seen = []
    real = inference.infer

    def spy(model, values, **kwargs):
        seen.append(kwargs.get("tile", "not passed"))
        return real(model, values, **kwargs)

    monkeypatch.setattr("tokeye.batch.infer", spy)  # tokeye run
    monkeypatch.setattr("tokeye.inference.infer", spy)  # tokeye elmspec
    return seen


def _params(out_dir, stem):
    return json.loads((out_dir / f"{stem}_params.json").read_text("utf-8"))


# ---------------------------------------------------------------------------
# Parsing and early validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("auto", "auto"),
        ("AUTO", "auto"),
        ("none", None),
        ("None", None),
        ("1024", 1024),
    ],
)
def test_tile_flag_parses(command, value, expected):
    args = build_parser().parse_args([command, "x.npy", "--tile", value])
    assert args.tile == expected


@pytest.mark.parametrize("command", COMMANDS)
def test_tile_defaults_to_auto(command):
    assert build_parser().parse_args([command, "x.npy"]).tile == "auto"


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize("value", ["big", "1.5", "1_024", ""])
def test_a_non_integer_tile_is_an_argparse_error(command, value, capsys):
    with pytest.raises(SystemExit) as info:
        build_parser().parse_args([command, "x.npy", "--tile", value])
    assert info.value.code == 2
    assert "argument --tile" in capsys.readouterr().err


@pytest.mark.parametrize("command", COMMANDS)
def test_a_small_tile_exits_2_before_the_model_loads(
    command, small_npy, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr("tokeye.hub.load_model", _must_not_run)
    out_dir = tmp_path / "out"

    code = main(
        [command, str(small_npy), "--tile", "100", "--output-dir", str(out_dir)]
    )

    assert code == 2
    assert capsys.readouterr().err.splitlines() == [
        "error: tile must be >= 512, got 100"
    ]
    assert not out_dir.exists()


@pytest.mark.parametrize("command", COMMANDS)
def test_the_help_states_the_real_floor(command, capsys):
    assert f"an int >= {inference.MIN_TILE};" in _options.TILE_HELP
    with pytest.raises(SystemExit):
        main([command, "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "--tile auto|none|N" in out
    assert " ".join(_options.TILE_HELP.split()) in out


def test_run_batch_checks_tile_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr("tokeye.hub.load_model", _must_not_run)
    with pytest.raises(ValueError, match="tile must be >= 512, got 100"):
        batch.run_batch([str(tmp_path)], out_dir=tmp_path, tile=100)
    with pytest.raises(TypeError):
        batch.run_batch([str(tmp_path)], out_dir=tmp_path, tile=True)


# ---------------------------------------------------------------------------
# Pass-through to infer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tile", ["auto", None, 1024])
def test_run_batch_passes_tile_to_infer(stub_model, small_npy, tmp_path, tiles, tile):
    assert batch.run_batch([str(small_npy)], out_dir=tmp_path / "out", tile=tile) == 0
    assert tiles == [tile]


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize(("flag", "tile"), [([], "auto"), (["--tile", "none"], None)])
def test_the_cli_passes_tile_to_infer(
    command, flag, tile, stub_model, small_npy, tmp_path, tiles
):
    argv = [command, str(small_npy), "--output-dir", str(tmp_path / "out"), *flag]
    assert main(argv) == 0
    assert tiles == [tile]


# ---------------------------------------------------------------------------
# params.json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tile", ["auto", None])
def test_params_record_an_untiled_run(stub_model, small_npy, tmp_path, tile):
    batch.run_batch([str(small_npy)], out_dir=tmp_path / "out", tile=tile)
    params = _params(tmp_path / "out", "small")
    assert params["tile"] == tile
    assert params["tile_shape"] is None


@pytest.mark.parametrize("tile", [512, np.int64(512)])
def test_params_record_the_tile_plan(stub_model, wide_npy, tmp_path, tile):
    batch.run_batch([str(wide_npy)], out_dir=tmp_path / "out", tile=tile)
    params = _params(tmp_path / "out", "wide")
    assert params["tile"] == 512
    assert params["tile_shape"] == [40, 512]
    assert inference._plan_tiles((40, 1300), 512) == (40, 512)


def test_a_tile_covering_the_input_records_no_plan(stub_model, wide_npy, tmp_path):
    batch.run_batch([str(wide_npy)], out_dir=tmp_path / "out", tile=2048)
    params = _params(tmp_path / "out", "wide")
    assert params["tile"] == 2048
    assert params["tile_shape"] is None


def test_a_numpy_threshold_is_recorded_as_a_float(stub_model, small_npy, tmp_path):
    batch.process_file(
        small_npy, stub_model, None, tmp_path, threshold=np.float32(0.25)
    )
    threshold = _params(tmp_path, "small")["threshold"]
    assert isinstance(threshold, float)
    assert threshold == 0.25


def test_process_file_rejects_a_bad_format_first(tmp_path, monkeypatch):
    monkeypatch.setattr("tokeye.batch.load_spectrogram", _must_not_run)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="fmt must be one of"):
        batch.process_file(tmp_path / "x.npy", nn.Identity(), None, out_dir, fmt="csv")
    assert list(out_dir.iterdir()) == []


def test_params_are_absent_unless_the_outputs_are_complete(
    stub_model, small_npy, tmp_path, monkeypatch
):
    batch.process_file(small_npy, stub_model, None, tmp_path)
    params_path = tmp_path / "small_params.json"
    assert params_path.exists()

    def broken_preview(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("tokeye.batch.save_preview", broken_preview)
    with pytest.raises(RuntimeError, match="disk full"):
        batch.process_file(small_npy, stub_model, None, tmp_path)

    assert (tmp_path / "small_mask.npy").exists()
    assert not params_path.exists()
    assert not list(tmp_path.glob(".*"))  # no temporary file left behind


def test_a_failed_params_write_leaves_no_temporary_file(
    stub_model, small_npy, tmp_path, monkeypatch
):
    def broken_replace(self, target):
        raise OSError("rename failed")

    monkeypatch.setattr("pathlib.Path.replace", broken_replace)
    with pytest.raises(OSError, match="rename failed"):
        batch.process_file(small_npy, stub_model, None, tmp_path, save_png=False)
    assert not (tmp_path / "small_params.json").exists()
    assert not list(tmp_path.glob(".*"))
