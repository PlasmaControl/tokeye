"""One-line CLI errors: the shared start-up checks, model loads and ``-v``.

Every usage or configuration error is one ``error:`` line with exit code 2,
never a traceback, and leaves no output directory behind. Failures that must
happen before the model loads are proven with a load spy that calls
``pytest.fail``: its ``Failed`` is a ``BaseException``, which the CLI's
``except Exception`` catch-alls let through.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
import torch
import torch.nn as nn
from cli_helpers import multiline_repository_not_found_error, one_error_line
from huggingface_hub.errors import LocalEntryNotFoundError

from tokeye import SpectrogramConfig, batch, hub
from tokeye.cli import _common, main

COMMANDS = ["run", "elmspec", "alfvenspec"]
DEFAULT_MODEL = {
    "run": "big_tf_unet",
    "elmspec": "big_tf_unet",
    "alfvenspec": "ae_tf_maskrcnn",
}
OFFLINE = "are not in the local cache and Hugging Face cannot be reached"


def _no_load(*args, **kwargs):
    pytest.fail("the model was loaded")


@pytest.fixture
def no_load(monkeypatch):
    """Fail the test if the model is loaded."""
    monkeypatch.setattr("tokeye.hub.load_model", _no_load)


@pytest.fixture
def stub_model(monkeypatch):
    model = nn.Conv2d(1, 2, kernel_size=1)
    monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: model)
    return model


@pytest.fixture
def spectrogram_npy(tmp_path):
    path = tmp_path / "input.npy"
    np.save(path, np.random.default_rng(0).random((64, 32)).astype(np.float32))
    return path


@pytest.fixture
def signal_npy(tmp_path):
    path = tmp_path / "signal.npy"
    np.save(path, np.random.default_rng(0).normal(size=8192).astype(np.float32))
    return path


@pytest.fixture
def out_dir(tmp_path):
    return tmp_path / "out"


def _usage_error(argv, out_dir, capsys) -> str:
    """Run ``argv``; assert exit 2, one error line and no output dir."""
    assert main(argv) == 2
    line = one_error_line(capsys.readouterr().err)
    assert not out_dir.exists()
    return line


# ---------------------------------------------------------------------------
# --device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize("device", ["gpu", "cuda:99"])
def test_a_bad_device_exits_2_before_loading(
    command, device, spectrogram_npy, out_dir, no_load, capsys
):
    argv = [command, str(spectrogram_npy), "--output-dir", str(out_dir)]
    line = _usage_error([*argv, "--device", device], out_dir, capsys)
    assert repr(device) in line


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize(
    ("device", "cuda", "count", "mps"),
    [("cuda", False, 0, False), ("cuda:1", True, 1, False), ("mps", False, 0, False)],
    ids=["cuda-missing", "cuda-index", "mps-missing"],
)
def test_an_unavailable_device_points_to_tokeye_info(
    command,
    device,
    cuda,
    count,
    mps,
    spectrogram_npy,
    out_dir,
    no_load,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(hub, "_mps_available", lambda: mps)
    argv = [command, str(spectrogram_npy), "--output-dir", str(out_dir)]

    line = _usage_error([*argv, "--device", device], out_dir, capsys)

    assert repr(device) in line
    assert line.endswith("; run `tokeye info` to see what this machine has")


@pytest.mark.parametrize("device", ["gpu", "cuda:x", "xpu", "", "CPU", "cuda:"])
def test_resolve_device_rejects_unknown_forms(device):
    with pytest.raises(ValueError, match="cpu, cuda, cuda:N, mps or auto"):
        hub.resolve_device(device)


def test_resolve_device_checks_availability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hub, "_mps_available", lambda: False)
    for device in ("cuda", "cuda:0", "mps"):
        with pytest.raises(ValueError, match="tokeye info"):
            hub.resolve_device(device)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match=r"'cuda:1'.*tokeye info"):
        hub.resolve_device("cuda:1")
    assert hub.resolve_device("cuda:0") == "cuda:0"
    assert hub.resolve_device("cuda") == "cuda"

    monkeypatch.setattr(hub, "_mps_available", lambda: True)
    assert hub.resolve_device("mps") == "mps"


def test_resolve_device_accepts_a_torch_device():
    assert hub.resolve_device(torch.device("cpu")) == "cpu"
    assert hub.resolve_device("cpu") == "cpu"


# ---------------------------------------------------------------------------
# --window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize("window", ["", "foo"], ids=["empty", "unknown"])
def test_a_bad_window_exits_2_before_loading(
    command, window, signal_npy, out_dir, no_load, capsys
):
    argv = [command, str(signal_npy), "--output-dir", str(out_dir)]
    line = _usage_error([*argv, "--window", window], out_dir, capsys)
    if window:
        assert "--window 'foo'" in line
    else:
        assert "window must be a non-empty string" in line


@pytest.mark.parametrize("command", COMMANDS)
def test_a_bad_window_fails_even_when_every_input_is_2d(
    command, spectrogram_npy, out_dir, no_load, capsys
):
    argv = [command, str(spectrogram_npy), "--output-dir", str(out_dir)]
    line = _usage_error([*argv, "--window", "foo"], out_dir, capsys)
    assert "--window 'foo'" in line


# ---------------------------------------------------------------------------
# Model-load errors
# ---------------------------------------------------------------------------


def _check_offline(line, command):
    assert OFFLINE in line
    assert f"run `tokeye download {DEFAULT_MODEL[command]}`" in line


LOAD_ERRORS = [
    pytest.param(
        lambda: OSError("disk on fire"),
        lambda line, command: (
            line == "error: disk on fire"
            and "Hugging Face" not in line
            and "TOKEYE_HF_REPO" not in line
        ),
        id="oserror",
    ),
    pytest.param(
        lambda: PermissionError(13, "Permission denied", "m.pt"),
        lambda line, command: "Permission denied" in line
        and "Hugging Face" not in line,
        id="permission",
    ),
    pytest.param(
        lambda: FileNotFoundError("Model file not found: x.pt"),
        lambda line, command: line == "error: Model file not found: x.pt",
        id="missing-file",
    ),
    pytest.param(
        multiline_repository_not_found_error,
        lambda line, command: (
            "could not download model" in line
            and "Repository Not Found" in line
            and line.endswith("If the repo has moved, set TOKEYE_HF_REPO to override.")
        ),
        id="hub-http",
    ),
    pytest.param(
        lambda: hub.DownloadError("could not download x.pt from a/b: boom"),
        lambda line, command: _check_offline(line, command) is None,
        id="download-error",
    ),
    pytest.param(
        lambda: RuntimeError("boom"),
        lambda line, command: line == "error: RuntimeError: boom",
        id="unexpected",
    ),
    pytest.param(
        lambda: LocalEntryNotFoundError("not in the cache"),
        lambda line, command: _check_offline(line, command) is None,
        id="local-entry",
    ),
]


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize(("make", "check"), LOAD_ERRORS)
def test_model_load_errors_are_one_line(
    command, make, check, spectrogram_npy, out_dir, monkeypatch, capsys
):
    def raising_load(source, device):
        raise make()

    monkeypatch.setattr("tokeye.hub.load_model", raising_load)
    argv = [command, str(spectrogram_npy), "--output-dir", str(out_dir)]

    line = _usage_error(argv, out_dir, capsys)

    assert check(line, command), line


@pytest.mark.parametrize("command", COMMANDS)
def test_ctrl_c_while_loading_returns_130(
    command, spectrogram_npy, out_dir, monkeypatch, capsys
):
    def interrupted(source, device):
        raise KeyboardInterrupt

    monkeypatch.setattr("tokeye.hub.load_model", interrupted)

    assert main([command, str(spectrogram_npy), "--output-dir", str(out_dir)]) == 130
    assert "interrupted" in capsys.readouterr().err
    assert not out_dir.exists()


@pytest.mark.parametrize("command", COMMANDS)
def test_an_unknown_model_name_is_one_line(command, spectrogram_npy, out_dir, capsys):
    argv = [command, str(spectrogram_npy), "--output-dir", str(out_dir)]
    line = _usage_error([*argv, "--model", "not_a_model"], out_dir, capsys)
    assert "Unknown model 'not_a_model'" in line


@pytest.mark.parametrize(
    "raised",
    [lambda: RuntimeError("client closed"), lambda: LocalEntryNotFoundError("offline")],
    ids=["runtime", "local-entry"],
)
@pytest.mark.parametrize("command", COMMANDS)
def test_an_unreachable_hub_without_cached_weights_is_one_line(
    command, raised, spectrogram_npy, out_dir, monkeypatch, capsys
):
    def hf_hub_download(*args, **kwargs):
        raise raised()

    monkeypatch.setattr("tokeye.hub.hf_hub_download", hf_hub_download)
    argv = [command, str(spectrogram_npy), "--output-dir", str(out_dir)]

    _check_offline(_usage_error(argv, out_dir, capsys), command)


@pytest.mark.parametrize(
    "raised",
    [lambda: RuntimeError("client closed"), lambda: LocalEntryNotFoundError("offline")],
    ids=["runtime", "local-entry"],
)
def test_download_without_a_network_is_one_line(raised, monkeypatch, capsys):
    def hf_hub_download(*args, **kwargs):
        raise raised()

    monkeypatch.setattr("tokeye.hub.hf_hub_download", hf_hub_download)

    assert main(["download"]) == 2
    line = one_error_line(capsys.readouterr().err)
    assert line.startswith("error: cannot reach Hugging Face to download 'big_tf_unet'")
    assert "HF_HUB_OFFLINE" in line


def _truncated_checkpoint(path):
    torch.save(nn.Conv2d(1, 2, 3).state_dict(), path)
    size = path.stat().st_size
    with path.open("r+b") as fh:
        fh.truncate(size // 2)


def _text_checkpoint(path):
    path.write_text("hello, this is not a checkpoint\nsecond line\n", "utf-8")


def _wrong_architecture(path):
    torch.save(nn.Conv2d(1, 2, 3).state_dict(), path)


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize(
    ("write", "expected"),
    [
        (_truncated_checkpoint, "not a readable checkpoint"),
        (_text_checkpoint, "not a readable checkpoint"),
        (_wrong_architecture, "does not match any known TokEye architecture"),
    ],
    ids=["truncated", "text", "wrong-architecture"],
)
def test_bad_local_checkpoints_are_one_line(
    command, write, expected, spectrogram_npy, tmp_path, out_dir, capsys
):
    checkpoint = tmp_path / "bad.pt"
    write(checkpoint)
    argv = [command, str(spectrogram_npy), "--output-dir", str(out_dir)]

    line = _usage_error([*argv, "--model", str(checkpoint)], out_dir, capsys)

    assert expected in line
    assert str(checkpoint) in line or "architecture" in expected
    assert len(line) < 500


@pytest.mark.parametrize("command", COMMANDS)
def test_an_output_dir_that_is_a_file_is_one_line(
    command, spectrogram_npy, tmp_path, stub_model, capsys
):
    target = tmp_path / "taken"
    target.write_text("", "utf-8")

    assert main([command, str(spectrogram_npy), "--output-dir", str(target)]) == 2
    line = one_error_line(capsys.readouterr().err)
    assert line.startswith(f"error: cannot create output directory {target}: ")
    assert "Hugging Face" not in line
    assert target.is_file()


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def test_print_hub_error_is_one_line(capsys):
    _common.print_hub_error("big_tf_unet", multiline_repository_not_found_error())

    err = capsys.readouterr().err
    assert len(err.splitlines()) == 1
    assert ". If the repo has moved, set TOKEYE_HF_REPO to override." in err
    assert "'nc1/big_tf_unet'" in err


def test_one_line_collapses_whitespace():
    assert _common.one_line("a\n\tb   c\n") == "a b c"
    assert _common.one_line(KeyError("x")) == "'x'"


def test_require_task_uses_the_right_articles():
    with pytest.raises(ValueError) as info:
        hub.require_task("ae_tf_maskrcnn", "segmentation")
    assert "is an instance model, but this needs a segmentation model" in str(
        info.value
    )
    with pytest.raises(ValueError) as info:
        hub.require_task("big_tf_unet", "instance")
    assert "is a segmentation model, but this needs an instance model" in str(
        info.value
    )


# ---------------------------------------------------------------------------
# Per-input failures and -v
# ---------------------------------------------------------------------------


class _Broken(nn.Module):
    """A model whose forward pass fails with an empty ``AssertionError``."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        raise AssertionError


@pytest.fixture
def broken_model(monkeypatch):
    model = _Broken()
    monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: model)
    return model


@pytest.fixture
def two_inputs(tmp_path):
    paths = [tmp_path / "a.npy", tmp_path / "b.npy"]
    for path in paths:
        np.save(path, np.random.default_rng(0).random((64, 32)).astype(np.float32))
    return paths


def _debug_tracebacks(caplog):
    return [
        record
        for record in caplog.records
        if record.name.split(".")[0] == "tokeye"
        and record.levelno == logging.DEBUG
        and record.exc_info
    ]


@pytest.mark.parametrize("command", COMMANDS)
def test_per_input_failures_name_the_type_on_their_own_line(
    command, two_inputs, out_dir, broken_model, capsys, caplog
):
    argv = [command, *map(str, two_inputs), "--output-dir", str(out_dir)]

    assert main(argv) == 1

    lines = capsys.readouterr().err.splitlines()
    for path in two_inputs:
        prefix = f"error: failed to process {path}: AssertionError: "
        assert any(line.startswith(prefix) for line in lines), lines
    assert not _debug_tracebacks(caplog)


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize("where", ["before", "after"])
def test_verbose_logs_the_traceback(
    command, where, two_inputs, out_dir, broken_model, capsys, caplog
):
    argv = [command, *map(str, two_inputs), "--output-dir", str(out_dir)]
    argv = ["-v", *argv] if where == "before" else [*argv, "-v"]

    assert main(argv) == 1

    assert len(_debug_tracebacks(caplog)) == 2
    err = capsys.readouterr().err
    assert "Traceback" in err
    prefix = f"error: failed to process {two_inputs[0]}: AssertionError: "
    assert any(line.startswith(prefix) for line in err.splitlines())


def test_verbose_is_accepted_by_every_subcommand():
    from tokeye.cli import build_parser

    parser = build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if action.choices and action.dest == "command"
    )
    assert set(subparsers.choices) == {
        "app",
        "run",
        "download",
        "example",
        "info",
        "elmspec",
        "alfvenspec",
    }
    for name, subparser in subparsers.choices.items():
        flags = {
            flag for action in subparser._actions for flag in action.option_strings
        }
        assert {"-v", "--verbose"} <= flags, name
    assert parser.parse_args(["-v", "info"]).verbose is True
    assert parser.parse_args(["info", "-v"]).verbose is True
    assert not hasattr(parser.parse_args(["info"]), "verbose")


def test_verbose_does_not_leak_into_later_calls(
    spectrogram_npy, out_dir, monkeypatch, capsys
):
    def raising_load(source, device):
        raise RuntimeError("boom")

    monkeypatch.setattr("tokeye.hub.load_model", raising_load)
    tokeye_logger = logging.getLogger("tokeye")
    level, handlers = tokeye_logger.level, list(tokeye_logger.handlers)
    argv = ["run", str(spectrogram_npy), "--output-dir", str(out_dir)]

    assert main([*argv, "-v"]) == 2
    assert "Traceback" in capsys.readouterr().err
    assert (tokeye_logger.level, tokeye_logger.handlers) == (level, handlers)

    assert main(argv) == 2
    assert one_error_line(capsys.readouterr().err) == "error: RuntimeError: boom"


def test_an_escaping_exception_is_one_line(monkeypatch, capsys):
    def buggy(args):
        raise RuntimeError("bug")

    monkeypatch.setattr("tokeye.cli.info._handle", buggy)

    assert main(["info"]) == 1
    line = one_error_line(capsys.readouterr().err)
    assert "unexpected RuntimeError: bug" in line
    assert "-v" in line


# ---------------------------------------------------------------------------
# batch.process_files
# ---------------------------------------------------------------------------


@pytest.fixture
def batch_inputs(tmp_path):
    """The inputs of tests/test_batch.py, plus one that fails."""
    signal = tmp_path / "signal.npy"
    np.save(signal, np.random.default_rng(0).normal(size=8192).astype(np.float32))
    spectrogram = tmp_path / "spectrogram.npy"
    np.save(
        spectrogram, np.random.default_rng(1).normal(size=(64, 32)).astype(np.float32)
    )
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((2, 3, 4)))
    return [signal, spectrogram, bad]


def _outputs(directory):
    return sorted(path.name for path in directory.iterdir())


def _params_without_time(directory, stem):
    params = json.loads((directory / f"{stem}_params.json").read_text("utf-8"))
    params.pop("created_utc")
    return params


def test_process_files_matches_run_batch(stub_model, batch_inputs, tmp_path):
    config = SpectrogramConfig(n_fft=256, hop=64)
    via_run_batch, via_process_files = tmp_path / "a", tmp_path / "b"
    via_process_files.mkdir()

    failures_a = batch.run_batch(
        [str(p) for p in batch_inputs], out_dir=via_run_batch, config=config
    )
    failures_b = batch.process_files(
        batch_inputs,
        stub_model,
        config,
        via_process_files,
        channels=("coherent", "transient"),
    )

    assert failures_a == failures_b == 1
    assert _outputs(via_run_batch) == _outputs(via_process_files)
    for stem in ("signal", "spectrogram"):
        np.testing.assert_array_equal(
            np.load(via_run_batch / f"{stem}_mask.npy"),
            np.load(via_process_files / f"{stem}_mask.npy"),
        )
        assert _params_without_time(via_run_batch, stem) == _params_without_time(
            via_process_files, stem
        )


def test_process_files_reports_each_failure(stub_model, batch_inputs, tmp_path):
    bad_too = tmp_path / "bad_too.npy"
    np.save(bad_too, np.zeros((2, 3, 4)))
    seen = []

    failures = batch.process_files(
        [*batch_inputs, bad_too],
        stub_model,
        None,
        tmp_path,
        on_error=lambda path, exc: seen.append((path, type(exc))),
    )

    assert failures == 2
    assert seen == [(batch_inputs[2], ValueError), (bad_too, ValueError)]


def test_process_files_logs_failures_without_on_error(
    stub_model, batch_inputs, tmp_path, caplog
):
    with caplog.at_level(logging.ERROR, logger="tokeye.batch"):
        assert batch.process_files(batch_inputs, stub_model, None, tmp_path) == 1
    assert f"Failed to process {batch_inputs[2]}: ValueError: " in caplog.text


def test_process_files_warns_once_for_a_dict_config(stub_model, batch_inputs, tmp_path):
    with pytest.warns(DeprecationWarning) as record:
        batch.process_files(batch_inputs[:2], stub_model, {"hop": 64}, tmp_path)

    assert len([w for w in record if "SpectrogramConfig" in str(w.message)]) == 1
    assert (tmp_path / "signal_mask.npy").exists()
