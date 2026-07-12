"""Tests for ``tokeye princeton-batch`` (sbatch rendering + local run)."""

from __future__ import annotations

import numpy as np
import pytest


def _sbatch_kwargs(**over):
    kw = {
        "outdir": "/scratch/gpfs/user/tokeye/run1",
        "shots": [190000, 190001],
        "probe": "mirnov/07",
        "tlim": None,
        "model": "big_tf_unet",
        "threshold": 0.5,
        "partition": "gpu",
        "gres": "gpu:a100:1",
        "time_limit": "0-02:00:00",
    }
    kw.update(over)
    return kw


def test_sbatch_script_contents(monkeypatch):
    from tokeye.cli.princeton_batch import build_sbatch_script

    monkeypatch.delenv("TOKEYE_MODULE_DIR", raising=False)
    s = build_sbatch_script(**_sbatch_kwargs(tlim=(0.0, 500.0)))

    assert s.startswith("#!/bin/bash")
    assert "#SBATCH --partition=gpu" in s
    assert "#SBATCH --gres=gpu:a100:1" in s
    assert "#SBATCH --time=0-02:00:00" in s
    # Compute nodes are offline: the job must not try to reach Hugging Face.
    assert "export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}" in s
    assert (
        "module use /projects/EKOLEMEN/Modules/modulefiles-shared "
        "&& module load tokeye" in s
    )
    # The job body re-invokes the CLI in local mode.
    assert "tokeye princeton-batch --local --shots 190000,190001" in s
    assert "--probe mirnov/07" in s
    assert "--tlim 0.0 500.0" in s


def test_sbatch_script_no_gres_line_when_empty(monkeypatch):
    from tokeye.cli.princeton_batch import build_sbatch_script

    s = build_sbatch_script(**_sbatch_kwargs(gres=""))
    assert "--gres" not in s


def test_module_dir_env_override(monkeypatch):
    from tokeye.cli.princeton_batch import build_sbatch_script

    monkeypatch.setenv("TOKEYE_MODULE_DIR", "/custom/modulefiles")
    s = build_sbatch_script(**_sbatch_kwargs())
    assert "module use /custom/modulefiles && module load tokeye" in s


def test_slurm_defaults_honor_env(monkeypatch):
    from tokeye.cli import princeton_batch as pb

    monkeypatch.setenv("TOKEYE_SLURM_PARTITION", "gpu-test")
    monkeypatch.setenv("TOKEYE_SLURM_GRES", "gpu:a100:2")
    monkeypatch.setenv("TOKEYE_SLURM_TIME", "0-00:30:00")
    assert pb._default_partition() == "gpu-test"
    assert pb._default_gres() == "gpu:a100:2"
    assert pb._default_time() == "0-00:30:00"

    monkeypatch.delenv("TOKEYE_SLURM_PARTITION", raising=False)
    monkeypatch.delenv("TOKEYE_SLURM_GRES", raising=False)
    monkeypatch.delenv("TOKEYE_SLURM_TIME", raising=False)
    assert pb._default_partition() == "gpu"
    assert pb._default_gres() == "gpu:a100:1"
    assert pb._default_time() == "0-02:00:00"


def test_princeton_batch_is_registered():
    from tokeye.cli import build_parser

    parser = build_parser()
    subactions = [
        a for a in parser._actions if getattr(a, "choices", None) and "run" in a.choices
    ]
    assert subactions, "no subparser action found"
    assert "princeton-batch" in subactions[0].choices


def test_dry_run_prints_script_without_submitting(tmp_path, capsys, monkeypatch):
    import subprocess as sp

    from tokeye.cli import main

    def _no_sbatch(*a, **k):
        raise AssertionError("--dry-run must not invoke sbatch")

    monkeypatch.setattr(sp, "run", _no_sbatch)
    outdir = tmp_path / "run"
    rc = main([
        "princeton-batch", "--shots", "190000-190002",
        "--outdir", str(outdir), "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "#SBATCH --partition=" in out
    assert "--shots 190000,190001,190002" in out
    assert "--probe mirnov/00" in out  # diag default resolved
    assert not (outdir / "submit.sh").exists()


def test_submit_writes_script_and_calls_sbatch(tmp_path, monkeypatch, capsys):
    import tokeye.cli.princeton_batch as pb
    from tokeye.cli import main

    calls = {}

    class _Result:
        returncode = 0
        stdout = "Submitted batch job 123456\n"
        stderr = ""

    def _fake_run(cmd, **kw):
        calls["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(pb.subprocess, "run", _fake_run)
    outdir = tmp_path / "run"
    rc = main(["princeton-batch", "--shots", "190000", "--outdir", str(outdir)])
    assert rc == 0
    script = outdir / "submit.sh"
    assert script.is_file()
    assert calls["cmd"][0] == "sbatch" and calls["cmd"][1] == str(script)
    assert "Submitted batch job 123456" in capsys.readouterr().out


def test_submit_without_sbatch_errors_cleanly(tmp_path, monkeypatch, capsys):
    import tokeye.cli.princeton_batch as pb
    from tokeye.cli import main

    def _no_binary(cmd, **kw):
        raise FileNotFoundError("sbatch")

    monkeypatch.setattr(pb.subprocess, "run", _no_binary)
    rc = main(["princeton-batch", "--shots", "1", "--outdir", str(tmp_path / "x")])
    assert rc == 2
    assert "sbatch not found" in capsys.readouterr().err


def test_bad_shots_or_diag_error(tmp_path, capsys):
    from tokeye.cli import main

    assert main(["princeton-batch", "--shots", "nope", "--outdir", str(tmp_path)]) == 2
    assert "bad --shots" in capsys.readouterr().err

    rc = main([
        "princeton-batch", "--shots", "1", "--outdir", str(tmp_path),
        "--diag", "bogus", "--dry-run",
    ])
    assert rc == 2
    assert "unknown --diag" in capsys.readouterr().err


def _write_shot(root, shot, n=20_000, n_ch=2, fs=100_000.0):
    h5py = pytest.importorskip("h5py")
    t = np.arange(n, dtype=np.float64) / fs
    y = np.random.default_rng(shot).standard_normal((n_ch, n))
    with h5py.File(root / f"{shot}_processed.h5", "w") as f:
        g = f.create_group("mirnov")
        g.create_dataset("xdata", data=t.astype(np.float32))
        g.create_dataset("ydata", data=y.astype(np.float32))


def test_local_run_extracts_slugged_inputs_and_batches_once(tmp_path, monkeypatch):
    """--local: per-shot channel extraction (slugged filenames), one run_batch
    call, failures counted per shot."""
    import tokeye.batch as batch
    from tokeye.cli import main

    archive = tmp_path / "archive"
    archive.mkdir()
    monkeypatch.setenv("TOKEYE_FOUNDATION_DIR", str(archive))
    _write_shot(archive, 190000)
    _write_shot(archive, 190001)
    # 190002 intentionally missing -> one failure

    seen = {}

    def _fake_run_batch(inputs, **kw):
        seen["inputs"] = list(inputs)
        seen["kw"] = kw
        return 0

    monkeypatch.setattr(batch, "run_batch", _fake_run_batch)

    outdir = tmp_path / "out"
    rc = main([
        "princeton-batch", "--local",
        "--shots", "190000-190002",
        "--outdir", str(outdir),
        "--probe", "mirnov/01",
    ])

    assert rc == 1  # exactly the missing shot
    names = [p.rsplit("/", 1)[-1] for p in seen["inputs"]]
    assert names == ["190000_mirnov-01.npy", "190001_mirnov-01.npy"]
    assert all("/" not in n.replace(".npy", "").split("_", 1)[1] for n in names)
    assert seen["kw"]["model"] == "big_tf_unet"
    assert seen["kw"]["threshold"] == 0.5
    for name in names:
        arr = np.load(outdir / "inputs" / name)
        assert arr.dtype == np.float32 and arr.size == 20_000


def test_local_run_batch_failures_add_to_exit_code(tmp_path, monkeypatch):
    import tokeye.batch as batch
    from tokeye.cli import main

    archive = tmp_path / "archive"
    archive.mkdir()
    monkeypatch.setenv("TOKEYE_FOUNDATION_DIR", str(archive))
    _write_shot(archive, 190000)

    monkeypatch.setattr(batch, "run_batch", lambda inputs, **kw: 1)
    rc = main([
        "princeton-batch", "--local", "--shots", "190000",
        "--outdir", str(tmp_path / "out"),
    ])
    assert rc == 1
