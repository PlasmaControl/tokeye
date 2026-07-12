"""Tests for the offline batch CLI helpers (offline / no MDSplus, no Slurm)."""

from __future__ import annotations

import pytest


def test_parse_shots_range_and_list():
    from tokeye.cli.diiid_batch import parse_shots

    assert parse_shots("150000-150003") == [150000, 150001, 150002, 150003]
    assert parse_shots("10,12,15") == [10, 12, 15]
    # ranges + list combine, dedup preserves order, whitespace ignored
    assert parse_shots("5-7, 6 , 20") == [5, 6, 7, 20]
    assert parse_shots("") == []


def test_parse_shots_rejects_garbage_and_huge_ranges():
    from tokeye.cli.diiid_batch import parse_shots

    with pytest.raises(ValueError):
        parse_shots("abc")
    with pytest.raises(ValueError):
        parse_shots("1-100000000")  # > MAX_SHOTS


def test_build_sbatch_script_gpu():
    from tokeye.cli.diiid_batch import build_sbatch_script

    s = build_sbatch_script(
        outdir="/cscratch/u/tokeye/data/runs/r1",
        shots=[100, 101],
        diag="mag",
        probe="MPI66M067D",
        tlim=(1000.0, 2000.0),
        model="big_tf_unet",
        threshold=0.5,
        do_tokeye=True,
        do_modespec=True,
        do_gate=True,
        n_range=(-5, 5),
        f_min=5.0,
        f_max=200.0,
        partition="gpus",
        gres="gpu:v100:1",
        time_limit="0-02:00:00",
    )
    assert "#SBATCH --partition=gpus" in s
    assert "#SBATCH --gres=gpu:v100:1" in s
    assert "module use /cscratch/share/tokeye/modulefiles && module load tokeye" in s
    assert "tokeye diiid-batch --shots 100,101" in s
    assert "--outdir /cscratch/u/tokeye/data/runs/r1" in s
    assert "--tlim 1000.0 2000.0" in s
    assert "--tokeye" in s and "--modespec" in s and "--gate" in s
    assert "--decimation 1" in s  # default off; emitted so the job is explicit


def test_build_sbatch_script_threads_decimation():
    from tokeye.cli.diiid_batch import build_sbatch_script

    s = build_sbatch_script(
        outdir="/x",
        shots=[1],
        diag="mag",
        probe="MPI66M067D",
        tlim=None,
        model="big_tf_unet",
        threshold=0.5,
        do_tokeye=False,
        do_modespec=True,
        do_gate=False,
        n_range=(-5, 5),
        f_min=5.0,
        f_max=200.0,
        partition="gpus",
        gres="gpu:v100:1",
        time_limit="0-02:00:00",
        decimation=6,
    )
    assert "--decimation 6" in s


def test_build_sbatch_script_emits_gate_source_when_gating():
    from tokeye.cli.diiid_batch import build_sbatch_script

    s = build_sbatch_script(
        outdir="/x",
        shots=[1],
        diag="mag",
        probe="MPI66M067D",
        tlim=None,
        model="big_tf_unet",
        threshold=0.5,
        do_tokeye=False,
        do_modespec=True,
        do_gate=True,
        n_range=(-5, 5),
        f_min=5.0,
        f_max=200.0,
        partition="gpus",
        gres="gpu:v100:1",
        time_limit="0-02:00:00",
        gate_source="reference",
        reference_probe="MPI66M137D",
    )
    assert "--gate-source reference" in s
    assert "--reference-probe MPI66M137D" in s


def test_build_sbatch_script_no_gate_source_without_gating():
    from tokeye.cli.diiid_batch import build_sbatch_script

    s = build_sbatch_script(
        outdir="/x",
        shots=[1],
        diag="mag",
        probe="MPI66M067D",
        tlim=None,
        model="big_tf_unet",
        threshold=0.5,
        do_tokeye=True,
        do_modespec=False,
        do_gate=False,
        n_range=(-5, 5),
        f_min=5.0,
        f_max=200.0,
        partition="gpus",
        gres="gpu:v100:1",
        time_limit="0-02:00:00",
    )
    # No gating -> neither "--gate" nor the "--gate-source" that contains it.
    assert "--gate" not in s


def test_build_sbatch_script_cpu_omits_gres():
    from tokeye.cli.diiid_batch import build_sbatch_script

    s = build_sbatch_script(
        outdir="/x",
        shots=[1],
        diag="mag",
        probe="MPI66M067D",
        tlim=None,
        model="big_tf_unet",
        threshold=0.5,
        do_tokeye=True,
        do_modespec=False,
        do_gate=False,
        n_range=(-5, 5),
        f_min=5.0,
        f_max=200.0,
        partition="medium",
        gres="",
        time_limit="0-04:00:00",
    )
    assert "#SBATCH --partition=medium" in s
    assert "--gres" not in s
    assert "--tlim" not in s
    assert "--modespec" not in s and "--gate" not in s


def _sbatch_kwargs():
    """Minimal valid kwargs for build_sbatch_script (module-dir tests)."""
    return {
        "outdir": "/x",
        "shots": [1],
        "diag": "mag",
        "probe": "MPI66M067D",
        "tlim": None,
        "model": "big_tf_unet",
        "threshold": 0.5,
        "do_tokeye": True,
        "do_modespec": False,
        "do_gate": False,
        "n_range": (-5, 5),
        "f_min": 5.0,
        "f_max": 200.0,
        "partition": "gpus",
        "gres": "gpu:v100:1",
        "time_limit": "0-02:00:00",
    }


def test_build_sbatch_script_module_dir_env_override(monkeypatch):
    from tokeye.cli.diiid_batch import build_sbatch_script

    monkeypatch.setenv("TOKEYE_MODULE_DIR", "/opt/mods")
    s = build_sbatch_script(**_sbatch_kwargs())
    assert "module use /opt/mods && module load tokeye" in s


def test_build_sbatch_script_module_dir_default(monkeypatch):
    from tokeye.cli.diiid_batch import build_sbatch_script

    monkeypatch.delenv("TOKEYE_MODULE_DIR", raising=False)
    s = build_sbatch_script(**_sbatch_kwargs())
    assert "module use /cscratch/share/tokeye/modulefiles && module load tokeye" in s


def test_diiid_batch_is_not_registered_here():
    """The princeton branch does not wire diiid-batch (no MDSplus route on
    stellar); the module stays importable and princeton-batch reuses its shot
    parsing."""
    from tokeye.cli import build_parser

    parser = build_parser()
    subactions = [
        a for a in parser._actions if getattr(a, "choices", None) and "run" in a.choices
    ]
    assert subactions, "no subparser action found"
    assert "diiid-batch" not in subactions[0].choices
    assert "princeton-batch" in subactions[0].choices
