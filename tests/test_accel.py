"""tokeye.accel — CPU thread sizing, spectrogram axes, OpenVINO/torch runners."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tokeye import accel
from tokeye.inference import model_infer
from tokeye.models.big_tf_unet.config_big_tf_unet import BigTFUNetConfig
from tokeye.models.big_tf_unet.model_big_tf_unet import BigTFUNetModel


def test_threads_env_override(monkeypatch):
    monkeypatch.setenv("TOKEYE_THREADS", "3")
    assert accel.available_cpus() == 3


def test_threads_follow_slurm_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert accel.available_cpus() <= 2


def test_configure_threads_sets_torch(monkeypatch):
    before = torch.get_num_threads()
    try:
        assert accel.configure_threads(2) == 2
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)


def test_spectrogram_matches_compute_stft_and_axes():
    from tokeye.transforms import compute_stft

    fs = 200e3
    x = np.random.default_rng(1).standard_normal(20_000)
    spec, t_ms, f_khz = accel.spectrogram(x, fs=fs, t0_ms=1000.0)
    np.testing.assert_allclose(spec, compute_stft(x[None, :]).astype(np.float32))
    assert spec.shape == (f_khz.size, t_ms.size)
    assert f_khz[0] == pytest.approx(fs / 1024 / 1e3)  # DC bin clipped
    assert f_khz[-1] == pytest.approx(fs / 2 / 1e3)
    assert np.diff(t_ms) == pytest.approx(128 / fs * 1e3)
    # frames are centred on multiples of hop from the first sample (scipy also adds
    # half-window frames before it), so t0 itself is one of the frame centres
    assert t_ms[0] <= 1000.0
    assert np.isclose(t_ms, 1000.0).any()


def test_runner_is_accepted_by_model_infer():
    runner = accel.TorchRunner(nn.Conv2d(1, 2, 1), "stub")
    spec = np.random.default_rng(0).random((16, 40))
    out = model_infer(spec, runner)
    assert out.shape == (2, 16, 40)
    assert ((out >= 0) & (out <= 1)).all()


def test_runner_matches_plain_model_infer():
    torch.manual_seed(0)
    model = nn.Conv2d(1, 2, 3, padding=1).eval()
    spec = np.random.default_rng(0).random((32, 64))
    expected = model_infer(spec, lambda_wrap(model))
    got = accel.TorchRunner(model, "stub").infer(spec)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def lambda_wrap(model):
    """A plain nn.Module that returns a 1-tuple, like the U-Net."""

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, x):
            return (self.m(x),)

    return Wrap().eval()


def test_backend_validation(monkeypatch):
    monkeypatch.setenv("TOKEYE_BACKEND", "tpu")
    with pytest.raises(ValueError, match="unknown backend"):
        accel.resolve_backend()


def test_openvino_matches_torch_and_caches_ir(tmp_path, monkeypatch):
    pytest.importorskip("openvino")
    torch.manual_seed(0)
    small = BigTFUNetModel(BigTFUNetConfig(num_layers=3, first_layer_size=4)).eval()
    monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: small)
    monkeypatch.setenv("TOKEYE_OV_DIR", str(tmp_path / "ov"))

    ov_runner = accel.load_runner("big_tf_unet", backend="openvino")
    assert ov_runner.backend == "openvino"
    assert list((tmp_path / "ov").glob("*.xml"))  # IR persisted for the next load

    spec = np.random.default_rng(0).random((64, 200)).astype(np.float32)  # non-square
    ref = accel.TorchRunner(small, "small").infer(spec)
    got = ov_runner.infer(spec)
    assert got.shape == ref.shape == (2, 64, 200)
    np.testing.assert_allclose(got, ref, atol=1e-4)
    # a second, different shape reuses the same dynamic-shape compiled model
    assert ov_runner.infer(spec[:, :97]).shape == (2, 64, 97)


def test_ir_cache_is_keyed_on_the_converted_weights(tmp_path, monkeypatch):
    """Regression: a cached IR must never stand in for different weights.

    The first cache keyed IRs by the weights *file*, so a stubbed model converted
    under the real file's name poisoned every later load.
    """
    pytest.importorskip("openvino")
    monkeypatch.setenv("TOKEYE_OV_DIR", str(tmp_path / "ov"))
    spec = np.random.default_rng(1).random((32, 48)).astype(np.float32)
    outs = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        m = BigTFUNetModel(BigTFUNetConfig(num_layers=2, first_layer_size=4)).eval()
        monkeypatch.setattr("tokeye.hub.load_model", lambda source, device, m=m: m)
        accel._RUNNERS.clear()
        got = accel.load_runner("big_tf_unet", backend="openvino").infer(spec)
        np.testing.assert_allclose(got, accel.TorchRunner(m, "m").infer(spec), atol=1e-4)
        outs.append(got)
    assert len(list((tmp_path / "ov").glob("*.xml"))) == 2
    assert not np.allclose(outs[0], outs[1])
    assert not list((tmp_path / "ov").glob(".*tmp*"))  # atomic writes leave no temp files


def test_auto_falls_back_to_torch_when_openvino_fails(monkeypatch):
    stub = nn.Conv2d(1, 2, 1)
    monkeypatch.setattr("tokeye.hub.load_model", lambda source, device: stub)

    def boom(*a, **k):
        raise RuntimeError("no OV today")

    monkeypatch.setattr(accel, "_openvino_runner", boom)
    runner = accel.load_runner("big_tf_unet", backend="auto")
    assert runner.backend == "torch"
    assert runner.model is stub


def test_cache_write_failure_is_not_fatal(tmp_path, capsys):
    from tokeye.modespec.classic import data_utils

    ro = tmp_path / "ro"
    (ro / "123").mkdir(parents=True)
    (ro / "123").chmod(0o555)
    try:
        data_utils.save_cache(123, "SIG", ([1.0], [0.0]), str(ro))
    finally:
        (ro / "123").chmod(0o755)
    assert "not cached" in capsys.readouterr().out


# ── shared signal cache: must never execute code from a file someone else wrote ──
def test_signal_cache_roundtrip_is_npz_without_pickle(tmp_path):
    from tokeye.modespec.classic import data_utils

    data, t = np.arange(5.0), np.linspace(0, 1, 5)
    data_utils.save_cache(1, "SIG", (data, t), str(tmp_path))
    entry = tmp_path / "1" / "1_SIG.npz"
    assert entry.is_file()
    assert not list((tmp_path / "1").glob("*.pkl"))
    got = data_utils.load_cache(1, "SIG", str(tmp_path))
    np.testing.assert_array_equal(got[0], data)
    np.testing.assert_array_equal(got[1], t)


def test_cache_lookup_does_not_create_directories(tmp_path):
    from tokeye.modespec.classic import data_utils

    root = tmp_path / "missing"
    assert data_utils.load_cache(5, "SIG", str(root)) is None
    assert not root.exists()  # a read never needs a writable cache root


def test_default_cache_is_per_user(monkeypatch):

    from tokeye.sources import mds

    monkeypatch.setenv("USER", "someone")
    root = mds._default_cache_root()
    if Path("/cscratch").is_dir():
        assert root == "/cscratch/someone/.tokeye/cache"
    else:
        assert root == str(Path("~/.cache/tokeye/cache").expanduser())


def test_planted_pickle_payload_in_npz_is_never_unpickled(tmp_path, capsys):
    from tokeye.modespec.classic import data_utils

    (tmp_path / "2").mkdir()
    np.savez(tmp_path / "2" / "2_SIG.npz", np.array([object()], dtype=object))
    assert data_utils.load_cache(2, "SIG", str(tmp_path)) is None
    assert "ignoring unreadable cache entry" in capsys.readouterr().out


def test_untrusted_legacy_pickle_is_ignored(tmp_path, capsys):
    import pickle

    from tokeye.modespec.classic import data_utils

    (tmp_path / "3").mkdir()
    pkl = tmp_path / "3" / "3_SIG.pkl"
    pkl.write_bytes(pickle.dumps(([1.0], [0.0])))
    pkl.chmod(0o664)  # group-writable: anyone in the group could have planted it
    assert data_utils.load_cache(3, "SIG", str(tmp_path)) is None
    assert "ignoring untrusted legacy cache" in capsys.readouterr().out
    pkl.chmod(0o644)  # ours and not writable by others: trusted
    assert data_utils.load_cache(3, "SIG", str(tmp_path)) == ([1.0], [0.0])


def test_object_payloads_are_not_cached(tmp_path, capsys):
    from tokeye.modespec.classic import data_utils

    data_utils.save_cache(4, "SIG", {"a": 1}, str(tmp_path))
    assert not list((tmp_path / "4").glob("*"))
    assert "not cached" in capsys.readouterr().out


def test_truncated_or_swept_cache_entry_is_a_refetch_not_a_crash(tmp_path, capsys):
    """/cscratch is swept daily: a cache file can be cut short or emptied."""
    from tokeye.modespec.classic import data_utils

    data_utils.save_cache(6, "SIG", (np.arange(1000.0), np.arange(1000.0)), str(tmp_path))
    entry = tmp_path / "6" / "6_SIG.npz"
    entry.write_bytes(entry.read_bytes()[:100])  # truncated mid-file
    assert data_utils.load_cache(6, "SIG", str(tmp_path)) is None
    entry.write_bytes(b"")  # emptied
    assert data_utils.load_cache(6, "SIG", str(tmp_path)) is None
    assert "ignoring unreadable cache entry" in capsys.readouterr().out
