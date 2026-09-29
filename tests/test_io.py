from __future__ import annotations

import sys

import numpy as np
import pytest
from scipy.io import savemat, wavfile

from tokeye.io import (
    DIRECTORY_SUFFIXES,
    SIGNAL_SUFFIXES,
    fs_from_name,
    load_signal,
)

SIG = np.sin(np.linspace(0, 20, 500))


class TestNumpy:
    def test_npy(self, tmp_path):
        np.save(tmp_path / "x.npy", SIG)
        data, fs = load_signal(tmp_path / "x.npy")
        np.testing.assert_array_equal(data, SIG)
        assert fs is None

    def test_npy_fs_from_filename(self, tmp_path):
        np.save(tmp_path / "shot_sr500000.npy", SIG)
        _, fs = load_signal(tmp_path / "shot_sr500000.npy")
        assert fs == 500_000.0

    def test_row_and_column_vectors_become_1d(self, tmp_path):
        np.save(tmp_path / "row.npy", SIG[np.newaxis])
        np.save(tmp_path / "col.npy", SIG[:, np.newaxis])
        assert load_signal(tmp_path / "row.npy")[0].shape == (500,)
        assert load_signal(tmp_path / "col.npy")[0].shape == (500,)

    def test_2d_spectrogram_is_kept(self, tmp_path):
        np.save(tmp_path / "spec.npy", np.zeros((64, 32)))
        assert load_signal(tmp_path / "spec.npy")[0].shape == (64, 32)

    def test_npz_named_signal_and_fs(self, tmp_path):
        np.savez(tmp_path / "x.npz", signal=SIG, fs=np.array(2000.0), other=np.ones(3))
        data, fs = load_signal(tmp_path / "x.npz")
        np.testing.assert_array_equal(data, SIG)
        assert fs == 2000.0

    def test_npz_single_unnamed_array(self, tmp_path):
        np.savez(tmp_path / "x.npz", SIG)
        np.testing.assert_array_equal(load_signal(tmp_path / "x.npz")[0], SIG)

    def test_npz_ambiguous_arrays_raise(self, tmp_path):
        np.savez(tmp_path / "x.npz", a=SIG, b=SIG)
        with pytest.raises(ValueError, match=r"cannot tell.*\['a', 'b'\]"):
            load_signal(tmp_path / "x.npz")


class TestAudio:
    def test_int16_wav_is_scaled_and_keeps_fs(self, tmp_path):
        pcm = (SIG * 32767).astype(np.int16)
        wavfile.write(tmp_path / "a.wav", 44_100, pcm)
        data, fs = load_signal(tmp_path / "a.wav")
        assert fs == 44_100.0
        np.testing.assert_allclose(data, pcm / 32768.0)

    def test_uint8_wav_is_centred(self, tmp_path):
        wavfile.write(tmp_path / "a.wav", 8000, np.full(100, 128, dtype=np.uint8))
        np.testing.assert_array_equal(load_signal(tmp_path / "a.wav")[0], 0.0)

    def test_stereo_wav_is_averaged_with_a_warning(self, tmp_path):
        stereo = np.stack([SIG, -SIG], axis=1).astype(np.float32)
        wavfile.write(tmp_path / "s.wav", 8000, stereo)
        with pytest.warns(UserWarning, match="averaging 2 audio channels"):
            data, _ = load_signal(tmp_path / "s.wav")
        np.testing.assert_allclose(data, 0.0, atol=1e-7)

    def test_flac(self, tmp_path):
        soundfile = pytest.importorskip("soundfile")
        soundfile.write(tmp_path / "a.flac", SIG * 0.5, 16_000)
        data, fs = load_signal(tmp_path / "a.flac")
        assert fs == 16_000.0
        np.testing.assert_allclose(data, SIG * 0.5, atol=1e-4)

    def test_flac_without_soundfile_hints(self, tmp_path, monkeypatch):
        (tmp_path / "a.flac").write_bytes(b"fLaC")
        monkeypatch.setitem(sys.modules, "soundfile", None)
        with pytest.raises(ImportError, match="pip install soundfile"):
            load_signal(tmp_path / "a.flac")


class TestText:
    def test_single_column_csv(self, tmp_path):
        np.savetxt(tmp_path / "x.csv", SIG, delimiter=",")
        data, fs = load_signal(tmp_path / "x.csv")
        np.testing.assert_allclose(data, SIG)
        assert fs is None

    def test_time_value_csv_with_header_gives_fs(self, tmp_path):
        t = np.arange(SIG.size) / 1000.0
        np.savetxt(
            tmp_path / "x.csv",
            np.stack([t, SIG], axis=1),
            delimiter=",",
            header="t,x",
            comments="",
        )
        data, fs = load_signal(tmp_path / "x.csv")
        np.testing.assert_allclose(data, SIG)
        assert fs == pytest.approx(1000.0)

    def test_whitespace_txt(self, tmp_path):
        np.savetxt(tmp_path / "x.txt", SIG)
        np.testing.assert_allclose(load_signal(tmp_path / "x.txt")[0], SIG)


class TestMatlabAndHdf5:
    def test_mat_vector_and_fs(self, tmp_path):
        savemat(tmp_path / "x.mat", {"x": SIG[np.newaxis], "fs": 250_000.0})
        data, fs = load_signal(tmp_path / "x.mat")
        np.testing.assert_allclose(data, SIG)
        assert fs == 250_000.0

    def test_h5_dataset_attr_fs(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        with h5py.File(tmp_path / "x.h5", "w") as fh:
            dset = fh.create_dataset("group/signal", data=SIG)
            dset.attrs["fs"] = 1e6
            fh.create_dataset("time", data=np.arange(SIG.size))
        data, fs = load_signal(tmp_path / "x.h5")
        np.testing.assert_allclose(data, SIG)
        assert fs == 1e6

    def test_h5_root_fs_dataset(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        with h5py.File(tmp_path / "x.hdf5", "w") as fh:
            fh.create_dataset("trace", data=SIG)
            fh.create_dataset("sample_rate", data=5000.0)
        data, fs = load_signal(tmp_path / "x.hdf5")
        assert data.shape == SIG.shape
        assert fs == 5000.0

    def test_h5_ambiguous_raises(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        with h5py.File(tmp_path / "x.h5", "w") as fh:
            fh.create_dataset("a", data=SIG)
            fh.create_dataset("b", data=SIG)
        with pytest.raises(ValueError, match="cannot tell"):
            load_signal(tmp_path / "x.h5")


class TestErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            load_signal(tmp_path / "nope.npy")

    def test_unsupported_suffix_lists_supported(self, tmp_path):
        (tmp_path / "x.json").write_text("{}")
        with pytest.raises(ValueError, match=r"\.wav"):
            load_signal(tmp_path / "x.json")


@pytest.mark.parametrize(
    ("name", "fs"),
    [
        ("shot_sr200000.npy", 200_000.0),
        ("shot_sr2000.5.npy", 2000.5),
        ("shot_sr0.npy", None),
        ("shot_sr1e5.npy", None),
        ("shot.npy", None),
    ],
)
def test_fs_from_name(name, fs):
    assert fs_from_name(name) == fs


def test_directory_suffixes_skip_text():
    assert ".csv" in SIGNAL_SUFFIXES
    assert ".csv" not in DIRECTORY_SUFFIXES
    assert ".npy" in DIRECTORY_SUFFIXES
