"""``Segmentation`` objects and files keep what they were given.

``save()`` writes the result's own axes, and ``load()`` never swaps a
recorded setting for a default without saying so (one ``UserWarning`` per
ignored value, naming the file).
"""

from __future__ import annotations

import inspect
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from tokeye import export
from tokeye.config import DEFAULT_CONFIG, SpectrogramConfig
from tokeye.preprocess import Spectrogram, _axes, prepare
from tokeye.result import Segmentation

H, W = 4, 3


def _spectrogram(fs: float | None, freqs=None, times=None) -> Spectrogram:
    values = np.random.default_rng(0).random((H, W)).astype(np.float32)
    index_freqs, index_times = _axes((H, W), DEFAULT_CONFIG, None)
    return Spectrogram(
        values,
        index_freqs if freqs is None else np.asarray(freqs, dtype=np.float64),
        index_times if times is None else np.asarray(times, dtype=np.float64),
        "stft",
        DEFAULT_CONFIG,
        fs,
    )


def _mask(channels: int = 2) -> np.ndarray:
    return np.random.default_rng(1).random((channels, H, W)).astype(np.float32)


def _record(func, *args):
    """``func(*args)`` and the warnings it raised, all recorded."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args)
    return result, caught


def _load_one_warning(path: Path) -> tuple[Segmentation, str]:
    """Load ``path``, which must raise exactly one UserWarning naming it."""
    seg, caught = _record(Segmentation.load, path)
    assert [w.category for w in caught] == [UserWarning], [
        str(w.message) for w in caught
    ]
    message = str(caught[0].message)
    assert message.startswith(f"{path}: ")
    assert Path(caught[0].filename).name == Path(__file__).name
    return seg, message


def _load_silently(path: Path) -> Segmentation:
    seg, caught = _record(Segmentation.load, path)
    assert not caught, [str(w.message) for w in caught]
    return seg


def _save_bundle(path: Path, *, params, channels: int = 2, axes=None, **edits):
    bundle = export.analysis_bundle(
        spectrogram=np.zeros((H, W)),
        mask=np.zeros((channels, H, W)),
        axes=axes,
        params=params,
        source="analyze",
    )
    bundle.update(edits)
    return export.save_npz(path, bundle)


class TestConstruction:
    def test_spectrogram_must_be_a_spectrogram(self):
        with pytest.raises(TypeError, match="spectrogram must be a tokeye Spectrogram"):
            Segmentation(np.zeros((2, 4, 4)), np.zeros((4, 4)))

    def test_missing_channels_are_missing_attributes(self):
        seg = Segmentation(_mask(1), _spectrogram(None), ("channel_0",))

        assert hasattr(seg, "coherent") is False
        assert getattr(seg, "transient", None) is None
        inspect.getmembers(seg)
        with pytest.raises(KeyError, match="no channel 'coherent'"):
            seg["coherent"]
        with pytest.raises(AttributeError, match=r"^no channel 'coherent'; channels"):
            _ = seg.coherent


class TestSaveAxes:
    def test_hand_built_axes_round_trip(self, tmp_path):
        times = [1.5, 1.502, 1.504]
        freqs = [10e3, 20e3, 30e3, 40e3]
        seg = Segmentation(_mask(), _spectrogram(500e3, freqs, times))

        again = _load_silently(seg.save(tmp_path / "shot"))

        np.testing.assert_allclose(again.times, times, rtol=1e-12)
        np.testing.assert_allclose(again.freqs, freqs, rtol=1e-12)
        assert again.fs == 500e3

    def test_axes_without_fs_warn_that_they_are_not_saved(self, tmp_path):
        seg = Segmentation(_mask(), _spectrogram(None, freqs=[5, 6, 7, 8]))

        path, caught = _record(seg.save, tmp_path / "shot")

        assert path == tmp_path / "shot.npz"
        assert [w.category for w in caught] == [UserWarning]
        message = str(caught[0].message)
        assert message.startswith(f"{path}: ")
        assert "axes without fs are not saved" in message
        assert Path(caught[0].filename).name == Path(__file__).name

    def test_index_axes_without_fs_save_silently(self, tmp_path):
        spec = prepare(np.random.default_rng(0).random((H, W)))
        seg = Segmentation(_mask(), spec)

        _, caught = _record(seg.save, tmp_path / "shot.npz")

        assert not caught

    def test_bundle_takes_axes_or_stft_meta_not_both(self):
        with pytest.raises(ValueError, match="not both"):
            export.analysis_bundle(
                spectrogram=np.zeros((H, W)),
                axes=(np.arange(W), np.arange(H)),
                stft_meta={"fs": 1000.0},
            )

    @pytest.mark.parametrize(
        "axes",
        [
            (np.arange(W + 1.0), np.arange(H * 1.0)),
            (np.arange(W * 1.0), np.arange(H - 1.0)),
            (np.zeros((1, W)), np.arange(H * 1.0)),
            (np.array([0.0, np.nan, 2.0]), np.arange(H * 1.0)),
        ],
        ids=["long-time", "short-freq", "2d-time", "nan"],
    )
    def test_bundle_rejects_bad_axes(self, axes):
        with pytest.raises(ValueError, match="time_ms|freq_khz"):
            export.analysis_bundle(spectrogram=np.zeros((H, W)), axes=axes)

    def test_bundle_stores_given_axes_as_float64(self):
        bundle = export.analysis_bundle(
            spectrogram=np.zeros((H, W)),
            axes=(np.arange(W, dtype=np.float32), np.arange(H)),
        )

        assert bundle["time_ms"].dtype == np.float64
        assert bundle["freq_khz"].dtype == np.float64
        np.testing.assert_array_equal(bundle["freq_khz"], np.arange(H))


class TestLoadConfig:
    def test_unknown_config_fields_warn_and_the_rest_is_kept(self, tmp_path):
        cfg = SpectrogramConfig(n_fft=256, hop=32, clip_dc=False)
        path = _save_bundle(
            tmp_path / "b.npz", params={"config": {**cfg.to_dict(), "future_field": 1}}
        )

        seg, message = _load_one_warning(path)

        assert "future_field" in message
        assert seg.spectrogram.config == cfg

    def test_an_invalid_clip_pair_falls_back_alone(self, tmp_path):
        params = {
            "n_fft": 512,
            "hop": 64,
            "clip_dc": False,
            "clip_low": 50,
            "clip_high": 40,
        }
        path = _save_bundle(tmp_path / "b.npz", params=params)

        seg, message = _load_one_warning(path)

        assert "clip_low=50" in message
        assert "clip_high=40" in message
        assert seg.spectrogram.config == SpectrogramConfig(
            n_fft=512, hop=64, clip_dc=False
        )

    def test_a_valid_clip_pair_is_kept(self, tmp_path):
        path = _save_bundle(
            tmp_path / "b.npz", params={"clip_low": 99.5, "clip_high": 100}
        )

        seg = _load_silently(path)

        assert seg.spectrogram.config.clip_low == 99.5
        assert seg.spectrogram.config.clip_high == 100.0

    def test_integral_floats_are_read_as_integers(self, tmp_path):
        path = _save_bundle(tmp_path / "b.npz", params={"n_fft": 512.0})

        seg = _load_silently(path)

        assert seg.spectrogram.config.n_fft == 512
        assert type(seg.spectrogram.config.n_fft) is int

    def test_an_invalid_field_falls_back_to_its_default(self, tmp_path):
        path = _save_bundle(tmp_path / "b.npz", params={"hop": 0, "n_fft": 256})

        seg, message = _load_one_warning(path)

        assert "hop=0" in message
        assert f"using the default {DEFAULT_CONFIG.hop!r}" in message
        assert seg.spectrogram.config == SpectrogramConfig(n_fft=256)

    def test_zero_and_one_flags_from_a_file_are_bools(self, tmp_path):
        path = _save_bundle(
            tmp_path / "b.npz", params={"config": {"clip_dc": 0, "log": 1}}
        )

        seg = _load_silently(path)

        assert seg.spectrogram.config.clip_dc is False
        assert seg.spectrogram.config.log is True

    def test_a_config_that_is_not_a_mapping_uses_the_flat_settings(self, tmp_path):
        path = _save_bundle(tmp_path / "b.npz", params={"config": [1], "hop": 64})

        seg, message = _load_one_warning(path)

        assert "params 'config' is not a mapping" in message
        assert seg.spectrogram.config == SpectrogramConfig(hop=64)


class TestLoadRecordedValues:
    AXES = (np.array([0.0, 2.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]))

    @pytest.mark.parametrize("fs", [-1, True, "1000", float("inf")])
    def test_an_invalid_fs_warns_and_gives_index_axes(self, tmp_path, fs):
        params = {"config": {"n_fft": 256}, "fs": fs, "kind": "stft"}
        path = _save_bundle(tmp_path / "b.npz", params=params, axes=self.AXES)

        seg, message = _load_one_warning(path)

        assert f"fs={fs!r}" in message
        assert "bin/frame indices" in message
        assert seg.fs is None
        freqs, times = _axes((H, W), seg.spectrogram.config, None)
        np.testing.assert_array_equal(seg.freqs, freqs)
        np.testing.assert_array_equal(seg.times, times)

    def test_an_unknown_kind_warns(self, tmp_path):
        path = _save_bundle(tmp_path / "b.npz", params={"kind": "bogus"})

        seg, message = _load_one_warning(path)

        assert "kind='bogus'" in message
        assert seg.spectrogram.kind == "spectrogram"

    def test_channel_names_of_the_wrong_count_warn(self, tmp_path):
        path = _save_bundle(tmp_path / "b.npz", params={"channels": ["a"]})

        seg, message = _load_one_warning(path)

        assert "channels=['a']" in message
        assert seg.channels == ("coherent", "transient")

    def test_non_string_channel_names_warn(self, tmp_path):
        path = _save_bundle(tmp_path / "b.npz", params={"channels": ["a", 2]})

        seg, _ = _load_one_warning(path)

        assert seg.channels == ("coherent", "transient")

    @pytest.mark.parametrize(
        ("time_ms", "shown"),
        [
            (np.arange(W + 2, dtype=np.float64), f"array of shape ({W + 2},)"),
            (np.array([0.0, np.nan, 4.0]), f"array of shape ({W},)"),
        ],
        ids=["wrong-length", "nan"],
    )
    def test_invalid_stored_axes_warn_and_are_rebuilt(self, tmp_path, time_ms, shown):
        params = {"fs": 1000.0, "kind": "spectrogram", "config": {"hop": 10}}
        path = _save_bundle(
            tmp_path / "b.npz", params=params, axes=self.AXES, time_ms=time_ms
        )

        seg, message = _load_one_warning(path)

        assert f"time_ms={shown}" in message
        freqs, times = _axes((H, W), SpectrogramConfig(hop=10), 1000.0)
        np.testing.assert_allclose(seg.times, times)
        np.testing.assert_allclose(seg.freqs, self.AXES[1] * 1e3)  # stored, valid
        assert not np.allclose(freqs, seg.freqs)

    def test_valid_stored_axes_win_over_rebuilt_ones(self, tmp_path):
        params = {"fs": 1000.0, "kind": "stft", "config": {}}
        path = _save_bundle(tmp_path / "b.npz", params=params, axes=self.AXES)

        seg = _load_silently(path)

        np.testing.assert_allclose(seg.times, self.AXES[0] / 1e3)
        np.testing.assert_allclose(seg.freqs, self.AXES[1] * 1e3)

    def test_a_missing_stored_axis_is_rebuilt_silently(self, tmp_path):
        params = {"fs": 1000.0, "kind": "spectrogram"}
        path = _save_bundle(tmp_path / "b.npz", params=params, axes=self.AXES)
        with np.load(path) as data:
            kept = {k: data[k] for k in data.files if k != "time_ms"}
        np.savez(path, **kept)

        seg = _load_silently(path)

        _, times = _axes((H, W), DEFAULT_CONFIG, 1000.0)
        np.testing.assert_allclose(seg.times, times)
        np.testing.assert_allclose(seg.freqs, self.AXES[1] * 1e3)

    def test_params_json_that_is_not_an_object_warns(self, tmp_path):
        path = _save_bundle(
            tmp_path / "b.npz", params={}, params_json=json.dumps([1, 2])
        )

        seg, message = _load_one_warning(path)

        assert re.search(r"params_json.*not a JSON object", message)
        assert seg.spectrogram.config == SpectrogramConfig()
        assert seg.model == "unknown"
