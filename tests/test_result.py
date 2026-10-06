from __future__ import annotations

import json
import subprocess
import sys

import matplotlib as mpl
import numpy as np
import pytest

from tokeye import export
from tokeye._plotting import overlay_rgba, save_preview
from tokeye.config import SpectrogramConfig
from tokeye.preprocess import prepare
from tokeye.result import Segmentation

mpl.use("Agg")

CFG = SpectrogramConfig(n_fft=128, hop=32)
FS = 8_000.0


def _segmentation(fs: float | None = FS) -> Segmentation:
    rng = np.random.default_rng(0)
    spec = prepare(rng.standard_normal(2048), CFG, fs=fs)
    mask = rng.random((2, *spec.shape)).astype(np.float32)
    return Segmentation(mask, spec, ("coherent", "transient"), "big_tf_unet")


class TestConstruction:
    def test_channel_access(self):
        seg = _segmentation()
        np.testing.assert_array_equal(seg.coherent, seg.mask[0])
        np.testing.assert_array_equal(seg["transient"], seg.mask[1])
        assert seg.freqs is seg.spectrogram.freqs
        assert seg.times is seg.spectrogram.times
        assert seg.fs == FS

    def test_unknown_channel_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="coherent"):
            _segmentation()["background"]

    def test_threshold_is_boolean(self):
        seg = _segmentation()
        masks = seg.threshold(0.5)
        assert masks.dtype == bool
        np.testing.assert_array_equal(masks, seg.mask >= 0.5)

    def test_shape_mismatch_raises(self):
        seg = _segmentation()
        with pytest.raises(ValueError, match="does not match"):
            Segmentation(seg.mask[:, :-1], seg.spectrogram)

    def test_channel_count_mismatch_raises(self):
        seg = _segmentation()
        with pytest.raises(ValueError, match="channel name"):
            Segmentation(seg.mask, seg.spectrogram, ("coherent",))


class TestSaveLoad:
    def test_round_trip_with_fs(self, tmp_path):
        seg = _segmentation()
        path = seg.save(tmp_path / "shot")

        assert path.name == "shot.npz"
        again = Segmentation.load(path)
        np.testing.assert_array_equal(again.mask, seg.mask)
        np.testing.assert_array_equal(again.spectrogram.values, seg.spectrogram.values)
        np.testing.assert_allclose(again.times, seg.times)
        np.testing.assert_allclose(again.freqs, seg.freqs)
        assert again.fs == FS
        assert again.spectrogram.config == CFG
        assert again.spectrogram.kind == "stft"
        assert again.channels == seg.channels
        assert again.model == "big_tf_unet"

    def test_bundle_follows_the_analysis_schema(self, tmp_path):
        path = _segmentation().save(tmp_path / "shot.npz")
        with np.load(path) as data:
            assert str(data["schema"]) == export.SCHEMA_ANALYSIS
            assert str(data["source"]) == "segment"
            params = json.loads(str(data["params_json"]))
        assert params["config"] == CFG.to_dict()
        assert params["channels"] == ["coherent", "transient"]
        assert "tokeye_version" in params

    def test_round_trip_without_fs_uses_index_axes(self, tmp_path):
        seg = _segmentation(fs=None)
        again = Segmentation.load(seg.save(tmp_path / "shot.npz"))
        assert again.fs is None
        np.testing.assert_array_equal(again.times, np.arange(seg.mask.shape[2]))

    def test_round_trip_keeps_n_samples(self, tmp_path):
        seg = _segmentation()
        assert seg.spectrogram.n_samples == 2048

        again = Segmentation.load(seg.save(tmp_path / "shot.npz"))

        assert again.spectrogram.n_samples == 2048

    def test_round_trip_of_a_2d_input_keeps_n_samples_none(self, tmp_path):
        spec = prepare(np.random.default_rng(0).random((8, 6)), CFG, fs=FS)
        seg = Segmentation(np.zeros((2, 8, 6)), spec)

        again = Segmentation.load(seg.save(tmp_path / "shot.npz"))

        assert again.spectrogram.n_samples is None

    def test_an_invalid_n_samples_warns_and_gives_none(self, tmp_path):
        bundle = export.analysis_bundle(
            spectrogram=np.zeros((4, 3)),
            mask=np.zeros((2, 4, 3)),
            params={"n_samples": "x"},
            source="analyze",
        )
        path = export.save_npz(tmp_path / "b.npz", bundle)

        with pytest.warns(UserWarning, match=r"ignoring n_samples='x' .*using None"):
            seg = Segmentation.load(path)

        assert seg.spectrogram.n_samples is None

    def test_loads_app_bundles(self, tmp_path):
        bundle = export.analysis_bundle(
            spectrogram=np.zeros((6, 8)),
            mask=np.zeros((2, 6, 8)),
            params={"model": "big_tf_unet", "n_fft": 512, "hop": 64, "clip_dc": True},
            source="analyze",
        )
        seg = Segmentation.load(export.save_npz(tmp_path / "app.npz", bundle))
        assert seg.spectrogram.config == SpectrogramConfig(n_fft=512, hop=64)
        assert seg.channels == ("coherent", "transient")
        assert seg.fs is None

    def test_two_d_mask_loads_as_one_channel(self, tmp_path):
        bundle = export.analysis_bundle(
            spectrogram=np.zeros((4, 4)), mask=np.ones((4, 4))
        )
        seg = Segmentation.load(export.save_npz(tmp_path / "one.npz", bundle))
        assert seg.mask.shape == (1, 4, 4)
        assert seg.channels == ("channel_0",)
        assert seg.model == "unknown"

    def test_bundle_without_mask_raises(self, tmp_path):
        bundle = export.analysis_bundle(spectrogram=np.zeros((4, 4)))
        path = export.save_npz(tmp_path / "nomask.npz", bundle)
        with pytest.raises(ValueError, match="no mask"):
            Segmentation.load(path)

    def test_other_schema_raises(self, tmp_path):
        path = tmp_path / "other.npz"
        np.savez(path, schema=np.array("something/else"))
        with pytest.raises(ValueError, match="not a tokeye-analysis/v1"):
            Segmentation.load(path)


class TestPlotting:
    def test_plot_labels_axes_in_physical_units(self):
        ax = _segmentation().plot()
        assert ax.get_xlabel() == "Time [s]"
        assert ax.get_ylabel() == "Frequency [Hz]"
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            "coherent",
            "transient",
        ]

    def test_plot_without_fs_labels_indices(self):
        ax = _segmentation(fs=None).plot()
        assert ax.get_xlabel() == "Frame"

    def test_overlay_later_channels_paint_over_earlier(self):
        mask = np.ones((2, 2, 2))
        rgba = overlay_rgba(mask, channels=("coherent", "transient"))
        np.testing.assert_allclose(rgba[0, 0], (1.0, 0.0, 0.0, 0.4))

    def test_overlay_below_threshold_is_transparent(self):
        rgba = overlay_rgba(np.zeros((2, 3, 3)))
        assert not rgba.any()

    def test_save_preview_writes_png(self, tmp_path):
        path = save_preview(_segmentation(), tmp_path / "p.png")
        assert path.read_bytes()[:4] == b"\x89PNG"

    def test_previews_never_import_pyplot(self, tmp_path):
        code = (
            "import sys, numpy as np; "
            "from tokeye.preprocess import prepare; "
            "from tokeye.result import Segmentation; "
            "from tokeye._plotting import save_preview; "
            "s = prepare(np.random.rand(32, 32)); "
            f"save_preview(Segmentation(np.zeros((2, 32, 32)), s), r'{tmp_path / 'x.png'}'); "
            "assert 'matplotlib.pyplot' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
