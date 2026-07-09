"""Pure-math GUI render helpers — no Qt needed, so these run in any env."""

from __future__ import annotations

import numpy as np
import pytest

from tokeye.gui.render import (
    auto_levels,
    axis_rect,
    discrete_mode_image,
    freq_crop_rows,
    is_rgb_view,
    mode_ticks,
    nd_masked_for_display,
    spectrogram_rect,
    view_display_array,
)

_META = {
    "fs": 1_000_000.0,
    "t0_ms": 100.0,
    "n_fft": 1024,
    "hop": 256,
    "clip_dc": True,
}


def test_spectrogram_rect_real_axes():
    x_left, y_bottom, x_span, y_span = spectrogram_rect(_META, n_rows=512, n_cols=40)
    df = 1_000_000.0 / 1024 / 1e3  # kHz per row
    dt = 256 / 1_000_000.0 * 1e3  # ms per column
    assert x_left == pytest.approx(100.0 - dt / 2)
    assert y_bottom == pytest.approx(1 * df - df / 2)  # offset=1 (clip_dc)
    assert x_span == pytest.approx(40 * dt)
    assert y_span == pytest.approx(512 * df)


def test_spectrogram_rect_crop_offset():
    # r_lo shifts the bottom frequency up by r_lo rows.
    df = 1_000_000.0 / 1024 / 1e3
    _, y_bottom, _, y_span = spectrogram_rect(_META, n_rows=100, n_cols=10, r_lo=20)
    assert y_bottom == pytest.approx((20 + 1) * df - df / 2)
    assert y_span == pytest.approx(100 * df)


def test_spectrogram_rect_pixel_fallback_without_fs():
    assert spectrogram_rect(None, n_rows=100, n_cols=50) == (0.0, 0.0, 50.0, 100.0)
    assert spectrogram_rect({"fs": 0.0}, n_rows=8, n_cols=4) == (0.0, 0.0, 4.0, 8.0)


def test_freq_crop_rows_no_meta():
    assert freq_crop_rows(300, None) == (0, 300)
    assert freq_crop_rows(300, {"fs": 0.0}) == (0, 300)


def test_freq_crop_rows_fmax():
    meta = {"fs": 1_000_000.0, "n_fft": 1024, "clip_dc": True, "fmax_khz": 100.0}
    r_lo, r_hi = freq_crop_rows(512, meta)
    assert r_lo == 0
    # bin(100 kHz) = floor(100e3*1024/1e6)=102; minus offset(1) plus 1 = 102
    assert r_hi == 102


def test_freq_crop_rows_degenerate_returns_full():
    meta = {"fs": 1_000_000.0, "n_fft": 1024, "clip_dc": True,
            "fmin_khz": 400.0, "fmax_khz": 401.0}
    assert freq_crop_rows(512, meta) == (0, 512)


def test_auto_levels():
    arr = np.arange(100.0)
    lo, hi = auto_levels(arr)
    assert lo == 0.0
    assert hi == pytest.approx(float(np.quantile(arr, 0.95)))


def test_auto_levels_all_nan_is_safe():
    assert auto_levels(np.full((4, 4), np.nan)) == (0.0, 1.0)


def test_view_display_array_shapes():
    rng = np.random.default_rng(0)
    arr = rng.random((20, 10))
    extract = rng.random((2, 20, 10))
    assert view_display_array("Original", arr, None, True, True, 1, 99, 0.5).shape == (
        20,
        10,
    )
    assert view_display_array("Enhanced", arr, extract, True, True, 1, 99, 0.5).shape == (
        20,
        10,
        3,
    )
    assert view_display_array("Mask", arr, extract, True, True, 1, 99, 0.5).shape == (
        20,
        10,
        3,
    )
    assert view_display_array("Amplitude", arr, extract, True, True, 1, 99, 0.5).shape == (
        20,
        10,
    )
    assert is_rgb_view("Enhanced") and is_rgb_view("Mask")
    assert not is_rgb_view("Original") and not is_rgb_view("Amplitude")


def test_view_display_array_missing_inputs_return_none():
    assert view_display_array("Original", None, None, True, True, 1, 99, 0.5) is None
    arr = np.zeros((4, 4))
    assert view_display_array("Enhanced", arr, None, True, True, 1, 99, 0.5) is None


# ── modespec map helpers ──────────────────────────────────────────────────────
def test_axis_rect_pixel_centres():
    x = np.linspace(1000.0, 1020.0, 30)
    y = np.linspace(5.0, 150.0, 25)
    x_left, y_bottom, x_span, y_span = axis_rect(x, y)
    dt = (1020.0 - 1000.0) / 29
    df = (150.0 - 5.0) / 24
    assert x_left == pytest.approx(1000.0 - dt / 2)
    assert x_span == pytest.approx(30 * dt)
    assert y_bottom == pytest.approx(5.0 - df / 2)
    assert y_span == pytest.approx(25 * df)


def test_mode_ticks_signed_integers():
    assert [t[1] for t in mode_ticks(-3, 3)] == [
        "-3", "-2", "-1", "+0", "+1", "+2", "+3",
    ]
    assert [t[0] for t in mode_ticks(-3, 3)] == [-3, -2, -1, 0, 1, 2, 3]


def test_discrete_mode_image_alpha_marks_suppressed():
    # (n_win=2, n_freq=2); NaN entries must become transparent.
    nd = np.array([[0.0, 1.0], [np.nan, -1.0]])
    rgba = discrete_mode_image(nd, -1, 1)
    assert rgba.shape == (2, 2, 4)  # (n_freq, n_win, RGBA)
    zt = nd.T  # displayed orientation
    assert (rgba[..., 3][np.isfinite(zt)] == 255).all()
    assert (rgba[..., 3][~np.isfinite(zt)] == 0).all()


def test_nd_masked_coherence_gate_and_passthrough():
    result = {
        "coherence": np.array([[0.1, 0.9]]),
        "n_dominant": np.array([[2, 3]]),
        "c95": 0.0,
    }
    out = nd_masked_for_display(result, None, 0.5)
    assert np.isnan(out[0, 0]) and out[0, 1] == 3  # coh<0.5 suppressed
    # a pre-gated nd is returned unchanged (coh_thresh ignored)
    nd = np.array([[np.nan, 1.0]])
    out2 = nd_masked_for_display(result, nd, 0.9)
    assert np.isnan(out2[0, 0]) and out2[0, 1] == 1.0
