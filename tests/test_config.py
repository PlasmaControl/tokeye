from __future__ import annotations

import json

import numpy as np
import pytest

from tokeye.config import (
    DEFAULT_CHANNELS,
    DEFAULT_CONFIG,
    SpectrogramConfig,
    resolve_channels,
)


def test_defaults_match_training_recipe():
    cfg = SpectrogramConfig()
    assert cfg.to_dict() == {
        "n_fft": 1024,
        "hop": 128,
        "window": "hann",
        "clip_dc": True,
        "clip_low": 1.0,
        "clip_high": 99.0,
        "log": False,
    }
    assert cfg == DEFAULT_CONFIG


def test_round_trip_through_json():
    cfg = SpectrogramConfig(n_fft=256, hop=64, clip_dc=False, log=True)
    again = SpectrogramConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg


def test_numpy_scalars_are_coerced_to_python_types():
    cfg = SpectrogramConfig(n_fft=np.int64(512), clip_low=np.float32(2.0))
    assert type(cfg.n_fft) is int
    assert type(cfg.clip_low) is float
    assert SpectrogramConfig(clip_low=1).clip_low == 1.0


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"n_fft": 1}, ValueError),
        ({"hop": 0}, ValueError),
        ({"clip_low": 50.0, "clip_high": 50.0}, ValueError),
        ({"clip_high": 101.0}, ValueError),
        ({"clip_low": -1.0}, ValueError),
        ({"n_fft": 1024.0}, TypeError),
        ({"n_fft": True}, TypeError),
        ({"clip_dc": 1}, TypeError),
        ({"log": "yes"}, TypeError),
        ({"window": ""}, TypeError),
        ({"clip_low": "1"}, TypeError),
    ],
)
def test_invalid_values_are_rejected(changes, error):
    with pytest.raises(error):
        SpectrogramConfig(**changes)


def test_from_dict_rejects_unknown_keys_and_lists_valid_ones():
    with pytest.raises(ValueError, match="hop_length") as info:
        SpectrogramConfig.from_dict({"hop_length": 64})
    assert "'hop'" in str(info.value)


def test_replace_validates():
    cfg = SpectrogramConfig()
    assert cfg.replace(hop=64).hop == 64
    assert cfg.hop == 128  # frozen original untouched
    with pytest.raises(ValueError):
        cfg.replace(hop=0)


def test_coerce_accepts_none_config_and_mapping():
    cfg = SpectrogramConfig(hop=64)
    assert SpectrogramConfig.coerce(None) == SpectrogramConfig()
    assert SpectrogramConfig.coerce(cfg) is cfg
    assert SpectrogramConfig.coerce({"hop": 64}) == cfg
    with pytest.raises(TypeError):
        SpectrogramConfig.coerce(64)


def test_stft_kwargs_excludes_log():
    kwargs = SpectrogramConfig().stft_kwargs()
    assert "log" not in kwargs
    assert kwargs["window"] == "hann"


def test_resolve_channels():
    assert resolve_channels(DEFAULT_CHANNELS, 2) == ("coherent", "transient")
    assert resolve_channels(DEFAULT_CHANNELS, 3) == (
        "channel_0",
        "channel_1",
        "channel_2",
    )


def test_config_module_is_stdlib_only():
    import subprocess
    import sys

    code = (
        "import sys, tokeye.config; "
        "bad = [m for m in ('numpy', 'scipy', 'torch') if m in sys.modules]; "
        "assert not bad, bad"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
