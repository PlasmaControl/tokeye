"""The golden guard's helpers; needs no weights, so it runs everywhere.

Never request the session-scoped ``real_weights`` fixture here: pytest caches
its value (and its skip) for the whole session.
"""

from __future__ import annotations

import platform
import sys

import golden_utils
import pytest
import torch


class _Boom(Exception):
    pass


class TestWeightsLookup:
    def test_cached_weights_is_the_hub_lookup(self, tmp_path, monkeypatch):
        sentinel = tmp_path / "w.pt"
        asked = []

        def lookup(name):
            asked.append(name)
            return sentinel

        monkeypatch.setattr("tokeye.hub.cached_path", lookup)

        assert golden_utils.cached_weights() is sentinel
        assert asked == ["big_tf_unet"]

    def test_not_cached(self, monkeypatch):
        monkeypatch.setattr("tokeye.hub.cached_path", lambda name: None)

        assert golden_utils.cached_weights() is None
        with pytest.raises(pytest.skip.Exception, match="tokeye download"):
            golden_utils.weights_or_skip()
        with pytest.raises(SystemExit, match="tokeye download"):
            golden_utils._weights_path()

    def test_cached(self, tmp_path, monkeypatch):
        sentinel = tmp_path / "w.pt"
        monkeypatch.setattr("tokeye.hub.cached_path", lambda name: sentinel)

        assert golden_utils.weights_or_skip() is sentinel
        assert golden_utils._weights_path() is sentinel


@pytest.mark.parametrize("body_raises", [False, True])
def test_fp32_cuda_turns_tf32_off_inside_only(monkeypatch, body_raises):
    cudnn, matmul = torch.backends.cudnn, torch.backends.cuda.matmul
    start = torch.get_float32_matmul_precision()
    try:
        monkeypatch.setattr(cudnn, "allow_tf32", True)
        monkeypatch.setattr(matmul, "allow_tf32", True)
        inside = []

        try:
            with golden_utils.fp32_cuda():
                inside.append((cudnn.allow_tf32, matmul.allow_tf32))
                if body_raises:
                    raise _Boom
        except _Boom:
            assert body_raises

        assert inside == [(False, False)]
        assert (cudnn.allow_tf32, matmul.allow_tf32) == (True, True)
    finally:
        monkeypatch.undo()
        # The undo sets the legacy matmul flag, which leaves torch >= 2.9 with
        # the legacy and the new precision API mixed (reading the precision
        # then raises); setting it through the new API puts both in step.
        torch.set_float32_matmul_precision(start)


class TestMain:
    @pytest.fixture
    def calls(self, tmp_path, monkeypatch):
        """Record the writers' calls; tests/data is never in reach."""
        seen = []
        monkeypatch.setattr(golden_utils, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(golden_utils, "GOLDEN_PATH", tmp_path / "data" / "g.npz")
        monkeypatch.setattr(golden_utils, "STFT_PATH", tmp_path / "data" / "s.npz")
        monkeypatch.setattr(
            golden_utils, "_write_golden", lambda: seen.append("golden")
        )
        monkeypatch.setattr(golden_utils, "_write_stft", lambda: seen.append("stft"))
        return seen

    @pytest.mark.parametrize("which", ["golden", "stft"])
    def test_write_one_file(self, calls, which):
        golden_utils.main(["--write", which])

        assert calls == [which]

    @pytest.mark.parametrize(
        "argv",
        [[], ["--write"], ["--write", "both"], ["--write", "golden", "stft"]],
    )
    def test_anything_else_is_a_usage_error(self, calls, argv):
        with pytest.raises(SystemExit) as info:
            golden_utils.main(argv)

        assert info.value.code == golden_utils.USAGE
        assert "--write golden|stft" in golden_utils.USAGE
        assert calls == []


class TestAtol:
    @staticmethod
    def _here(**changes):
        golden = {
            "torch_version": torch.__version__,
            "platform": sys.platform,
            "machine": platform.machine(),
        }
        golden.update(changes)
        return golden

    def test_exact_only_on_the_cpu_where_it_was_made(self):
        assert golden_utils._atol("cpu", self._here()) == 1e-5
        assert golden_utils._atol("cuda", self._here()) == 1e-4
        assert golden_utils._atol("mps", self._here()) == 1e-4

    def test_a_local_version_suffix_does_not_matter(self):
        base = torch.__version__.split("+")[0]
        golden = self._here(torch_version=f"{base}+somewhere")

        assert golden_utils._atol("cpu", golden) == 1e-5

    @pytest.mark.parametrize(
        "changes",
        [
            {"platform": "no-such-os"},
            {"machine": "no-such-machine"},
            {"torch_version": "0.0.1"},
        ],
    )
    def test_anywhere_else_is_loose(self, changes):
        assert golden_utils._atol("cpu", self._here(**changes)) == 1e-4

    def test_old_goldens_were_made_on_linux_x86_64(self):
        golden = {"torch_version": torch.__version__}
        on_linux_x86_64 = sys.platform == "linux" and platform.machine() == "x86_64"

        assert golden_utils._atol("cpu", golden) == (1e-5 if on_linux_x86_64 else 1e-4)
