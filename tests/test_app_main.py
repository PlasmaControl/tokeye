"""Tests for src/tokeye/app/__main__.py main() function."""

from __future__ import annotations

import threading
import time
from unittest.mock import Mock, patch

import pytest

from tokeye.app import __main__ as appmain
from tokeye.app.__main__ import DEFAULT_PORT, MAX_PORT_ATTEMPTS, create_app, main
from tokeye.app.utils.theme import PALETTE, make_theme


class TestMainPortRetry:
    """Test the port-retry logic in main()."""

    def test_retry_upward_on_oserror(self):
        """Port should increment on OSError, not decrement."""
        fake_app = Mock()
        # Fail twice on ports 7860 and 7861, succeed on 7862
        fake_app.launch.side_effect = [
            OSError("Port in use"),
            OSError("Port in use"),
            None,  # success
        ]

        with patch("tokeye.app.__main__.create_app", return_value=fake_app):
            main(port=DEFAULT_PORT)

        # Verify the ports tried were 7860, 7861, 7862
        assert fake_app.launch.call_count == 3
        calls = fake_app.launch.call_args_list
        assert calls[0][1]["server_port"] == DEFAULT_PORT
        assert calls[1][1]["server_port"] == DEFAULT_PORT + 1
        assert calls[2][1]["server_port"] == DEFAULT_PORT + 2

    def test_return_on_success(self):
        """main() should return (not fall through) after successful launch."""
        fake_app = Mock()
        fake_app.launch.return_value = None  # success

        with patch("tokeye.app.__main__.create_app", return_value=fake_app):
            result = main(port=DEFAULT_PORT)

        # Should return cleanly
        assert result is None
        # Should only try once
        assert fake_app.launch.call_count == 1

    def test_systemeexit_after_all_attempts_fail(self):
        """Should raise SystemExit naming the port range after all attempts fail."""
        fake_app = Mock()
        fake_app.launch.side_effect = OSError("Port in use")

        with patch("tokeye.app.__main__.create_app", return_value=fake_app), \
                pytest.raises(SystemExit) as exc_info:
            main(port=DEFAULT_PORT)

        # Check the message names the port range
        message = str(exc_info.value)
        assert "7860" in message
        assert str(DEFAULT_PORT + MAX_PORT_ATTEMPTS - 1) in message
        # Should have tried all MAX_PORT_ATTEMPTS
        assert fake_app.launch.call_count == MAX_PORT_ATTEMPTS

    def test_launch_kwargs_preserved(self):
        """launch() should receive all original kwargs, only port changes."""
        fake_app = Mock()
        fake_app.launch.return_value = None

        with patch("tokeye.app.__main__.create_app", return_value=fake_app):
            main(port=7777, share=True, open_browser=True)

        fake_app.launch.assert_called_once_with(
            share=True, inbrowser=True, server_port=7777
        )

    def test_non_oserror_exceptions_propagate(self):
        """Non-OSError exceptions should propagate (not be caught by retry logic)."""
        fake_app = Mock()
        fake_app.launch.side_effect = RuntimeError("Some other error")

        with patch("tokeye.app.__main__.create_app", return_value=fake_app), \
                pytest.raises(RuntimeError):
            main(port=DEFAULT_PORT)


class TestDarkControlRoomTheme:
    """Tests for the dark control-room theme (mirrors the native Qt GUI palette)."""

    def test_body_background_is_dark_in_both_variants(self):
        """The page background must be forced dark regardless of browser preference."""
        t = make_theme()
        assert t.body_background_fill == PALETTE["bg_window"]
        assert t.body_background_fill_dark == PALETTE["bg_window"]
        assert t.body_background_fill == t.body_background_fill_dark

    def test_block_background_is_dark_in_both_variants(self):
        t = make_theme()
        assert t.block_background_fill == PALETTE["bg_surface"]
        assert t.block_background_fill_dark == PALETTE["bg_surface"]

    def test_body_text_color_is_dark_in_both_variants(self):
        t = make_theme()
        assert t.body_text_color == PALETTE["text"]
        assert t.body_text_color_dark == PALETTE["text"]

    def test_primary_button_uses_accent_in_both_variants(self):
        """The accent-filled primary button must match the shared palette exactly."""
        t = make_theme()
        assert t.button_primary_background_fill == PALETTE["accent"]
        assert t.button_primary_background_fill_dark == PALETTE["accent"]
        assert t.button_primary_background_fill == t.button_primary_background_fill_dark

    def test_primary_button_text_uses_accent_text_in_both_variants(self):
        t = make_theme()
        assert t.button_primary_text_color == PALETTE["accent_text"]
        assert t.button_primary_text_color_dark == PALETTE["accent_text"]

    def test_input_background_uses_bg_input_in_both_variants(self):
        t = make_theme()
        assert t.input_background_fill == PALETTE["bg_input"]
        assert t.input_background_fill_dark == PALETTE["bg_input"]

    def test_input_focus_border_uses_accent_in_both_variants(self):
        t = make_theme()
        assert t.input_border_color_focus == PALETTE["accent"]
        assert t.input_border_color_focus_dark == PALETTE["accent"]

    def test_slider_uses_accent_in_both_variants(self):
        t = make_theme()
        assert t.slider_color == PALETTE["accent"]
        assert t.slider_color_dark == PALETTE["accent"]

    def test_block_label_text_uses_muted_color_in_both_variants(self):
        t = make_theme()
        assert t.block_label_text_color == PALETTE["text_muted"]
        assert t.block_label_text_color_dark == PALETTE["text_muted"]

    def test_palette_hex_values_are_the_cross_branch_contract(self):
        """These hex values mirror gui/theme.py::COLORS on the diiid branch —
        change only in lockstep with that file."""
        assert PALETTE == {
            "bg_window": "#13151a",
            "bg_surface": "#1b1e26",
            "bg_raised": "#22262f",
            "bg_input": "#0f1115",
            "border": "#2a2f3a",
            "text": "#e9ecf1",
            "text_muted": "#8b93a1",
            "accent": "#45b8cb",
            "accent_hover": "#63d0e2",
            "accent_pressed": "#3aa2b3",
            "accent_text": "#08222a",
        }

    def test_create_app_builds_with_dark_theme(self):
        """create_app() must still build cleanly with the new theme + CSS wired in."""
        app = create_app()
        assert app is not None


# PALETTE key -> COLORS key. Locks the web mirror to the GUI source of truth.
_PARITY = {
    "bg_window": "bg", "bg_surface": "panel", "bg_raised": "panel2",
    "border": "line", "text": "text", "text_muted": "muted",
    "accent": "accent", "accent_hover": "accentHi",
    "accent_text": "accentInk",
}


class TestPaletteParity:
    """The three copies of the control-room palette (web PALETTE, GUI COLORS,
    viz.py module constants) must never drift — one test locks them together."""

    def test_web_palette_mirrors_gui_colors(self):
        """Every mapped PALETTE hex equals its GUI COLORS counterpart."""
        from tokeye.gui.theme import COLORS

        for palette_key, colors_key in _PARITY.items():
            assert PALETTE[palette_key] == COLORS[colors_key], (
                f"PALETTE[{palette_key!r}] != COLORS[{colors_key!r}]"
            )

    def test_gui_theme_module_imports_no_qt(self):
        """Importing gui.theme must stay Qt-free (Qt is deferred into functions),
        so this parity check never drags PySide6 into a headless test run.
        Fresh interpreter = order-independent."""
        import subprocess
        import sys

        subprocess.run(
            [
                sys.executable,
                "-c",
                "import tokeye.gui.theme, sys; assert 'PySide6' not in sys.modules",
            ],
            check=True,
        )

    def test_viz_constants_match_gui_colors(self):
        """viz.py's Plotly palette constants mirror the same GUI COLORS."""
        from tokeye.gui.theme import COLORS
        from tokeye.sources import viz

        assert COLORS["bg"] == viz._PAPER_HEX
        assert COLORS["plot"] == viz._PLOT_HEX
        assert COLORS["accent"] == viz._ACCENT_HEX
        assert COLORS["line"] == viz._LINE_HEX  # grid / axis lines
        assert COLORS["muted"] == viz._MUTED_HEX  # axis + label text
        # Same two constants mirror the web PALETTE keys they were derived from.
        assert PALETTE["border"] == viz._LINE_HEX
        assert PALETTE["text_muted"] == viz._MUTED_HEX


def _wait_for(predicate, timeout=5.0):
    """Poll ``predicate`` until true or ``timeout`` elapses; assert it became true."""
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate(), "condition was not met within the timeout"


@pytest.fixture
def shot_state():
    """Reset the module-level latest-shot cache/executor/pending around a test.

    The prefill state (cache dict, lazy executor, in-flight future) lives in
    module globals; without a reset the background fetch and TTL cache leak
    between tests. Yields the __main__ module so tests can poke those globals.
    """

    def _reset():
        if appmain._SHOT_EXECUTOR is not None:
            appmain._SHOT_EXECUTOR.shutdown(wait=False)
        appmain._SHOT_EXECUTOR = None
        appmain._SHOT_PENDING = None
        appmain._SHOT_CACHE["value"] = 0
        appmain._SHOT_CACHE["ts"] = None

    _reset()
    yield appmain
    _reset()


class TestLatestShotPrefill:
    """The non-blocking latest-shot prefill: TTL cache, timeout, fold-back."""

    def test_cache_hit_fetches_only_once(self, monkeypatch, shot_state):
        """Two calls inside the TTL both return the shot; MDS is hit once."""
        m = shot_state
        monkeypatch.setattr(m, "_SHOT_TIMEOUT_S", 2.0)
        monkeypatch.setattr(m, "_SHOT_TTL_S", 60.0)
        calls = {"n": 0}

        def fetcher():
            calls["n"] += 1
            return 199999

        monkeypatch.setattr("tokeye.sources.factory.latest_shot", fetcher)

        assert m._latest_shot() == 199999
        assert m._latest_shot() == 199999
        assert calls["n"] == 1

    def test_timeout_falls_back_without_stacking(self, monkeypatch, shot_state):
        """A hung fetch times out to the fallback, is never stacked, and its late
        result is folded into the cache for the next call."""
        m = shot_state
        monkeypatch.setattr(m, "_SHOT_TIMEOUT_S", 0.05)
        monkeypatch.setattr(m, "_SHOT_TTL_S", 60.0)

        gate = threading.Event()
        entered = threading.Event()
        calls = {"n": 0}

        def blocking_fetcher():
            calls["n"] += 1
            entered.set()
            gate.wait(5.0)  # safety cap so a failed assert can't hang the suite
            return 199999

        monkeypatch.setattr("tokeye.sources.factory.latest_shot", blocking_fetcher)

        # First call: the fetch blocks -> times out fast -> fallback 0.
        t0 = time.monotonic()
        assert m._latest_shot() == 0
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0, f"timeout path took too long: {elapsed:.3f}s"
        assert entered.wait(1.0), "worker never entered the fetch"

        # Second call while still blocked: reuse the pending future, no new submit.
        assert m._latest_shot() == 0
        assert calls["n"] == 1, "a second fetch was stacked behind the hung one"

        # Release the fetch: the done-callback must fold the result into the cache.
        gate.set()
        _wait_for(lambda: m._SHOT_CACHE["ts"] is not None)
        assert calls["n"] == 1
        # Next call gets the real shot instantly from the folded cache.
        assert m._latest_shot() == 199999

    def test_stale_cache_falls_back_to_last_known(self, monkeypatch, shot_state):
        """A failing refresh of a stale cache returns the last good shot, not 0."""
        m = shot_state
        monkeypatch.setattr(m, "_SHOT_TIMEOUT_S", 2.0)
        monkeypatch.setattr(m, "_SHOT_TTL_S", 60.0)

        state = {"fail": False}

        def fetcher():
            if state["fail"]:
                raise RuntimeError("atlas unreachable")
            return 199999

        monkeypatch.setattr("tokeye.sources.factory.latest_shot", fetcher)

        # Prime the cache with a good shot.
        assert m._latest_shot() == 199999
        # Let the done-callback clear the pending slot so a stale call re-fetches
        # rather than reusing the still-good future.
        _wait_for(lambda: m._SHOT_PENDING is None)

        # Force the cache stale and make the refresh fail.
        m._SHOT_CACHE["ts"] = time.monotonic() - 999.0
        state["fail"] = True
        assert m._latest_shot() == 199999  # last known, not 0

    def test_pair_returns_the_value_twice(self, monkeypatch, shot_state):
        """_latest_shot_pair feeds both DIII-D shot fields from one fetch."""
        m = shot_state
        monkeypatch.setattr(m, "_latest_shot", lambda: 190904)
        assert m._latest_shot_pair() == (190904, 190904)
