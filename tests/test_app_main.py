"""Tests for src/tokeye/app/__main__.py main() function."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from tokeye.app.__main__ import DEFAULT_PORT, MAX_PORT_ATTEMPTS, main


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
