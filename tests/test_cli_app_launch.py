"""``tokeye app``: port probing, the launch sequence and its one-line errors.

CLI level only: ``tokeye.app.__main__`` is stubbed (or imported against a
stub gradio), so this file also runs where gradio is not installed. Its name
must not match ``test_app_*.py``, which ``conftest.py`` skips without gradio.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest
from cli_helpers import one_error_line

from tokeye.cli import app as app_cli
from tokeye.cli import build_parser, main
from tokeye.cli.app import PORT_ATTEMPTS, pick_port

REAL_PICK_PORT = pick_port  # the fixture replaces app_cli.pick_port, not this name


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _listening_socket() -> socket.socket:
    """A socket bound and listening on a free port, as a real server holds it.

    macOS hands out ephemeral ports up to 65535, which would leave no room
    for the upward search, so a high port is given back and another tried.
    """
    while True:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        if sock.getsockname()[1] <= 65000:
            return sock
        sock.close()


class TestPickPort:
    def test_a_free_port_is_returned_as_is(self):
        port = _free_port()

        assert pick_port("127.0.0.1", port) == port

    def test_a_busy_port_is_skipped(self):
        with _listening_socket() as held:
            busy = held.getsockname()[1]

            chosen = pick_port("127.0.0.1", busy)

        assert busy < chosen < busy + PORT_ATTEMPTS

    def test_the_probe_sockets_are_closed(self):
        port = pick_port("127.0.0.1", _free_port())

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))

    def test_an_address_of_another_machine_is_one_probe(self, monkeypatch):
        probes = []
        real_probe = app_cli._probe

        def counting(host, port):
            probes.append(port)
            return real_probe(host, port)

        monkeypatch.setattr(app_cli, "_probe", counting)

        with pytest.raises(ValueError, match="is not an address of this machine"):
            pick_port("192.0.2.1", 7860)

        assert len(probes) == 1

    def test_an_unresolvable_host_names_the_host(self, monkeypatch):
        def refuse(*args, **kwargs):
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

        monkeypatch.setattr(socket, "getaddrinfo", refuse)

        with pytest.raises(ValueError, match="nope.invalid.*cannot resolve it"):
            pick_port("nope.invalid", 7860)

    def test_an_undecodable_host_is_the_same_error(self):
        with pytest.raises(ValueError, match="cannot resolve it"):
            pick_port("a" * 70, 7860)

    @pytest.mark.parametrize(
        ("first", "span"), [(7860, "7860-7869"), (65530, "65530-65535")]
    )
    def test_every_port_busy_names_the_range(self, monkeypatch, first, span):
        monkeypatch.setattr(app_cli, "_probe", lambda host, port: "busy")

        with pytest.raises(ValueError, match=f"no free port in {span} on 127.0.0.1"):
            pick_port("127.0.0.1", first)

    def test_another_bind_error_asks_for_another_port(self, monkeypatch):
        class Refusing(socket.socket):
            def bind(self, address):
                raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(socket, "socket", Refusing)

        with pytest.raises(ValueError, match=r"cannot listen on 127.0.0.1:80 \(Perm"):
            pick_port("127.0.0.1", 80)

    def test_duplicate_addresses_are_bound_once(self, monkeypatch):
        port = _free_port()
        info = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))
        monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [info, info])

        assert app_cli._probe("localhost", port) == "free"

    def test_an_unusable_address_family_is_skipped(self, monkeypatch):
        info = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 7860))
        monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [info])

        def no_socket(*args, **kwargs):
            raise OSError(97, "Address family not supported by protocol")

        monkeypatch.setattr(socket, "socket", no_socket)

        assert app_cli._probe("localhost", 7860) == "skipped"


@pytest.fixture
def calls(monkeypatch):
    """Stub tokeye.app.__main__ (no gradio) and keep the environment out."""
    calls = {}

    def fake_app_main(port, share, open_browser, host):
        calls.update(port=port, share=share, open_browser=open_browser, host=host)

    stub = types.ModuleType("tokeye.app.__main__")
    stub.main = fake_app_main
    monkeypatch.setitem(sys.modules, "tokeye.app.__main__", stub)
    monkeypatch.delenv("GRADIO_SERVER_NAME", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.setattr(
        "tokeye.cli.app.pick_port", lambda host, first, attempts=10: first
    )
    return calls


class TestAppHandler:
    def test_a_bad_host_is_one_error_and_never_launches(
        self, calls, monkeypatch, capsys
    ):
        monkeypatch.setattr(app_cli, "pick_port", REAL_PICK_PORT)

        exit_code = main(["app", "--host", "192.0.2.1", "--no-browser"])

        assert exit_code == 2
        line = one_error_line(capsys.readouterr().err)
        assert "192.0.2.1" in line
        assert calls == {}

    def test_a_bad_host_creates_no_workspace(self, calls, monkeypatch, tmp_path):
        monkeypatch.setattr(app_cli, "pick_port", REAL_PICK_PORT)
        monkeypatch.chdir(tmp_path)
        workspace = tmp_path / "ws"

        exit_code = main(["app", "--host", "192.0.2.1", "--workspace", str(workspace)])

        assert exit_code == 2
        assert not workspace.exists()

    def test_a_workspace_that_is_a_file_is_one_error(
        self, calls, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        occupied = tmp_path / "occupied"
        occupied.write_text("x")

        exit_code = main(["app", "--workspace", str(occupied), "--no-browser"])

        assert exit_code == 2
        assert str(occupied) in one_error_line(capsys.readouterr().err)
        assert calls == {}
        assert Path.cwd().samefile(tmp_path)

    def test_a_busy_port_is_named_and_the_hint_follows_it(
        self, calls, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            app_cli, "pick_port", lambda host, first, attempts=10: first + 1
        )
        monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 5 6.7.8.9 22")

        exit_code = main(["app"])

        assert exit_code == 0
        err = capsys.readouterr().err
        assert "note: port 7860 is in use; using 7861" in err
        assert "ssh -L 7861:localhost:7861" in err
        assert "7860:localhost" not in err
        assert calls["port"] == 7861

    def test_a_free_port_prints_no_port_note(self, calls, capsys):
        main(["app", "--no-browser"])

        assert "is in use" not in capsys.readouterr().err

    def test_share_in_an_ssh_session_has_no_forwarding_hint(
        self, calls, monkeypatch, capsys
    ):
        monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 5 6.7.8.9 22")

        main(["app", "--share"])

        err = capsys.readouterr().err
        assert "ssh -L" not in err
        assert "anyone with the link" in err

    def test_gradio_server_name_is_the_host_default(self, calls, monkeypatch):
        monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")

        main(["app", "--no-browser"])
        assert calls["host"] == "0.0.0.0"

        main(["app", "--no-browser", "--host", "127.0.0.1"])
        assert calls["host"] == "127.0.0.1"

    def test_a_blank_gradio_server_name_counts_as_unset(self, calls, monkeypatch):
        monkeypatch.setenv("GRADIO_SERVER_NAME", "  ")

        main(["app", "--no-browser"])

        assert calls["host"] == "127.0.0.1"

    @pytest.mark.parametrize(
        ("argv", "complaint"),
        [
            (["--port", "0"], "must be an integer from 1 to 65535, got 0"),
            (["--port", "70000"], "got 70000"),
            (["--port", "abc"], "got abc"),
            (["--host", ""], "non-empty"),
            (["--host", "  "], "non-empty"),
        ],
    )
    def test_bad_port_and_host_flags_are_usage_errors(self, argv, complaint, capsys):
        with pytest.raises(SystemExit) as exc_info:
            build_parser().parse_args(["app", *argv])

        assert exc_info.value.code == 2
        assert complaint in capsys.readouterr().err

    def test_a_failed_launch_is_one_error(self, calls, monkeypatch, capsys):
        def boom(**kwargs):
            raise OSError("boom\nsecond line")

        monkeypatch.setattr(sys.modules["tokeye.app.__main__"], "main", boom)

        exit_code = main(["app", "--no-browser"])

        assert exit_code == 2
        line = one_error_line(capsys.readouterr().err)
        assert "could not start the app on 127.0.0.1:7860" in line
        assert "boom second line" in line

    def test_another_launch_exception_propagates(self, calls, monkeypatch):
        def broken(**kwargs):
            raise RuntimeError("bug")

        monkeypatch.setattr(sys.modules["tokeye.app.__main__"], "main", broken)
        args = build_parser().parse_args(["app", "--no-browser"])

        with pytest.raises(RuntimeError, match="bug"):
            args.handler(args)


class TestImportErrors:
    @pytest.fixture
    def broken_app_import(self, monkeypatch):
        """gradio is present (a stub) but a module of the app does not import."""
        monkeypatch.setitem(sys.modules, "gradio", types.ModuleType("gradio"))
        monkeypatch.delitem(sys.modules, "tokeye.app.__main__", raising=False)
        monkeypatch.setitem(sys.modules, "tokeye.app.analyze.analyze", None)

    def test_any_other_import_error_is_not_a_missing_extra(
        self, broken_app_import, capsys
    ):
        exit_code = main(["app"])

        assert exit_code == 1
        line = one_error_line(capsys.readouterr().err)
        assert "unexpected ModuleNotFoundError" in line
        assert "tokeye.app.analyze.analyze" in line
        assert "pip install" not in line

    def test_the_handler_does_not_swallow_it(self, broken_app_import):
        args = build_parser().parse_args(["app"])

        with pytest.raises(ModuleNotFoundError):
            args.handler(args)

    def test_a_missing_gradio_submodule_is_a_missing_extra(self, monkeypatch, capsys):
        monkeypatch.delitem(sys.modules, "tokeye.app.__main__", raising=False)
        monkeypatch.setitem(sys.modules, "gradio", None)

        exit_code = main(["app"])

        assert exit_code == 2
        assert 'pip install "tokeye[app]"' in capsys.readouterr().err

    def test_python_m_tokeye_app_without_gradio(self):
        code = (
            "import runpy, sys; sys.modules['gradio'] = None; "
            "sys.argv = ['tokeye.app', '--no-browser']; "
            "runpy.run_module('tokeye.app', run_name='__main__', alter_sys=True)"
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

        assert result.returncode == 2, result.stderr
        assert 'pip install "tokeye[app]"' in result.stderr
        assert "Traceback" not in result.stderr
