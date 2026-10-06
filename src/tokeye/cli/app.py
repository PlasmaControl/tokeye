"""``tokeye app`` — launch the Gradio web app."""

from __future__ import annotations

import errno
import os
import socket
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tokeye.cli import _common, _options

if TYPE_CHECKING:
    import argparse

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
PORT_ATTEMPTS = 10

MISSING_EXTRA = (
    "`tokeye app` needs the 'app' extra (gradio), which is not installed.\n"
    "Install it with:\n"
    '    pip install "tokeye[app]"      # or:  uv pip install "tokeye[app]"\n'
    "(underlying import error: {exc})"
)


def add_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "app",
        parents=[_options.VERBOSE],
        help="Launch the TokEye web app in your browser.",
        description=(
            "Serve the app on http://HOST:PORT (the next free port if PORT is "
            "busy). On a remote machine, forward the port over SSH instead of "
            "using --share."
        ),
    )
    parser.add_argument(
        "--host",
        type=_options.nonempty_str,
        # Gradio read this variable before 1.0; a blank one counts as unset
        # (argparse runs type= on a string default).
        default=os.environ.get("GRADIO_SERVER_NAME", "").strip() or DEFAULT_HOST,
        help="interface to bind; 0.0.0.0 exposes the app to your network "
        "(default: $GRADIO_SERVER_NAME if set, else 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=_options.port_number,
        default=DEFAULT_PORT,
        help="first port to try (default: %(default)s)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="also create a public gradio.live link (anyone with it can use the app)",
    )
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument(
        "--open",
        dest="browser",
        action="store_const",
        const=True,
        help="open a browser tab (default when not in an SSH session)",
    )
    browser.add_argument(
        "--no-browser",
        dest="browser",
        action="store_const",
        const=False,
        help="do not open a browser tab",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        metavar="DIR",
        help="directory the app reads and writes files in (created if missing; "
        "default: the current directory)",
    )
    parser.set_defaults(handler=_handle, browser=None)


def _probe(host: str, port: int) -> str:
    """Try to bind ``port`` on every address ``host`` resolves to.

    Returns
    -------
    str
        ``"free"``, ``"busy"`` (some address is in use) or ``"skipped"``
        (``host`` has no address that belongs to this machine).

    Raises
    ------
    ValueError
        ``host`` cannot be resolved, or a bind failed for another reason
        (a privileged or reserved port, say).
    """
    try:
        infos = socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
        )
    except (socket.gaierror, UnicodeError) as exc:
        reason = getattr(exc, "strerror", None) or exc
        raise ValueError(f"--host {host!r}: cannot resolve it ({reason})") from exc

    # localhost can resolve to one address twice; a second bind of the same
    # address fails on Windows and macOS.
    addresses = list(dict.fromkeys((info[0], info[4]) for info in infos))
    socks = []
    skipped = 0
    try:
        for family, sockaddr in addresses:
            try:
                sock = socket.socket(family, socket.SOCK_STREAM)
            except OSError:  # e.g. EAFNOSUPPORT: no IPv6 here
                skipped += 1
                continue
            socks.append(sock)
            try:
                if os.name != "nt":  # on Windows it would bind a held port
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if family == socket.AF_INET6:
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                sock.bind(sockaddr)
            except OSError as exc:
                if exc.errno == errno.EADDRNOTAVAIL:  # asyncio skips these too
                    skipped += 1
                elif exc.errno == errno.EADDRINUSE:
                    return "busy"
                else:
                    raise ValueError(
                        f"cannot listen on {host}:{port} "
                        f"({exc.strerror or exc}); pass another --port"
                    ) from exc
    finally:
        for sock in socks:
            sock.close()
    return "skipped" if skipped == len(addresses) else "free"


def pick_port(host: str, first: int, attempts: int = PORT_ATTEMPTS) -> int:
    """The first port from ``first`` upward that ``host`` can listen on.

    Parameters
    ----------
    host : str
        Interface to bind (a name or an address).
    first : int
        First port to try.
    attempts : int, default ``PORT_ATTEMPTS``
        How many consecutive ports to try (never past 65535).

    Returns
    -------
    int
        A port that was free a moment ago.

    Raises
    ------
    ValueError
        ``host`` cannot be resolved or is not an address of this machine, a
        bind failed for a reason other than a busy port, or no port in the
        range is free. The message is one line, fit for ``error:``.
    """
    last = min(first + attempts - 1, 65535)
    for port in range(first, last + 1):
        state = _probe(host, port)
        if state == "skipped":
            raise ValueError(f"--host {host!r} is not an address of this machine")
        if state == "free":
            return port
    raise ValueError(f"no free port in {first}-{last} on {host}; pass another --port")


def _handle(args: argparse.Namespace) -> int:
    try:
        from tokeye.app.__main__ import main as app_main
    except ModuleNotFoundError as exc:
        # Only a missing gradio is a missing extra; any other import error is
        # a broken install and goes to main()'s last resort (-v: traceback).
        name = exc.name or ""
        if name != "gradio" and not name.startswith("gradio."):
            raise
        print(MISSING_EXTRA.format(exc=exc), file=sys.stderr)
        return _common.EXIT_USAGE

    try:
        port = pick_port(args.host, args.port)
    except ValueError as exc:
        return _common.error(str(exc))

    if args.workspace is not None:
        try:
            args.workspace.mkdir(parents=True, exist_ok=True)
            os.chdir(args.workspace)
        except OSError as exc:
            return _common.error(f"--workspace {args.workspace}: {exc.strerror or exc}")

    remote = "SSH_CONNECTION" in os.environ
    open_browser = args.browser if args.browser is not None else not remote
    if port != args.port:
        print(f"note: port {args.port} is in use; using {port}", file=sys.stderr)
    if remote and not args.share:
        print(
            f"note: SSH session detected; forward the port from your laptop: "
            f"ssh -L {port}:localhost:{port} <this-host>",
            file=sys.stderr,
        )
    if args.share:
        print(
            "warning: --share makes the app reachable by anyone with the link",
            file=sys.stderr,
        )

    try:
        app_main(port=port, share=args.share, open_browser=open_browser, host=args.host)
    except OSError as exc:
        return _common.error(f"could not start the app on {args.host}:{port} ({exc})")
    return _common.EXIT_OK
