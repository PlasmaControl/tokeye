"""
TokEye Main Inference
"""

from __future__ import annotations

import argparse
import importlib.resources
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import gradio as gr

# Import tabs
from .analyze.analyze import analyze_tab
from .tabs.annotate import annotate_tab
from .tabs.diiid import diiid_tab
from .tabs.diiid_modespec import diiid_modespec_tab
from .tabs.diiid_offline import diiid_offline_tab
from .tabs.utilities import utilities_tab
from .utils.theme import CUSTOM_CSS, make_theme

# Constants
APP_TITLE = "TokEye"
DEFAULT_PORT = 7860
MAX_PORT_ATTEMPTS = 10

# Set up logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# --- Non-blocking latest-shot prefill --------------------------------------
# The latest-shot prefill runs on every page load. The underlying MDS thin
# client (atlas.gat.com) can hang off-cluster, so the fetch is pushed onto a
# single background thread, bounded by a timeout, and memoised with a short TTL.
# The constants are read at call time so tests can monkeypatch them.
_SHOT_TIMEOUT_S = 5.0
_SHOT_TTL_S = 60.0

# Cache guarded by _SHOT_LOCK: ``value`` = last good shot (0 = never), ``ts`` =
# time.monotonic() of that fetch (None = never fetched). ``_SHOT_PENDING`` is
# the in-flight fetch (at most one, never stacked); ``_SHOT_EXECUTOR`` is the
# lazily created single-worker daemon pool that runs it.
_SHOT_LOCK = threading.Lock()
_SHOT_CACHE: dict[str, float | None] = {"value": 0, "ts": None}
_SHOT_EXECUTOR: ThreadPoolExecutor | None = None
_SHOT_PENDING: Future | None = None


def _fetch_latest_shot() -> int | None:
    """Worker body: the lazy MDS import + call, run off the request thread.

    The import stays inside the worker so the module never pulls in MDSplus at
    import time (import-hygiene) and the network call never touches the thread
    serving the page load.
    """
    from tokeye.sources import latest_shot

    return latest_shot()


def _store_shot_locked(shot: int) -> None:
    """Record a good fetch in the cache (caller must hold ``_SHOT_LOCK``)."""
    _SHOT_CACHE["value"] = int(shot)
    _SHOT_CACHE["ts"] = time.monotonic()


def _submit_or_reuse_locked() -> tuple[Future, bool]:
    """Return the in-flight fetch, submitting a new one only if none is pending.

    Caller must hold ``_SHOT_LOCK``. The bool is True only for the caller that
    actually submitted, so exactly one done-callback gets attached per fetch and
    a hung atlas call is never stacked behind a second submit.
    """
    global _SHOT_EXECUTOR, _SHOT_PENDING
    if _SHOT_PENDING is not None:
        return _SHOT_PENDING, False
    if _SHOT_EXECUTOR is None:
        _SHOT_EXECUTOR = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="latest-shot"
        )
    future = _SHOT_EXECUTOR.submit(_fetch_latest_shot)
    _SHOT_PENDING = future
    return future, True


def _on_shot_fetched(future: Future) -> None:
    """Fold a completed fetch into the cache and free the pending slot.

    Runs on the worker thread once the fetch finishes. Takes ``_SHOT_LOCK`` so
    a late result folded here can never race a concurrent ``_latest_shot()``
    reading or updating the cache. This is what lets a fetch that outran its
    timeout still populate the cache for the next page load.
    """
    global _SHOT_PENDING
    try:
        shot = future.result()
    except Exception:  # noqa: BLE001 - a failed fetch leaves the cache as-is
        shot = None
    with _SHOT_LOCK:
        if shot is not None:
            _store_shot_locked(shot)
        if _SHOT_PENDING is future:
            _SHOT_PENDING = None


def _latest_shot() -> int:
    """Latest DIII-D shot for the DIII-D tab defaults, resolved without blocking.

    Runs on every page load. Returns the cached shot immediately while it is
    younger than ``_SHOT_TTL_S``; otherwise it kicks off (or reuses) a single
    background fetch and waits at most ``_SHOT_TIMEOUT_S`` for it. A ``None``
    result, a timeout, or any error falls back to the last known shot (``0`` if
    never fetched) so the page never stalls. A fetch that outruns the timeout is
    left in flight and its result is folded into the cache by the done-callback,
    so the next load gets the real shot instantly.
    """
    timeout = _SHOT_TIMEOUT_S
    ttl = _SHOT_TTL_S

    with _SHOT_LOCK:
        ts = _SHOT_CACHE["ts"]
        if ts is not None and time.monotonic() - ts < ttl:
            return int(_SHOT_CACHE["value"] or 0)
        future, submitted = _submit_or_reuse_locked()
        last_known = int(_SHOT_CACHE["value"] or 0)

    # Attach the done-callback outside the lock: if the fetch already finished
    # add_done_callback fires it inline, and it would deadlock on _SHOT_LOCK.
    if submitted:
        future.add_done_callback(_on_shot_fetched)

    try:
        shot = future.result(timeout=timeout)
    except Exception:  # noqa: BLE001 - timeout or fetch error -> last known
        return last_known
    if shot is None:
        return last_known

    with _SHOT_LOCK:
        _store_shot_locked(shot)
    return int(shot)


def _latest_shot_pair() -> tuple[int, int]:
    """Latest shot for both DIII-D shot fields — one fetch, returned twice."""
    shot = _latest_shot()
    return shot, shot


def create_app() -> gr.Blocks:
    with gr.Blocks(
        title=APP_TITLE,
        theme=make_theme(),
        css=CUSTOM_CSS,
    ) as app:
        logo_path = importlib.resources.files("tokeye.app").joinpath("assets/logo.png")
        if logo_path.is_file():
            gr.Image(
                str(logo_path),
                height=300,
                interactive=False,
                container=False,
                show_download_button=False,
                show_fullscreen_button=False,
                elem_classes=["logo-image"],
            )
        with gr.Tab("Analyze"):
            analyze_tab()
        with gr.Tab("Annotate"):
            annotate_tab()
        with gr.Tab("Utilities"):
            utilities_tab()
        with gr.Tab("DIII-D"):
            diiid_shot = diiid_tab()
        with gr.Tab("DIII-D Modespec"):
            diiid_modespec_shot = diiid_modespec_tab()
        with gr.Tab("DIII-D Offline"):
            diiid_offline_tab()
        # Prefill both DIII-D shot fields with the latest shot on page load: one
        # bounded background fetch (see _latest_shot), not two blocking calls.
        app.load(
            fn=_latest_shot_pair,
            inputs=None,
            outputs=[diiid_shot, diiid_modespec_shot],
        )
    return app


def main(
    port: int = DEFAULT_PORT,
    share: bool = False,
    open_browser: bool = False,
) -> None:
    logger.info(f"Initializing TokEye in: {Path.cwd()}")
    app = create_app()
    for attempt in range(MAX_PORT_ATTEMPTS):
        try:
            app.launch(
                share=share,
                inbrowser=open_browser,
                server_port=port + attempt,
            )
            return
        except OSError:
            logger.warning("Port %d in use, trying %d",
                           port + attempt, port + attempt + 1)
    raise SystemExit(
        f"No free port in {port}-{port + MAX_PORT_ATTEMPTS - 1}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m tokeye.app",
        description="Launch the TokEye Gradio app.",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to serve the app on."
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link."
    )
    parser.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        help="Open the app in a browser on launch.",
    )
    args = parser.parse_args()
    main(port=args.port, share=args.share, open_browser=args.open_browser)
