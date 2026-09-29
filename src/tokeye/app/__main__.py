"""
TokEye Main Inference
"""

from __future__ import annotations

import importlib.resources
import logging
import sys
from pathlib import Path

import gradio as gr

# Import tabs
from .analyze.analyze import analyze_tab
from .tabs.annotate import annotate_tab
from .tabs.utilities import utilities_tab
from .utils.theme import CUSTOM_CSS, make_theme

# Constants
APP_TITLE = "TokEye"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
MAX_PORT_ATTEMPTS = 10

# Set up logging
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


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
    return app


def main(
    port: int = DEFAULT_PORT,
    share: bool = False,
    open_browser: bool = False,
    host: str = DEFAULT_HOST,
) -> None:
    """Build the app and serve it on ``host``, trying ports upward from ``port``."""
    logger.info(f"Initializing TokEye in: {Path.cwd()}")
    app = create_app()
    for attempt in range(MAX_PORT_ATTEMPTS):
        try:
            app.launch(
                server_name=host,
                share=share,
                inbrowser=open_browser,
                server_port=port + attempt,
            )
            return
        except OSError:
            logger.warning(
                "Port %d in use, trying %d", port + attempt, port + attempt + 1
            )
    raise SystemExit(f"No free port in {port}-{port + MAX_PORT_ATTEMPTS - 1}")


if __name__ == "__main__":
    # `python -m tokeye.app [flags]` == `tokeye app [flags]`
    from tokeye import cli

    sys.exit(cli.main(["app", *sys.argv[1:]]))
