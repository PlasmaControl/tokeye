"""Keep the published command reference in sync with the CLI."""

import importlib.util
from pathlib import Path


def test_cli_reference_is_current(monkeypatch):
    monkeypatch.setenv("COLUMNS", "60")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "generate_cli_docs", root / "scripts/generate_cli_docs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # argparse reflows help according to terminal width and Python version.
    assert (root / "docs/reference/cli.md").read_text(
        encoding="utf-8"
    ).split() == module.render().split()
