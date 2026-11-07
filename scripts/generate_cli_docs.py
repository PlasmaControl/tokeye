"""Write the CLI reference from the actual argument parser."""

import argparse
import os
from pathlib import Path
from unittest.mock import patch

from tokeye.cli import build_parser


def render() -> str:
    # argparse otherwise changes wrapping to match the caller's terminal.
    with patch.dict(os.environ, {"COLUMNS": "80"}):
        return _render()


def _render() -> str:
    parser = build_parser()
    sections = ["# Command-line reference\n", "Generated from the TokEye argument parser.\n"]
    sections.append(f"## `tokeye`\n\n```text\n{parser.format_help().rstrip()}\n```\n")
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, subparser in action.choices.items():
                sections.append(
                    f"## `tokeye {name}`\n\n```text\n"
                    f"{subparser.format_help().rstrip()}\n```\n"
                )
    return "\n".join(sections)


if __name__ == "__main__":
    Path("docs/reference/cli.md").write_text(render(), encoding="utf-8")
