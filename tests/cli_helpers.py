"""Helpers shared by the CLI test files.

A plain module, not a test file: ``tests/`` has no ``__init__.py``, so test
files import it as ``from cli_helpers import ...`` (like ``golden_utils``).
"""

from __future__ import annotations

import huggingface_hub
from huggingface_hub.errors import RepositoryNotFoundError

_REPO_URL = "https://huggingface.co/does/not/exist"
_FILE_URL = f"{_REPO_URL}/resolve/main/x.pt"
# What huggingface_hub >= 1 raises for a 404 on _FILE_URL (6 lines).
_REPO_NOT_FOUND_TEXT = (
    "404 Client Error.\n"
    "\n"
    f"Repository Not Found for url: {_FILE_URL}.\n"
    "Please make sure you specified the correct `repo_id` and `repo_type`.\n"
    "If you are trying to access a private or gated repo, make sure you are "
    "authenticated and your token has the required permissions.\n"
    "For more details, see https://huggingface.co/docs/huggingface_hub/"
    "authentication"
)


def _hf_major() -> int:
    return int(huggingface_hub.__version__.split(".")[0])


def one_error_line(err: str) -> str:
    """The only non-blank line of ``err``, which must start with ``error: ``."""
    lines = [line for line in err.splitlines() if line.strip()]
    assert len(lines) == 1 and lines[0].startswith("error: "), err
    assert "Traceback" not in err
    return lines[0]


def repository_not_found_error() -> RepositoryNotFoundError:
    """Build a real one-line RepositoryNotFoundError the way huggingface_hub does.

    ``HfHubHTTPError`` needs a ``response``: an ``httpx.Response`` on
    huggingface_hub >= 1.0, a ``requests.Response`` before that (the floor
    of the supported range).
    """
    if _hf_major() >= 1:
        import httpx

        response = httpx.Response(404, request=httpx.Request("GET", _REPO_URL))
    else:
        import requests

        response = requests.Response()
        response.status_code = 404
        response.url = _REPO_URL
    return RepositoryNotFoundError("Repository Not Found", response=response)


def multiline_repository_not_found_error() -> RepositoryNotFoundError:
    """The realistic 6-line RepositoryNotFoundError of a missing repo.

    On huggingface_hub >= 1 it comes from ``hf_raise_for_status``; before
    that it is built from the same text, so no httpx is needed.
    """
    if _hf_major() >= 1:
        import httpx
        from huggingface_hub.utils import hf_raise_for_status

        response = httpx.Response(
            404,
            headers={"X-Error-Code": "RepoNotFound"},
            request=httpx.Request("GET", _FILE_URL),
        )
        try:
            hf_raise_for_status(response)
        except RepositoryNotFoundError as exc:
            return exc
        raise AssertionError("hf_raise_for_status did not raise")

    import requests

    return RepositoryNotFoundError(_REPO_NOT_FOUND_TEXT, response=requests.Response())
