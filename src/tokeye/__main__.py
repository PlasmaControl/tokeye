"""``python -m tokeye``: the same as the ``tokeye`` command."""

from __future__ import annotations

if __name__ == "__main__":
    from tokeye.cli import main

    raise SystemExit(main())
