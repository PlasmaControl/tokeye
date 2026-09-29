# Contributing

## Development setup

```bash
uv sync --dev --extra all  # dev tools (ruff, pytest, pre-commit) + every extra
uv run pre-commit install  # enable the git hooks (large-file guard, ruff)
uv run tokeye download     # weights for the golden tests (offline otherwise)
```

## Large files / data — keep them out of git

Raw data, model weights, and caches are **gitignored** (`data/`, `*.h5`, `*.npy`,
`*.pt`, `*.ckpt`, scratch dirs) and must never be committed. The history was once
bloated to ~67 GB by accidentally-committed multi-GB data files, and a 53 MB demo
GIF had to be purged from history.

Two safeguards are in place:

- A **pre-commit hook** (`check-added-large-files`) blocks staging any file over
  **2 MB** locally — run `uv run pre-commit install` so it's active.
- **CI** re-checks every pull request and every push to `main` or a `v*` tag
  (`large-files` job). The build fails if any commit in the range adds a file over
  2 MB, even one that a later commit deletes (it would stay in the history), so
  the guard holds even if someone skips the hook. To check a branch before you
  push it: `bash .github/scripts/large-files.sh origin/main..`

Large assets (demos, datasets, weights) belong in **external hosting** — Hugging
Face, a GitHub Release, or Git LFS — not in the repo. Demo media should be a small
compressed clip (e.g. VP9 `.webm`), not a multi-MB GIF.

If you use an editor "auto-checkpoint" / auto-commit extension, **disable it or make
sure it honors `.gitignore`** — that was the original source of the bloat. (Note:
those orphaned commits stay local; they are unreachable and never get pushed.)

## If you cloned `ablation-pipeline` before June 2026

That branch's history was rewritten to remove the 53 MB GIF, so the commit IDs
changed. If you have an older clone, **reset — do not merge** (a `git pull`/merge
would drag the old file back and re-push it):

```bash
git fetch origin
git checkout ablation-pipeline
git reset --hard origin/ablation-pipeline   # stash/save any local work first
```

Or simply re-clone. `main` was not rewritten, so main-based work is unaffected.

## Lint & test

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pytest -n auto
```

The suite is offline: the hub is mocked, and tests marked `weights` (the golden
numbers) run only when the default weights are cached. CI (`.github/workflows/test.yml`)
also runs the suite on macOS and Windows, a laptop smoke test of the built wheel,
and a job at the oldest supported dependency versions
(`.github/floor-constraints.txt`).
