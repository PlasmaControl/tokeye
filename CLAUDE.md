# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TokEye segments *fluctuating signals* in spectrograms (coherent + transient
activity) with a Transformer U-Net. Built for DIII-D tokamak diagnostics but
domain-agnostic (also validated on bioacoustics). Ships as a Python API, a CLI,
a Gradio web app, and a mode-analysis suite. Package version and the released
model version are independent (`pyproject.toml` vs the registry filename).

## Commands

`uv` is the dev tool; prefix commands with `uv run` (or activate `.venv`).

```bash
uv sync --dev                      # dev tools: ruff, pytest, ty, pre-commit
uv sync --extra app                # + gradio (needed for tokeye.app tests)
uv sync --group train              # + training deps (lightning, h5py, ...)
uv run pre-commit install          # enable the 10 MB large-file guard (do once)

uv run ruff check .                # lint (CI gate #1)
uv run pytest                      # full suite (CI gate #2)
uv run pytest tests/test_api.py    # one file
uv run pytest -k "hub and not upload"   # by expression
uv run pytest -n auto              # parallel (pytest-xdist is installed)
```

CI (`.github/workflows/python-package.yml`) runs ruff + pytest on Python 3.13
and 3.14 with `uv sync --dev --extra app`. **`--extra app` is required** —
without gradio, `test_app_load.py` and the `tokeye app` CLI test fail with
`ModuleNotFoundError`. Python floor is 3.13.

## Architecture

### Lazy-import discipline (load-bearing invariant)

`import tokeye` and `tokeye --help` must stay free of torch and gradio.
Enforced by `tests/test_import_hygiene.py` — don't break it:

- `tokeye/__init__.py` resolves `TokEye` via PEP 562 `__getattr__` (no eager
  torch import).
- `cli/__init__.py` defers `torch`, `tokeye.batch`, and `tokeye.app` imports
  *into each subcommand handler*, so the CLI dispatch stays instant.
- `hub.py` and `batch.py` never import gradio; `batch.py` forces the matplotlib
  `Agg` backend before importing pyplot (headless HPC/CI safety).

When adding code to the core package or CLI, keep heavy imports local to the
function that needs them.

### Core inference stack (`src/tokeye/`)

`api.py` → `inference.py` + `transforms.py` + `hub.py`, with `batch.py` as the
headless multi-file driver.

- **`api.py`** — the one-import public surface: `from tokeye import TokEye`.
  `TokEye()` is callable; input is auto-detected by shape (1D = raw time series
  → STFT; 2D = ready spectrogram). Returns a `(2, H, W)` float mask.
- **Preprocessing is split across two files** — know both when touching either:
  - `transforms.py::compute_stft` does the STFT, `log1p`, DC-bin drop, and
    percentile clip. Defaults: `n_fft=1024`, `hop=256`. **The released model
    was trained with `hop=128`** — the 256 default is a UI convenience, so use
    `--hop 128` for closest match.
  - `inference.py::model_infer` does the **per-input mean/std standardization**
    (`(x - mean) / (std + 1e-6)`) and the sigmoid — *not* transforms.py. The
    model expects standardized, log-scaled input.
  - `log=` (off by default) applies `log1p` only to **2D linear-scale** inputs;
    it's ignored for 1D (the STFT already log-scales). `log_scale` rejects
    negative input as a guard against double-scaling.
- **Output channel semantics**: channel 0 = coherent activity (the useful one
  for most tasks), channel 1 = transient. Both sigmoid scores in `[0, 1]`.

### Model registry + hub (`hub.py`, `models/`)

`MODEL_REGISTRY` maps a name → `ModelSpec(name, filename, builder, repo_id)`.
Weights auto-download from Hugging Face on first use (`~/.cache/huggingface`);
`TOKEYE_HF_REPO` overrides the default repo. Registered models today:
`big_tf_unet` (default, segmentation) and `ae_tf_maskrcnn` (AE instance
detector, used by `alfvenspec`).

Loading is **state-dict sniffing**, and this has ordering constraints:

- A registry name → `_load_from_registry` (builds the known architecture).
- A local `.pt` holding a bare state dict → `_build_from_state_dict`, which
  tries each registry spec **in insertion order** and keeps the first that
  loads `strict=True`. Keep the fast U-Net spec first so U-Net checkpoints
  never construct the slow R-CNN builder.
- A local `.pt` that fails `weights_only=True` → warns and unpickles the full
  module (legacy, local-only).
- A `.pt2` → `torch.export.load`.

Each model lives in `models/<name>/` as a `config_<name>.py` + `model_<name>.py`
pair; add a `ModelSpec` to register a new one.

### CLI (`cli/`)

Argparse, one module per subcommand, each exposing `add_subcommand(subparsers)`
and wiring a `handler`. `cli/__init__.py::build_parser` registers them. Entry
point: `tokeye = "tokeye.cli:main"`. Subcommands: `app`, `run`, `download`,
`example`, `modespec`, `elmspec`, `alfvenspec`, `eigspec`, `modesearch`.

`tokeye run` writes `<stem>_mask.npy` (float32 `(2,H,W)`) + `<stem>_preview.png`
per input; its exit code is the count of failed files.

### Mode-analysis suite

Beyond segmentation, several subcommands wrap analyses DIII-D researchers
otherwise reach for separate tools to get. Two are **vendored ports** — treat
them as third-party: don't reformat, and see their `PROVENANCE.md`:

- `modespec/classic/` — pymodespec (IDL `modespec` port). Data fetch needs
  MDSplus or a local cache.
- `eigspec/` — MATLAB-toolbox port (MIT), with local SSI numeric fixes noted
  in `docs/ROADMAP.md` as worth upstreaming.

`elmspec` and `alfvenspec` are first-party thin wrappers over the models.
`docs/ROADMAP.md` is the suite roadmap and the "mode catalogue schema" north star.

### App (`app/`)

Gradio, three tabs (Analyze / Annotate / Utilities). `tokeye app` or
`python -m tokeye.app`. On a remote node, SSH-forward the port rather than
`--share`.

### Training pipelines (`training/`)

**Not** part of the installed inference path — research code behind the `train`
extra. Pipelines are **numbered-step scripts** (`step_<N><letter>_<name>.py`,
run in order). Three generations exist:

- `big_tf_unet/` — the original multiscale teacher pipeline (steps 0–7).
- `big_tf_unet_ablation/` — dual-mask ablation variant.
- `big_tf_unet_2/` — current single-scale-teacher, notebook-driven pipeline.
  You edit **one `run.yaml` of knobs** (many default to `auto`), never the code;
  `scaffold.py` generates a per-scale run dir. Entry:
  `uv run python -m tokeye.training.big_tf_unet_2.scaffold --nfft 512 --hop 128`.

All pipeline **outputs land in gitignored dirs** (`dev/training/`, `model/`,
`output/`, `logs/`) — never under `src/`.

## Repo conventions & gotchas

- **Never commit large files.** A pre-commit hook + a CI `large-files` job both
  reject any tracked file over **10 MB** (history was once bloated to ~67 GB by
  auto-committed data). `dev/`, `model/`, `output/`, `logs/`, `.tmp/`, and
  `data/` are gitignored; weights/data/demos belong on Hugging Face or a
  Release, not in git. If you use an auto-commit editor extension, ensure it
  honors `.gitignore`.
- **`dev/` is gitignored and holds real work** — paper figures/eval/tests
  (`dev/paper/`), training run outputs (`dev/training/`), collaborator scratch.
  It's not throwaway; it's just kept out of the package/history.
- **Dependency extras keep the core lean**: `app` (gradio), `train` (lightning,
  h5py, ...), `eigspec` (scikit-learn). A plain install pulls the **CUDA torch
  wheel (~2.5 GB)** on Linux; on a CPU-only box add `--torch-backend=cpu`.
- **ruff** (line-length 88, config in `ruff.toml`) exempts the vendored
  `modespec/classic/` and `eigspec/` trees from nearly all rules and relaxes
  `scripts/**` — respect those exemptions, don't "clean up" vendored code.
- HPC compute nodes often lack internet: `tokeye download <model>` on the login
  node first, then run the job on the compute node against the cached weights.
