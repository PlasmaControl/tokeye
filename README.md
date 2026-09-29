<p align="center">
  <img src="src/tokeye/app/assets/logo.png" alt="TokEye Logo" width="400">
</p>

# TokEye

[![test](https://github.com/PlasmaControl/tokeye/actions/workflows/test.yml/badge.svg)](https://github.com/PlasmaControl/tokeye/actions/workflows/test.yml)

TokEye is an open-source Python-based application for automatic classification and localization of fluctuating signals.
It is designed to be used in the context of plasma physics, but can be used for any type of fluctuating signal.

Check out [this preprint](https://arxiv.org/abs/2602.20317) for more information.

## Example Demonstration
<video src="https://github.com/user-attachments/assets/03560db6-1941-483e-b9d7-706c164833f7" autoplay loop muted playsinline controls width="100%"></video>

Expected processing time:
- V100: < 0.5 seconds on any size spectrogram after warmup.
- CPU: ~5-10 seconds.

## Quickstart

```bash
pip install "tokeye[app]"   # web app + CLI   (or: uv tool install "tokeye[app]")
tokeye app                  # opens web app on http://localhost:7860
```

- The default model downloads automatically from Hugging Face on first use (~30 MB).
- No data on hand? Click "Load Example Signal" in the app, or generate one from the shell with `tokeye example`.
- Runs on Linux, macOS (Apple silicon) and Windows with Python >= 3.11; a laptop CPU is enough. `uvx`/`uv tool install` fetch a compatible Python automatically.
- Something off? `tokeye info` shows the versions, the device TokEye will use, and which models are cached.

Zero-install trial: `uvx "tokeye[app]" app` runs the app without installing anything into your environment (on Linux without a GPU, add `--torch-backend=cpu`; see below).

### Install variants

Heavy dependencies are split into extras so you download only what you use:

| Install | What you get |
| --- | --- |
| `pip install tokeye` | Python API + CLI (`tokeye run`, `tokeye elmspec`). Smallest core install. |
| `pip install "tokeye[app]"` | + the Gradio web app (`tokeye app`) and `.flac`/`.ogg` input. |
| `pip install "tokeye[ae]"` | + torchvision, for the `ae_tf_maskrcnn` model (`tokeye alfvenspec`). |
| `pip install "tokeye[hdf5]"` | + `.h5` and MATLAB v7.3 `.mat` input. |
| `pip install "tokeye[all]"` | All of the above. |

Training is research code that runs from a clone (`uv sync --group train`); it is not part of the installed package.

**GPU vs CPU PyTorch.** A plain install pulls the default PyTorch wheel. On macOS and Windows that is already small; on Linux it is the **CUDA (GPU) build (~2.5 GB)**. On a Linux machine without a GPU, install the CPU wheel (~200 MB) instead:

```bash
# pip: install CPU torch first, then tokeye
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "tokeye[app]"

# uv
uv pip install "tokeye[app]" --torch-backend=cpu
uv tool install "tokeye[app]" --torch-backend=cpu
uvx --torch-backend=cpu "tokeye[app]" app
```

## Python API

To use TokEye inside your own program, import the `TokEye` class:

```python
import numpy as np
from tokeye import TokEye

eye = TokEye()  # loads the default model (auto-downloads on first use)

mask = eye(signal)             # 1D time series → STFT → inference
mask = eye(spectrogram)        # 2D spectrogram → inference directly
coherent, transient = mask     # (2, H, W) sigmoid scores in [0, 1]

seg = eye.segment(signal, fs=200_000)   # mask + spectrogram + axes
seg["coherent"], seg.freqs, seg.times   # Hz and seconds when fs is known
seg.save("shot.npz")                    # reload with tokeye.Segmentation.load
```

Input is auto-detected by shape: a 1D array is treated as a raw time series (TokEye computes the spectrogram), a 2D array as a ready spectrogram. Standardization happens internally, and long inputs run in tiles so memory stays bounded. With `big_tf_unet`, tiled output matches an untiled run to float32 rounding; other models may differ slightly, and TokEye warns when it can tell.

If your 2D spectrogram is stored in **linear scale** (raw STFT magnitude/power), pass `log=True` so TokEye applies `log1p` first since the model expects log-scaled input:

```python
mask = eye(linear_spectrogram, log=True)      # per call
eye = TokEye(log=True)                        # or for every call
```

`log` is off by default and ignored for 1D inputs (the STFT already log-scales). Everything is configurable through the constructor, but the defaults just work:

```python
eye = TokEye(
    model="big_tf_unet",   # registry name or path to a local .pt/.pt2
    device="auto",         # "cpu", "cuda", "mps", or "auto" (CUDA, then MPS, then CPU)
    n_fft=1024, hop=128,   # STFT settings (1D inputs only)
    clip_dc=True, clip_low=1.0, clip_high=99.0,
    log=False,             # log1p for linear-scale 2D spectrograms
)

from tokeye import SpectrogramConfig
config = SpectrogramConfig(n_fft=1024, hop=128, clip_dc=True,
                           clip_low=1.0, clip_high=99.0, log=False)
eye = TokEye(config=config)  # the same settings, as one object
```

## Batch processing (CLI)

For headless / scripted use (no browser needed), run inference directly. For example:

```bash
tokeye run "files/*.npy" --output-dir results
```

`INPUT` arguments can be files (`.npy`, `.npz`, `.wav`, `.flac`, `.ogg`, `.csv`, `.txt`, `.mat`, `.h5`), directories (every supported file inside except `.csv`/`.txt`), or quoted glob patterns. The sampling rate is read from the file when it records one (or from a `_sr<fs>` filename suffix); `--fs` sets it explicitly. Each input is interpreted by its shape:
- **1D array** — a raw time series. TokEye computes its STFT spectrogram using the flags below before running inference.
- **2D array** — a precomputed spectrogram, fed to the model directly.

For each input file, `tokeye run` writes:
- `<stem>_mask.npy` — float32 array, shape `(2, H, W)`, sigmoid scores per pixel (channel 0 = coherent, channel 1 = transient).
- `<stem>_preview.png` — a grayscale spectrogram with the mask overlaid (green = coherent, red = transient), unless `--no-png` is passed.
- `<stem>_params.json` — the model, version, device, input, `fs`, preprocessing settings and tiling used (`tile` as requested; `tile_shape` as `[h, w]`, or `null` when the run was untiled).
- With `--format npz`, `<stem>_tokeye.npz` (mask + spectrogram + axes, loadable with `tokeye.Segmentation.load`) replaces the `.npy` mask.

Exit codes: 0 = every input succeeded, 1 = at least one input failed, 2 = usage or configuration error (bad flag, unknown model, no inputs, model cannot be loaded), 130 = interrupted. Errors are one line; add `-v` for details.

Flags:
| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `big_tf_unet` | Registry name or path to a `.pt`/`.pt2` checkpoint. |
| `--output-dir` | `tokeye_output` | Directory for masks and previews. |
| `--format` | `npy` | `npy` (mask only) or `npz` (full bundle). |
| `--fs` | from the file | Sampling rate in Hz for every input. |
| `--n-fft` | `1024` | STFT window size (1D inputs only). |
| `--hop` | `128` | STFT hop size (1D inputs only). |
| `--window` | `hann` | STFT window. |
| `--clip-dc` / `--no-clip-dc` | on | Drop the DC bin. |
| `--clip-low` / `--clip-high` | `1.0` / `99.0` | Percentile clip bounds applied to the spectrogram. |
| `--log` / `--no-log` | off | Apply `log1p` to 2D spectrogram inputs stored in linear scale (1D signals are always log-scaled during the STFT). |
| `--threshold` | `0.5` | Mask threshold used only for the preview PNG overlay. |
| `--png` / `--no-png` | on | Write preview PNGs. |
| `--device` | `auto` | `cpu`, `cuda`, `cuda:N`, `mps`, or `auto` (an unavailable device is an error; see `tokeye info`). |
| `--tile` | `auto` | Tile side for long inputs: `auto` (untiled up to 2^21 pixels), `none`, or an int ≥ 512. For `big_tf_unet`, tiled output matches untiled output to float32 rounding. |

The defaults (`n_fft=1024`, `hop=128`) match the released model's training configuration. A larger hop (e.g. `--hop 256`) halves the columns for faster, lighter runs at some fidelity cost.

On HPC clusters where compute nodes have no internet access, pre-fetch the weights on the login node, then run the batch job on the compute node. Without cached weights, the CLI says so in one line and exits 2.

```bash
tokeye download big_tf_unet   # on the login node; prints the cached path
tokeye run ... --model big_tf_unet   # on the compute node — model is already cached
```

## Mode-analysis suite

Beyond segmentation, `tokeye` bundles the analyses DIII-D researchers usually reach for separate tools to get. Each is a subcommand; `--help` on any of them shows the full flags.

| Command | What it does |
| --- | --- |
| `tokeye elmspec INPUTS...` | ELM detection from the segmentation model's transient channel: per-event time intervals plus per-shot count, ELM frequency (with `--fs` or `--dt`), and duty cycle, written to `elm_events.csv` / `elm_summary.csv`. |
| `tokeye alfvenspec INPUTS...` | Alfvén-eigenmode detection with the `ae_tf_maskrcnn` instance model (needs `tokeye[ae]`): per-detection boxes/scores (`ae_detections.csv`) and a per-input instance map (`<stem>_ae_instances.npy`, `i + 1` = detection `i`). Wide spectrograms are processed in training-width windows automatically. |

## Web app guide

`tokeye app` (or `python -m tokeye.app`) launches a Gradio interface with three tabs:
- **Analyze** — load a signal, compute its spectrogram, run a model, and visualize the result. Guided for first-time use: the model dropdown defaults to the bundled `big_tf_unet` model, the STFT transform has working defaults, and "Load Example Signal" generates a synthetic demo signal so a brand-new user needs zero files. "Analyze" runs the whole load-model → infer → visualize pipeline in one click. View modes: Original, Enhanced (percentile-clipped amplitude), Mask (thresholded model output), Amplitude.
- **Annotate** — manually draw and save mask annotations over a read-only backdrop image.
- **Utilities** — audio-format conversion and `.npy` file inspection.

Flags: `tokeye app [--host 127.0.0.1] [--port 7860] [--open | --no-browser] [--workspace DIR] [--share]`. The app binds to `127.0.0.1` (moving to the next free port if 7860 is busy) and opens a browser tab when you run it locally. `--share` creates a public Gradio link that anyone with the URL can use.

If you're on a remote server (e.g. an HPC login node), forward the port over SSH instead of using `--share`:
```bash
ssh -L 7860:localhost:7860 user@remote
```
Then open `http://localhost:7860` in your local browser.

## Verified Datatypes
- DIII-D Fast Magnetics (cite)
- DIII-D CO2 Interferometer (cite)
- DIII-D Electron Cyclotron Emission (cite)
- DIII-D Beam Emission Spectroscopy (cite)

## Evaluation
Recall Scores:
- TJII2021: 0.8254
- DCLDE2011 (Delphinus capensis): 0.7708
- DCLDE2011 (Delphinus delphis): 0.7953

With more data, comes better models. Please contribute to the project!

## Installation (from source / development)

[uv](https://docs.astral.sh/uv/) is the dev tool for this repo:
```bash
git clone git@github.com:PlasmaControl/TokEye.git
cd TokEye
uv sync --dev --extra all                # dev tools (pytest, ruff, etc.) + every extra (web app, ae model, HDF5)
uv sync --dev --extra all --group train  # the same + training deps (lightning, h5py, etc.)
```

Each `uv sync` makes `.venv` match its flags exactly, removing any extra or group you leave out, so name everything you need in one command. PyTorch comes as the default build (GPU/CUDA on Linux).

Additionally, run these the first time
```bash
uv run pre-commit install
uv run ruff check .
uv run pytest
```

This creates a `.venv/`; activate it with `source .venv/bin/activate`, or prefix commands with `uv run`.

## Models

| Registry name | HF repo | HF file | Description |
| --- | --- | --- | --- |
| `big_tf_unet` | [`nc1/big_tf_unet`](https://huggingface.co/nc1/big_tf_unet) | `big_tf_unet_251210.pt` | Convolutional U-Net trained on multiscale (multiwindow, multihop) spectrograms. |
| `ae_tf_maskrcnn` | `nc1/ae_tf_maskrcnn` | `ae_tf_maskrcnn_251223.pt` | Mask R-CNN instance detector for Alfvén-eigenmode activity (used by `tokeye alfvenspec`). |

Weights download automatically the first time a registry name is used (cached in `~/.cache/huggingface`). Override the default repo with the `TOKEYE_HF_REPO` environment variable (per-model repos are fixed in the registry).

To use a local checkpoint instead, put `.pt`/`.pt2` files in a `model/` directory (picked up by the app's model dropdown) or pass a path directly via `--model PATH`.

Input should be a tensor that has shape (B, 1, H, W) where B, H, and W can vary
Output will be a tensor of shape (B, 2, H, W)

Best performance when spectrograms are oriented so that when they are plotted with matplotlib, the lowest frequency bin is oriented with the bottom when `origin='lower'`. Spectrograms should be standardized (mean = 0, std = 1). If baseline activity is very strong, clipping the input may help, but is generally not needed.

The first channel of the output will return preferential measurements of coherent activity (useful for most tasks)
The second channel of the output will return preferential measurements of transient activity

## Data
Keep signals as 1D float arrays (raw time series) in any supported format (see [Batch processing](#batch-processing-cli)). No need to normalize or preprocess them. The CLI also accepts 2D arrays (precomputed spectrograms) directly. The app scans a signal directory for `.npy` files (default `data/input`, configurable in the Analyze tab).

Bringing your own data takes two lines:

```python
import numpy as np

signal = ...  # any 1D float array: tokamak diagnostic, hydrophone, etc.
np.save("shots/myshot.npy", signal)
```

```bash
tokeye run shots/myshot.npy --output-dir results
```

No data yet? `tokeye example` writes a synthetic demo signal (`tokeye_example_sr200000.npy`) you can run immediately, and the web app has a matching "Load Example Signal" button.

## Citation
If you use this code in your research, please cite:
```bibtex
@article{chen_TokEye_2026,
  title={TokEye: Fast Signal Extraction for Fluctuating Time Series via Offline Self-Supervised Learning From Fusion Diagnostics to Bioacoustics},
  author={Chen, Nathaniel},
  year={2026},
  publisher={ArXiv},
  doi={10.48550/arXiv.2602.20317},
  url={https://www.arxiv.org/abs/2602.20317}
}
```

## Contact
Nathaniel Chen — nathaniel [at] princeton [dot] edu — https://nathanielchen.net
