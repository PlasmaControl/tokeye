# Installation

TokEye requires Python 3.11 or newer. Install the core package for the Python API and CLI, or include the app extra for the web interface:

```bash
pip install tokeye
pip install "tokeye[app]"
```



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
