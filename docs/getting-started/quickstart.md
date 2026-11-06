# Quickstart

```bash
pip install "tokeye[app]"   # web app + CLI   (or: uv tool install "tokeye[app]")
tokeye app                  # opens web app on http://localhost:7860
```

In a uv project, use `uv add "tokeye[app]"` and `uv run tokeye app`. Keep the quotes (zsh, the macOS default, treats `[app]` as a glob), and note that a plain `tokeye` install has no gradio. If `uv tool install tokeye` was run first, add `--force` to switch it to `"tokeye[app]"`.

- The default model downloads automatically from Hugging Face on first use (~30 MB).
- No data on hand? Click "Load Example Signal" in the app, or generate one from the shell with `tokeye example`.
- Runs on Linux, macOS (Apple silicon) and Windows with Python >= 3.11; a laptop CPU is enough. `uvx`/`uv tool install` fetch a compatible Python automatically.
- Something off? `tokeye info` shows the versions, the device TokEye will use, and which models are cached; it exits 1 if a check fails.

Zero-install trial: `uvx "tokeye[app]" app` runs the app without installing anything into your environment (on Linux without a GPU, add `--torch-backend=cpu`; see below).



## Python

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



## Command line

```bash
tokeye example
tokeye run tokeye_example_sr200000.npy --format npz --output-dir results
```

The model weights download on first use. The output directory contains a bundle, preview, and provenance file.
