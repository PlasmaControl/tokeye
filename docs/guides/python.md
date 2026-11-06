# Python API guide

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

From a file to a labelled plot, `tokeye.io.load_signal` reads the same formats as `tokeye run` (`.h5` needs `tokeye[hdf5]`) and returns the array with the sampling rate the file records (or `None`):

```python
from tokeye import TokEye
from tokeye.io import load_signal

x, fs = load_signal("shot.h5", key="data")  # key only when the file holds several arrays
seg = TokEye().segment(x, fs=fs)
ax = seg.plot()                             # matplotlib Axes: spectrogram + mask, in Hz and s
ax.figure.savefig("shot.png")
```

Input is auto-detected by shape: a 1D array is treated as a raw time series (TokEye computes the spectrogram), a 2D array as a ready spectrogram. `eye(...)` and `eye.predict(...)` return `(C, H, W)`, or `(H, W)` for a single-channel model (as before 1.0); `eye.segment(...).mask` is always `(C, H, W)`. Standardization happens internally, and long inputs run in tiles so memory stays bounded. With `big_tf_unet`, tiled output matches an untiled run to float32 rounding; other models may differ slightly, and TokEye warns when it can tell.

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
