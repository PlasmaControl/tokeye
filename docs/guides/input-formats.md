# Input formats

Keep signals as 1D float arrays (raw time series) in any supported format (see [Batch processing](cli.md)). No need to normalize or preprocess them. The CLI also accepts 2D arrays (precomputed spectrograms) directly. The app scans a signal directory for `.npy` files (default `data/input`, configurable in the Analyze tab).

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

For headless / scripted use (no browser needed), run inference directly (`tokeye`, or `python -m tokeye` when the script is not on your `PATH`). For example:

```bash
tokeye run "files/*.npy" --output-dir results
```

`INPUT` arguments can be files (`.npy`, `.npz`, `.wav`, `.flac`, `.ogg`, `.csv`, `.txt`, `.mat`, `.h5`), directories (every supported file inside except `.csv`/`.txt`), or quoted glob patterns. Glob patterns match supported files only. Outputs are named after each input's file stem, so inputs that share a stem, ignoring case (`shot.npy` and `shot.wav`, or `Shot.npy` and `shot.npy`), are rejected; run them into separate `--output-dir`s. The sampling rate is read from the file when it records one (or from a `_sr<fs>` filename suffix); `--fs` sets it explicitly. In `.npz`, `.mat` and `.h5`/`.hdf5` files, TokEye reads the array named `data`, `signal`, `x`, `spectrogram`, `values` or `y`, or else the only numeric array. When several qualify, it stops and lists them; `--key NAME` picks one. A suffix of whole HDF5 path components, such as `tree/pointname/data`, selects the same array in every file. Quote the key in the shell when it contains a backslash. A 2-column `.csv`/`.txt` is read as (time, value), with time in seconds unless its header says `ms` or `us`. Each input is interpreted by its shape:
- **1D array** — a raw time series. TokEye computes its STFT spectrogram using the flags below before running inference. Frames are centred on samples 0, hop, 2·hop, … (the model's training framing), so N samples give `1 + N // hop` columns (for the default, even, `n_fft`) and column j is at `j·hop/fs` s.
- **2D array** — a precomputed spectrogram, fed to the model directly. Its columns are assumed to follow the same rule.
