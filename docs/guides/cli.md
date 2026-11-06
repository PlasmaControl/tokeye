# Command-line guide

For headless / scripted use (no browser needed), run inference directly (`tokeye`, or `python -m tokeye` when the script is not on your `PATH`). For example:

```bash
tokeye run "files/*.npy" --output-dir results
```

`INPUT` arguments can be files (`.npy`, `.npz`, `.wav`, `.flac`, `.ogg`, `.csv`, `.txt`, `.mat`, `.h5`), directories (every supported file inside except `.csv`/`.txt`), or quoted glob patterns. Glob patterns match supported files only. Outputs are named after each input's file stem, so inputs that share a stem, ignoring case (`shot.npy` and `shot.wav`, or `Shot.npy` and `shot.npy`), are rejected; run them into separate `--output-dir`s. The sampling rate is read from the file when it records one (or from a `_sr<fs>` filename suffix); `--fs` sets it explicitly. In `.npz`, `.mat` and `.h5`/`.hdf5` files, TokEye reads the array named `data`, `signal`, `x`, `spectrogram`, `values` or `y`, or else the only numeric array. When several qualify, it stops and lists them; `--key NAME` picks one. A suffix of whole HDF5 path components, such as `tree/pointname/data`, selects the same array in every file. Quote the key in the shell when it contains a backslash. A 2-column `.csv`/`.txt` is read as (time, value), with time in seconds unless its header says `ms` or `us`. Each input is interpreted by its shape:
- **1D array** — a raw time series. TokEye computes its STFT spectrogram using the flags below before running inference. Frames are centred on samples 0, hop, 2·hop, … (the model's training framing), so N samples give `1 + N // hop` columns (for the default, even, `n_fft`) and column j is at `j·hop/fs` s.
- **2D array** — a precomputed spectrogram, fed to the model directly. Its columns are assumed to follow the same rule.

For each input file, `tokeye run` writes:
- `<stem>_mask.npy` — float32 sigmoid scores per pixel, shape `(C, H, W)`; `(2, H, W)` for the default model (channel 0 = coherent, channel 1 = transient).
- `<stem>_preview.png` — a grayscale spectrogram with the mask overlaid (green = coherent, red = transient), unless `--no-png` is passed.
- `<stem>_params.json` — the model, version, device, input, `key`, `fs`, preprocessing settings and tiling used (`tile` as requested; `tile_shape` as `[h, w]`, or `null` when the run was untiled), and the weights used (`weights`: repo, file name, revision and sha256 for a registry model; file name and sha256 for a local checkpoint, which `model` names by its file name only).
- With `--format npz`, `<stem>_tokeye.npz` (mask + spectrogram + axes, loadable with `tokeye.Segmentation.load`) replaces the `.npy` mask.

The run ends with one line on stdout, such as `wrote 3 inputs -> tokeye_output/ (mask, preview, params)`, and shows a progress bar only on a terminal. Exit codes: 0 = every input succeeded, 1 = at least one input failed, 2 = usage or configuration error (bad flag, unknown model, no inputs, model cannot be loaded), 130 = interrupted. Errors and warnings are one line each; add `-v` for details.

Flags:
| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `big_tf_unet` | Registry name or path to a `.pt`/`.pt2` checkpoint. |
| `--output-dir` | `tokeye_output` | Directory for masks and previews. |
| `--format` | `npy` | `npy` (mask only) or `npz` (full bundle). |
| `--fs` | from the file | Sampling rate in Hz for every input. |
| `--key` | the signal array | Array to read from `.npz`/`.mat`/`.h5` inputs: an entry name, or an HDF5 dataset path or its trailing path components (`tree/pointname/data`). Without it, the one named `data`, `signal`, `x`, `spectrogram`, `values` or `y`, else the only numeric one; several candidates are an error that lists them. |
| `--n-fft` | `1024` | STFT window size (1D inputs only). |
| `--hop` | `128` | STFT hop size (1D inputs only). |
| `--window` | `hann` | STFT window. |
| `--clip-dc` / `--no-clip-dc` | on | Drop the DC bin. |
| `--clip-low` / `--clip-high` | `1.0` / `99.0` | Percentile clip bounds applied to the spectrogram. |
| `--log` / `--no-log` | off | Apply `log1p` to 2D spectrogram inputs stored in linear scale (1D signals are always log-scaled during the STFT). |
| `--threshold` | `0.5` | Mask threshold (0–1) used only for the preview PNG overlay. |
| `--png` / `--no-png` | on | Write preview PNGs. |
| `--device` | `auto` | `cpu`, `cuda`, `cuda:N`, `mps`, or `auto` (an unavailable device is an error; see `tokeye info`). |
| `--tile` | `auto` | Tile side for long inputs: `auto` (untiled up to 2^21 pixels), `none`, or an int ≥ 512. For `big_tf_unet`, tiled output matches untiled output to float32 rounding. |

The defaults (`n_fft=1024`, `hop=128`) match the released model's training configuration. A larger hop (e.g. `--hop 256`) halves the columns for faster, lighter runs at some fidelity cost.

On HPC clusters where compute nodes have no internet access, pre-fetch the weights on the login node, then run the batch job on the compute node. Without cached weights, the CLI says so in one line and exits 2.

```bash
tokeye download big_tf_unet   # on the login node; prints the cached path
tokeye run ... --model big_tf_unet   # on the compute node — model is already cached
```
