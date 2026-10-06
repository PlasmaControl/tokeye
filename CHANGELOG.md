# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0.0 on.

## [Unreleased]

The first stable release (1.0.0): laptop-friendly inference with the default
`big_tf_unet` model on Linux, macOS and Windows. Pre-1.0 spellings keep working
and warn (a `DeprecationWarning`, or a `warning:` line on stderr for the
`--keep-dc` flag); they are removed in 2.0. The default model is unchanged,
1D inputs are now framed exactly as in its training (see Changed), and a
golden test now pins its output.

### Added

- `TokEye.segment()` returns a `Segmentation`: the mask plus the spectrogram,
  frequency/time axes (in Hz and seconds when the sampling rate `fs` is known),
  channel access by name (`seg["coherent"]`), `save()`/`load()` as a
  `tokeye-analysis/v1` `.npz` bundle, and `plot()`. It records the weights
  used (`weights`: repo, file name, revision and sha256 for registry models;
  file name and sha256 for local checkpoints) and names a local checkpoint by
  its file name only.
- `tokeye.SpectrogramConfig`, the single, validated source of preprocessing
  defaults (`n_fft=1024`, `hop=128`, ...). `TokEye(config=...)` and
  `run_batch(config=...)` accept it; the CLI builds its flags from it.
- Input formats: `.npy`, `.npz`, `.wav`, `.flac`, `.ogg`, `.csv`, `.txt`,
  `.mat` and `.h5`, with the sampling rate read from the file (or a `_sr<fs>`
  filename suffix) when recorded.
- Tiled inference: long inputs run in overlapping tiles, bounding peak memory
  (`TokEye(tile=...)`, `--tile auto|none|N` for `tokeye run` and
  `tokeye elmspec`); inputs up to 2^21 pixels run untiled. For `big_tf_unet`
  and any `BigTFUNetModel` with `num_layers <= 5`, tiled output matches an
  untiled run to float32 rounding, because each tile's upsampling uses the
  whole image's grid; other models may differ slightly, with one warning
  when TokEye can tell. `<stem>_params.json` records `tile` and `tile_shape`.
- Apple-silicon GPUs: `device="auto"` tries CUDA, then MPS, then CPU, and an
  op MPS cannot run falls back to CPU with one warning.
- `tokeye info` reports versions, the device `auto` picks, installed extras
  and cached models, and exits 1 if a check fails.
- `tokeye run --format npz` writes the `Segmentation` bundle; every run also
  writes `<stem>_params.json` recording the model, version, device, input, `fs`
  and preprocessing settings, and the weights used (`weights`: repo, file
  name, revision and sha256 for registry models; file name and sha256 for
  local checkpoints). A local checkpoint is named by its file name only.
  `--fs` sets the sampling rate.
- `tokeye elmspec --dt` sets the column spacing, and each input keeps its own
  timebase.
- `tokeye alfvenspec` writes `<stem>_ae_instances.npy` — an `(H, W)` label map
  where `i + 1` marks detection `i` — for windowed inputs too.
- `tokeye app --host/--workspace/--no-browser`; the app binds to `127.0.0.1`,
  or `$GRADIO_SERVER_NAME` when set (with a `note:` line when that address is
  reachable from other machines). A bad `--host`, `--port` or `--workspace`
  is a one-line error.
- `tokeye.export`: the `.npz` export helpers shared by the app and
  `Segmentation.save()`.
- Extras `ae` (torchvision, for `ae_tf_maskrcnn`), `hdf5` (h5py) and `all`;
  `app` now also installs soundfile. `tokeye.__version__` and `py.typed`.
- `-v`/`--verbose` on every subcommand (before or after it) shows debug logs,
  including the tracebacks of per-input failures.
- `tokeye.batch.process_files`: the per-file loop of `run_batch`, for callers
  that already hold a loaded model.
- `key=` (in `load_signal`, `load_spectrogram`, `process_file`,
  `process_files` and `run_batch`) and `--key` (in `run`, `elmspec` and
  `alfvenspec`) select the array in `.npz`/`.mat`/`.h5`/`.hdf5` inputs. The
  key is an entry name, or an HDF5 dataset path or a suffix of whole path
  components; `<stem>_params.json` records it.

### Changed

- The default STFT hop is 128 samples (was 256), the released model's training
  recipe. A 1D input now gives about twice as many time columns; pass
  `hop=256` (`--hop 256`) for 0.12.0's column spacing. The frames themselves
  changed too (next bullet), so values are close to 0.12.0's but not
  identical.
- 1D inputs are framed as in the model's training: frames are centred on
  samples 0, hop, 2·hop, …, with the signal reflected at both ends
  (`torch.stft(center=True)`). N samples give `1 + N // hop` columns (for the
  default, even, `n_fft`), and column j is at `j·hop/fs` s. 0.12.0 also
  computed zero-padded partial frames hanging over both ends (1566 columns
  instead of 1563 for 400,000 samples at its defaults). Mask values change
  slightly everywhere, because the input's global standardization no longer
  includes the near-empty edge frames (mean |Δ| under 1e-3 on the example);
  the largest changes are in the first few columns. The app and the pre-1.0
  `signal_to_spectrogram` use the same framing.
- Python 3.11 or newer (was 3.13).
- Core dependencies are only numpy, scipy, matplotlib, torch, huggingface-hub
  and tqdm, with tested minimum versions. torchvision moved to the `ae` extra
  (loading `ae_tf_maskrcnn` without it says so); torchinfo, omegaconf,
  pydantic and pyyaml are no longer installed.
- `<stem>_mask.npy` is always `(C, H, W)`, also for a one-channel model
  (0.12.0 wrote `(H, W)`).
- CLI exit codes: 0 = every input succeeded, 1 = at least one failed,
  2 = usage or configuration error, 130 = interrupted (was the failure count).
- `tokeye --version` prints `tokeye <version>`.
- CLI boolean flags come in pairs: `--png/--no-png`, `--clip-dc/--no-clip-dc`,
  `--log/--no-log`, `--masks/--no-masks`. `tokeye elmspec` writes previews with
  `--png` as before.
- `tokeye example` names its output `tokeye_example_sr<fs>.npy`, so the rate
  travels with the file.
- A directory input (`tokeye run`, `elmspec`, `alfvenspec`, `run_batch`)
  takes every supported file in it except `.csv` and `.txt`, which you pass
  by name (was `*.npy` only).
- Passing an instance model (`ae_tf_maskrcnn`) where a segmentation model is
  needed (or the reverse) fails early with a pointer to the right command.
- Standardization uses float64 statistics over the whole input, then float32.
- An unknown or unavailable device (`--device`, or `device=` in Python) is a
  `ValueError` that lists the accepted forms (`cpu`, `cuda`, `cuda:N`, `mps`,
  `auto`) or points to `tokeye info`; in the CLI it is one line and exit 2.
- An empty `window` is a `ValueError` (was a `TypeError`), and `--window` is
  checked before the model loads, even when every input is 2D.
- Numeric flags are range-checked: `--threshold`, `--score-min` and
  `--activity-min` in [0, 1], `--window-cols` 0 or ≥ 32, a finite `--mean`,
  `--min-gap-cols` ≥ 0 and `--min-duration-cols` ≥ 1.
- `detect_windowed(window_cols=…)` rejects 1–31; `run_batch` rejects
  colliding stems and a non-finite or non-positive `fs` before loading the
  model.
- `tokeye.elmspec.summarize` and `write_events_csv` raise `ValueError` when
  `fs` or `hop` is zero, negative or not finite. 0.12.0 treated `fs=0` as an
  unknown rate (no ELM frequency, blank times) and `hop=0` as zero-length
  columns.
- A container with several candidate arrays now raises and lists them.
  Before, it silently took the first `data`/`signal`/`x`/… name: for example,
  `np.savez(f, x=t, y=v)` returned the time vector, and an HDF5 file with
  many datasets named `data` returned an arbitrary one.
- `tokeye app` opens a browser automatically in a local (non-SSH) session
  (0.12.0 opened one only with `--open`); `--no-browser` turns it off.
- The app has a dark theme, a one-click Analyze flow with `.npz` export, mask
  download from Annotate, and moves to the next free port when 7860 is busy
  and says which.
- `tokeye.examples.write_example_signal` names its file
  `tokeye_example_sr200000.npy` by default (was `tokeye_example.npy`), so
  `tokeye run` reads the rate from the name.

### Deprecated

- `run_batch(stft_kwargs=..., log=...)`: use `config=SpectrogramConfig(...)`.
  An `fs` key in `stft_kwargs` moves to `fs=`.
- Passing a `dict` of STFT settings to `process_file` (or `process_files`),
  including 0.12.0's `process_file(stft_kwargs=...)`: use
  `config=SpectrogramConfig(...)`. An `fs` key in the dict moves to `fs=`.
- `--keep-dc`: use `--no-clip-dc`.
- Integers `0`/`1` for `clip_dc` and `log`: pass `True`/`False`.
- `tokeye alfvenspec`'s `<stem>_ae_masks.npy` (per-detection soft masks,
  written only for unwindowed inputs): use `<stem>_ae_instances.npy`.

### Removed

- The `modesearch` placeholder subcommand and package.
- The vendored `modespec` and `eigspec` modules, their subcommands and the
  `eigspec` extra.
- The abandoned `big_tf_unet_2` training pipeline (tagged `big-tf-unet-2-final`),
  the unused `ae_tf_boxrcnn` model, `tokeye.extra`, and other dead code.
- The `train` extra: training runs from a clone with `uv sync --group train`,
  and `tokeye.training` is no longer shipped in the wheel or sdist.
- `tokeye.export.modes_csv_text`, with modespec (it was on `main` after
  0.12.0 but in no release).

### Fixed

- Install hints use double quotes (`pip install "tokeye[app]"`), so they work
  in Windows `cmd.exe` too.
- Model-load failures (offline without cached weights, unreadable or truncated
  checkpoints, hub errors) are one `error:` line with exit 2, never a
  traceback.
- The Hugging Face message appears only for hub errors; an unreachable hub
  without cached weights says so and points to `tokeye download`.
- Per-input errors name the exception type.
- Inputs whose outputs would overwrite each other (the same file stem,
  ignoring case) are rejected by `tokeye run`, `elmspec --png`, `alfvenspec`
  (unless `--no-masks`) and `run_batch` before the model loads, and the error
  names them.
- Glob patterns match only supported files.
- Unreadable text tables and empty or non-numeric arrays get an error naming
  the file.
- 2D arrays from MATLAB v7.3 `.mat` files keep MATLAB's orientation; they
  used to come back transposed. The missing-h5py hint names MATLAB v7.3.
- `fs` fields match case-insensitively (`Fs`).
- `.npz` inputs read only the selected entry, and an object-array entry no
  longer makes the file unreadable.
- Empty HDF5 datasets no longer crash the loader.
- A 2-column table without a strictly increasing first column raises,
  instead of becoming a 2-column spectrogram, and a time column headed `ms`
  or `us` is converted to seconds.
- `tokeye elmspec` on a 1D input: event times were one hop late with 0.12.0's
  defaults (the first, partial frame was labelled 0 s; 1.28 ms at 200 kHz),
  and the ELM frequency divided by the spectrogram's width instead of the
  signal's duration.

## [0.12.0] - 2026-07-08

Mode-analysis suite: `tokeye elmspec`, `tokeye alfvenspec` (with the
`ae_tf_maskrcnn` model), the vendored `modespec` and `eigspec`, and a
`modesearch` placeholder. gradio moved to the `app` extra.

## [0.11.0] - 2026-07-04

The `TokEye` Python API class (`from tokeye import TokEye`) and its `log`
option for linear-scale spectrograms.

## [0.10.0] - 2026-07-04

The `tokeye` CLI with headless batch inference, weights downloaded from Hugging
Face on first use, and guided onboarding in the app.

[Unreleased]: https://github.com/PlasmaControl/tokeye/compare/v0.12.0...HEAD
[0.12.0]: https://github.com/PlasmaControl/tokeye/releases/tag/v0.12.0
[0.11.0]: https://github.com/PlasmaControl/tokeye/releases/tag/v0.11.0
[0.10.0]: https://github.com/PlasmaControl/tokeye/releases/tag/v0.10.0
