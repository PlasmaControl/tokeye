# TokEye on Omega (DIII-D) — shared CPU install

TokEye finds coherent modes (green) and transient events (red) in spectrograms.
This is the `diiid` build, installed once for everyone in
`/fusion/projects/codes/tokeye` and tuned for Omega's CPU nodes (CPU-only
PyTorch + OpenVINO).

## Use

```bash
module use /fusion/projects/codes/tokeye/modulefiles   # until CSS adds tokeye globally
module load tokeye

tokeye                                    # GUI (NoMachine or ssh -X to somega)
tokeye 190904 MPI66M067D                  # analyze one signal, whole shot
```

`tokeye <shot> <signal>` fetches the signal from MDSplus, brings it to the default
analysis rate, computes the spectrogram with the recommended (training-recipe)
STFT (`n_fft=1024`, `hop=128`), runs the model, and ends with:

```
Saved to /fusion/projects/results/tokeye/190904/
  190904_MPI66M067D.png     spectrogram + mask overlay
  190904_MPI66M067D.npz     full-resolution arrays
  190904_MPI66M067D.json    parameters, provenance, timings
```

### Default sampling rate: ~500 kHz

Signals faster than 500 kHz are decimated (anti-aliased FIR, integer factor,
never below 500 kHz) to about 500 kHz; slower signals stay at their native rate.
500 kHz is the rate the model was trained at (its training data were resampled
to a uniform 500 kHz) and the native rate of current DIII-D Mirnov probes, which
modespec and pyspecview normally run on (neither tool decimates).

| signal | native | default analysis rate |
|---|---|---|
| Mirnov `MPI66M*` (recent shots) / ECE `TECEF*` | 500 kHz | 500 kHz (native) |
| Mirnov `MPI66M*` (older shots) | 200 kHz | 200 kHz (native) |
| BES `BESFU*` | 1 MHz | 500 kHz (×2) |
| CO2 `DENV*` | ~1.67 MHz | ~556 kHz (×3) |
| high-res magnetics `B1`..`B8` | 2 MHz | 500 kHz (×4) |

`fs=<kHz>` picks another target, `fs=native` turns it off, and `decim=N` forces a
factor. The GUI's *Decimation* box defaults to **auto** (the same rule); 1 = native.

### Parameters (modespec style)

Like `modespec,shot,tmin,tmax` / `modespec,shot=...,tmin=...`: positional
`shot [signal] [tmin] [tmax]`, then `key=value` pairs (comma lists OK). The
`--key value` form works too.

```bash
tokeye 190904 MPI66M067D 2000 3000                      # time window [ms]
tokeye 190904 MPI66M067D tmin=2000,tmax=3000 nfft=2048 hop=256
tokeye 190904 B1 fs=native fmax=600                     # keep a 2 MHz probe at 2 MHz
tokeye 190904 TECEF20 out=$HOME/tokeye_runs             # save somewhere else
tokeye 0 mag                                            # latest shot, default Mirnov probe
tokeye shot --help                                      # every parameter
```

| key | meaning | default |
|---|---|---|
| `tmin`, `tmax` (`xmin`, `xmax`) | time window [ms] | whole shot |
| `nfft` | STFT window [samples] | 1024 |
| `hop` | STFT hop [samples] | 128 |
| `fs` | target sampling rate [kHz]; `native` = no decimation | 500 |
| `decim` | explicit decimation factor (overrides `fs`; 1 = native) | auto |
| `fmin`, `fmax` | band shown in the PNG [kHz] | full band |
| `thresh` | mask threshold for the overlay | 0.5 |
| `clip_low`, `clip_high` | percentile clip of the log spectrogram | 1, 99 |
| `keep_dc` | keep the DC bin (flag) | off |
| `out` | output root (files go in `<out>/<shot>/`) | `$TOKEYE_RESULTS` = `/fusion/projects/results/tokeye` |
| `tag` | extra label in the file names | — |
| `model` | registry name or `.pt`/`.pt2` file | `big_tf_unet` |
| `backend` | `auto`, `openvino`, `torch` | `auto` (OpenVINO) |
| `threads` | CPU threads | Slurm allocation / CPU quota |
| `noplot`, `nodata` | skip the PNG / the NPZ | off |
| `cmap`, `dpi` | PNG colormap / resolution | viridis, 150 |

Default runs are named `<shot>_<signal>`. A run with non-default data parameters
gets them appended (`190904_MPI66M067D_t2000-3000_nfft2048`), so it never
overwrites the default result. `fmin`/`fmax`/`thresh` only change the picture.

`signal` is any PTDATA pointname (Mirnov `MPI66M067D`, high-res `B1`..`B8`, ECE
`TECEF01`..`48`, CO2 `DENV2_UF`, BES `BESFU01`..`40`) or a preset name (`mag`,
`mag_pol`, `mhr`, `ece`, `co2`, `bes`) meaning its default probe.

### Reading the NPZ

```python
import json, numpy as np
d = np.load("/fusion/projects/results/tokeye/190904/190904_MPI66M067D.npz")
spec = d["spectrogram"]            # (n_freq, n_time) float32, the model input
prob = d["mask"] / 255.0           # (2, n_freq, n_time): [0] coherent, [1] transient
t_ms, f_khz = d["t_ms"], d["f_khz"]
meta = json.loads(str(d["meta"]))
```

### Batch / Slurm

The CLI is headless, so it runs fine in batch jobs. Threads follow
`--cpus-per-task`:

```bash
sbatch -p medium -c 16 --mem=48G -t 1:00:00 --wrap \
  'module use /fusion/projects/codes/tokeye/modulefiles; module load tokeye;
   for s in 190904 190905 190906; do tokeye $s MPI66M067D; done'
```

### Your settings and data

TokEye keeps per-user state in two places, split by lifetime:

| where | what | lifetime |
|---|---|---|
| `~/.tokeye/settings` | your settings (results folder, default rate, threads, engine, data folder) | kept |
| `~/.tokeye/mpl/` | matplotlib config + font cache | kept (regenerated if missing) |
| `/cscratch/$USER/.tokeye/cache/` | fetched signals | **files deleted daily** |
| `/cscratch/$USER/.tokeye/runs/` | web-UI "DIII-D Offline" batch jobs | files deleted daily |
| `/cscratch/$USER/.tokeye/gradio/` | web-UI uploads / rendered outputs | files deleted daily |

Nothing in either folder is assumed to exist: every `tokeye` start rebuilds
whatever is missing, so an empty, half-deleted or missing folder behaves like a
first run. A missing or empty `~/.tokeye/settings` is rewritten with commented
defaults; uncomment a line to change a setting, e.g.

```bash
TOKEYE_RESULTS=~/tokeye_results      # instead of /fusion/projects/results/tokeye
TOKEYE_FS_KHZ=native                 # no decimation by default
TOKEYE_THREADS=8
```

An environment variable of the same name overrides the file, and command-line
parameters (`out=`, `fs=`, `threads=`) override both. The file is read as plain
`KEY=VALUE` lines for those keys only; it is never executed.

Sensor data never goes to permanent storage: fetched signals live only on
`/cscratch`, so the cache is good for the same day (a later run just refetches,
~8 s per probe from atlas). Compute nodes may not reach `atlas.gat.com`, so for
batch jobs fetch on somega the same day, or run on a node that can reach atlas.
Cache entries are plain `.npz` files read with `allow_pickle=False`.

Nothing the install needs lives on `/cscratch`: the env, weights, OpenVINO IR,
launcher and module are all in `/fusion/projects/codes/tokeye`. Only the saved
results of `tokeye <shot> <signal>` (PNG/NPZ/JSON) go to
`/fusion/projects/results/tokeye`, or wherever `out=` points.

### Web app

`tokeye app` (or `tokeye-app`, which also prints the SSH tunnel line) serves the
browser UI on `127.0.0.1:7860`. Tunnel it from your laptop:
`ssh -N -L 7860:localhost:7860 <you>@<node>.gat.com`.

## Memory and time

Inference runs on the whole spectrogram at once. Memory grows linearly with the
number of spectrogram pixels: **about 1.3 KB per pixel, plus ~0.5 GB fixed**,
i.e. ~0.65 GB per 1000 time columns at the default 512 frequency rows.
With the defaults (`n_fft=1024`, `hop=128`) that is **~2.5 GB per second of
signal at 500 kHz** and ~1 GB/s at 200 kHz. Inference time is ~0.4 s per 1000
columns on 10 cores (OpenVINO).

| run (measured on somega, 10 cores) | spectrogram | peak memory | inference |
|---|---|---|---|
| `190904 MPI66M067D` (200 kHz, 10.2 s, whole shot) | 512 × 15,879 | 11 GB | 5–6 s |
| `190904 B1` (2 MHz → 500 kHz, 4.2 s) | 512 × 16,391 | 11.5 GB | 6.5 s |
| 500 kHz, 10 s whole shot (estimate) | 512 × ~39,000 | ~26 GB | ~15 s |
| `fs=native` 2 MHz, 4.2 s (estimate) | 512 × ~65,500 | ~43 GB | ~25 s |

Doubling `n_fft` doubles the rows; halving `hop` doubles the columns; memory
follows the pixel count. A `tmin`/`tmax` window shrinks it proportionally. In
Slurm, ask for memory explicitly (the default is 8 GB per CPU): e.g.
`-c 8 --mem=48G` covers any whole-shot 500 kHz signal.

## CPU performance notes

* Inference uses **OpenVINO** on CPU (falls back to PyTorch). The IR is converted
  once at install time into `models/openvino/`.
* The thread pool is sized to what the process can actually use:
  `$TOKEYE_THREADS` > `SLURM_CPUS_PER_TASK` > cgroup CPU quota (somega caps each
  user at 10 cores) > CPU affinity. The login profile's `OMP_NUM_THREADS=1` would
  otherwise pin PyTorch to one core.
* Heavy interactive work belongs on **somega** (omega14–17), not the login nodes.

## Layout / maintenance

```
/fusion/projects/codes/tokeye/
  src/        git checkout, branch diiid (installed editable)
  env/        conda env: Python 3.13, MDSplus, Qt; pip: CPU torch, OpenVINO, tokeye[app]
  models/     staged weights + OpenVINO IR (no Hugging Face access needed at runtime)
  bin/        tokeye launcher (+ tokeye-app)
  modulefiles/tokeye/default.lua
```

Rebuild or refresh (group `tokeye`, from a node with internet):

```bash
/fusion/projects/codes/tokeye/src/deploy/omega/fusion/install.sh            # refresh
/fusion/projects/codes/tokeye/src/deploy/omega/fusion/install.sh --pull     # update code first
/fusion/projects/codes/tokeye/src/deploy/omega/fusion/install.sh --rebuild  # fresh env
```

Because the checkout is installed editable, a `git pull` in `src/` goes live
immediately. Run `install.sh` again when dependencies change.

Making `module load tokeye` work without `module use`: CSS needs to add
`deploy/omega/fusion/modulefiles/tokeye/default.lua` to the
`GAmfe/css-omega-modules` repo (or to `/fusion/projects/modulefiles-c8`). The
file has absolute paths, so it works as-is from either place.
