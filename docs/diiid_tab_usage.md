# DIII-D Tabs - User Guide

Two tabs turn TokEye into a "pyspecview + TokEye" viewer for DIII-D shots read
straight from MDSplus (`atlas.gat.com`), served in your browser over an SSH
tunnel (no X11):

- **DIII-D** — load one shot, see its spectrogram with real axes, overlay the
  ML mode mask, and run the classic toroidal mode-number analysis (optionally
  gated by the mask).
- **DIII-D Offline** — batch many shots on the cluster (Slurm) and view the
  gallery of results when the job finishes.

For how to launch the app on the cluster, see `deploy/omega/README.md`. For the
background/plan and cluster facts, see `diiid.md` and `omega-cluster.md`.

---

## Quick Start (DIII-D tab)

1. **Shot** — defaults to the latest shot on MDS. Type a different one if you
   want (0 means MDS was unreachable at load).
2. **Diagnostic + Probe** — pick a diagnostic; the probe dropdown repopulates.
   Defaults to the toroidal Mirnov array (`mag` / `MPI66M067D`).
3. **t min / t max (ms)** — optional time window. Leave blank for the full shot
   (slower to fetch; a window is recommended).
4. **Load shot** — fetches the signal (cached under `$TOKEYE_CACHE`) and shows
   the spectrogram with real frequency (kHz) and time (ms) axes.
5. **View Mode** — choose how to see the result (below), then **Analyze**. The
   built-in `big_tf_unet` model downloads on first use (~30 MB, cached).

Fetching needs a node that can reach `atlas.gat.com` (a login or `somega` node).

---

## Diagnostics

Selectable in both tabs. All are fetched as PTDATA on the `D3D` tree except
where noted; the probe dropdown also accepts custom pointnames.

| Key | What | Status |
|---|---|---|
| `mag` | Toroidal Mirnov array (14 B-dot probes) — the default; feeds modespec | ✅ verified |
| `mag_pol` | 322° poloidal Mirnov array (31 probes) — single-probe viewing | ✅ verified |
| `mhr` | High-res magnetics `B1`–`B8` (~2 MHz) — the "b1-b8" set (not Mirnov) | ✅ verified |
| `co2` | CO2 / BCI density chords | ✅ verified |
| `bes` | Beam-emission `BESFU01`–`BESFU40` (~1 MHz) | ✅ verified (availability varies by shot) |
| `ece` | Electron-cyclotron `TECEF01`–`TECEF40` | ⚠️ not yet fetchable — see Troubleshooting |

---

## View Modes

Every view is drawn with real **kHz / ms axes** and (for scalar views) a
colorbar. Wide arrays are auto-binned to display width so full-shot plots stay
fast.

- **Original** — the raw spectrogram.
- **Enhanced** — the model mask as a smooth green (coherent) / red (transient)
  overlay; **% Min Clip / % Max Clip** set the transparency window.
- **Mask** — the mask thresholded to a hard binary; **Threshold** sets the cut.
- **Amplitude** — the spectrogram spectrally gated by the mask (shows only the
  power the model flagged).
- **Modespec** — the classic toroidal mode-number analysis (see below).

**Coherent / Transient** checkboxes toggle the two mask channels.

### Live sliders

**Threshold**, **% Min Clip**, **% Max Clip**, and the modespec **Coherence
threshold** re-render the picture the moment you release the slider — they only
re-color an already-computed array, so there's no recompute and no re-inference.
(`n_fft` / `hop` / clip percentiles under **STFT settings** *do* change the
transform, so they take effect on the next **Apply Transform Settings** +
**Load shot**.)

---

## Modespec + TokEye gating

Set **View Mode → Modespec** and **Analyze**. This runs the vendored
`mode_spectrogram` on the **toroidal Mirnov array** (independent of the single
probe you loaded) and plots the **dominant toroidal mode number `n`** vs
frequency and time, colored by `n`, masked where the mode coherence exceeds the
threshold.

Controls (in the Modespec group):

- **n min / n max** — range of toroidal mode numbers to fit (default −5…5).
- **f min / f max (kHz)** — analysis band (default 5–200).
- **Coherence threshold** — keep bins above this coherence (live slider).
- **Gate with TokEye** — the payoff. TokEye is a strong gate: it takes the
  loaded probe's **coherent** mask (thresholded), resamples it onto the modespec
  frequency×time grid, and keeps a mode number only where **the mask is on AND
  coherence > threshold**. The result is a much cleaner mode plot — the scattered
  low-coherence noise drops out and only mode activity the model confirms remains.

> First Modespec run for a shot fetches all 14 probes (slower); they're cached,
> so re-runs and threshold tweaks are fast. To gate, load a probe first (the gate
> uses that probe's mask), then Analyze in Modespec mode.

---

## DIII-D Offline tab (batch many shots)

Because cluster compute nodes can't reach atlas, the app **prefetches every shot
here on `somega`** (which can), then submits one Slurm job that runs over the
cached data.

1. **Shots** — an inclusive range `150000-150010` and/or a comma list
   `150000,150002,150005` (both can be combined).
2. **Diagnostic / Probe / Model / time window** — as in the online tab.
3. **Analyses** — tick any of **TokEye mask**, **Modespec**, **TokEye-gated
   modespec** (gating implies both of the others).
4. **Cluster / output** (accordion) — **Output folder** (default
   `/cscratch/$USER/tokeye/data/runs`, editable), **Partition** (`gpus`), **GRES**
   (`gpu:v100:1`; blank = CPU partition), **Time limit**.
5. **Prefetch + Submit job** — shows a prefetch progress bar, then submits and
   reports the Slurm job id. Results go to `<output folder>/<shots>-<diag>/`.
6. **Refresh status / results** — polls `squeue`/`sacct`, tails the job log, and
   fills the **gallery** with the per-shot images as they appear.

Per shot the job writes, into the output folder:

- `<shot>_<probe>_mask.npy` + `<shot>_<probe>_preview.png` (TokEye)
- `<shot>_modespec.png` + `<shot>_modes.csv` (modespec + detected mode events)
- `<shot>_modespec_gated.png` (TokEye-gated modespec)

> `/cscratch` is shared and visible on every node but **not backed up, and files
> older than ~32 days are swept** — copy anything you want to keep.

---

## CLI equivalents

Everything the tabs do is scriptable (heavy imports are deferred, so `--help` is
instant):

```bash
# Pre-cache one shot's signal to a .npy (run where atlas is reachable):
tokeye fetch --shot 190904 --diag mag --pointname MPI66M067D --tlim 1000 3000

# The offline job body (cache-only — what the Offline tab submits via Slurm):
tokeye diiid-batch --shots 190900-190905,190910 --outdir results \
    --diag mag --probe MPI66M067D --tlim 1000 3000 \
    --tokeye --modespec --gate --device auto
```

`tokeye diiid-batch` reads prefetched signals from the cache, so it runs on a
no-internet compute node. Its exit code is the number of shots that failed.

---

## Key Concepts

### Caching

Every probe fetch is cached to `$TOKEYE_CACHE` (default
`/cscratch/share/tokeye/cache`) as `{shot}/{shot}_{pointname}.pkl`. A shot
fetched once on `somega` replays instantly anywhere — this is what lets the
offline Slurm job work with no atlas access.

### Real axes without changing the model

`Load shot` records the true sampling rate and window start. Those are used only
to **label the display axes** — the spectrogram fed to the model is computed
exactly as before, so results are unchanged from earlier versions.

### Why prefetch on `somega`

Login/`somega` nodes reach `atlas.gat.com`; the `gpus` compute nodes do not. So
the app fetches (and downloads the model) on `somega`, then the batch job only
does inference/analysis over the cache.

---

## Troubleshooting

**"Enter a shot number and pick a diagnostic/probe first."** — Shot or probe was
empty. If Shot is 0, MDS was unreachable at page load; type a shot manually.

**Fetch failed / `TdiABORT` for ECE (`TECEF…`).** — Expected for now. ECE
channels live in a separate `ece` MDSplus tree, not PTDATA, so the generic fetch
can't reach them (the repo's `modespec.fetch_ece` shows the tree path). The
channels are listed for selection; wiring the tree fetch is a follow-up. All
other diagnostics fetch via PTDATA.

**A diagnostic returns no samples on a recent shot.** — Not every diagnostic is
digitized on every shot (BES especially). Try another shot or diagnostic.

**Modespec view is blank.** — It needs an Analyze in Modespec mode (switching the
view alone won't fetch the 14-probe array). The first run for a shot is slower.

**Gated modespec looks empty / too sparse.** — The gate is deliberately strict.
Lower the **Threshold** (mask) and/or **Coherence threshold** sliders — both are
live — to keep more bins.

**Offline: "sbatch not found".** — Run the app on a Slurm submit node (`somega`).
The prefetch still ran; the script is written to `<output>/submit.sh`.

**Offline: gallery empty after submit.** — The job may still be queued/running;
click **Refresh**. Check the state line and the tailed log; results appear as the
job writes them.

Do **not** use `tokeye app --share` on GA — it relays through Gradio's cloud,
off-network.

---

## For maintainers

This is a purely additive overlay on the `diiid` branch. The shared pipeline
(`app/analyze/*`, `transforms.py`, `inference.py`, `batch.py`, `hub.py`, vendored
`modespec/classic/*`) is **untouched**; only `app/__main__.py` and
`cli/__init__.py` gain a couple of registration lines. Feature code lives in
diiid-only files:

- `sources/presets.py` — diagnostic → pointnames (+ toroidal angles).
- `sources/mds.py` — single-probe fetch + `latest_shot()`.
- `sources/mirnov.py` — cached Mirnov-array fetch, `run_mode_spectrogram`,
  `gate_dominant` (the TokEye→modespec grid resample + intersect).
- `sources/viz.py` — gradio-free matplotlib renderers with axes (`render_view`,
  `render_modespec`); reused by the CLI.
- `app/tabs/diiid.py`, `app/tabs/diiid_offline.py` — the two tabs.
- `cli/diiid_batch.py` — the `tokeye diiid-batch` runner + `parse_shots` /
  `build_sbatch_script`.

Tests (`tests/test_sources.py`, `tests/test_app_diiid_tab.py`,
`tests/test_diiid_batch.py`) are all offline / MDSplus-free.
