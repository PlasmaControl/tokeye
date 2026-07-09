# DIII-D Tabs - User Guide

Two tabs turn TokEye into a "pyspecview + TokEye" viewer for DIII-D shots read
straight from MDSplus (`atlas.gat.com`), served in your browser over an SSH
tunnel (no X11):

- **DIII-D** — load one shot, see its spectrogram as a clean image, overlay the
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
   Defaults to the toroidal Mirnov array (`mag` / `MPI66M067D`). Picking a shot +
   probe **auto-fills t-min / t-max** with that signal's actual data window (a
   quick metadata query; cached after the first read).
3. **t min / t max (ms)** — the analysis window, pre-filled to the data range.
   **Narrow it to go faster** — a shorter window means less to fetch and analyze.
4. **STFT settings** (above the spectrogram) — optional. Set the display
   **f-min / f-max (kHz)** band, a **decimation** factor, and the usual
   `n_fft` / hop / clip knobs.
5. **Load shot** — fetches the signal (cached under `$TOKEYE_CACHE`) and shows the
   spectrogram as a clean image cropped to your f-min/f-max band.
6. **View Mode** — choose how to see the result (below), then **Analyze**. The
   built-in `big_tf_unet` model downloads on first use (~30 MB, cached).

Slow steps (fetching, mode analysis) show a **"Fetching… / Computing…"**
progress indicator so you know something is happening.

Fetching needs a node that can reach `atlas.gat.com` (a login or `somega` node).

---

## Diagnostics

Selectable in both tabs. Most are fetched as PTDATA on the `D3D` tree; the probe
dropdown also accepts custom pointnames.

| Key | What | Status |
|---|---|---|
| `mag` | Toroidal Mirnov array (14 B-dot probes) — the default; feeds modespec | ✅ verified |
| `mag_pol` | 322° poloidal Mirnov array (31 probes) — single-probe viewing | ✅ verified |
| `mhr` | High-res magnetics `B1`–`B8` (~2 MHz) — the "b1-b8" set (not Mirnov) | ✅ verified |
| `co2` | CO2 / BCI density chords `DENV1_UF`…`DENR0_UF` (~2 MHz) | ✅ verified (real BCI.DPD source) |
| `bes` | Beam-emission `BESFU01`–`BESFU40` (~1 MHz) | ✅ verified (availability varies by shot) |
| `ece` | Electron-cyclotron `TECEF01`–`TECEF40` | ⚠️ not yet fetchable — see Troubleshooting |

> **CO2 fix.** The obvious `DENV1UF`-style PTDATA pointnames resolve to an
> **all-zeros** array (correct size/timebase, no data) — which is why CO2 used to
> show nothing. The real fast-CO2 data is the `\D3D::TOP.ELECTRONS.BCI.DPD.*:DENUF`
> tree node (recent shots) or segmented `BCI`-tree signals (older shots), chosen by
> a non-zero gate. `tokeye.sources.co2` handles this; `MDSSource.fetch` routes the
> `DENVn_UF` pointnames there automatically (online and offline).

---

## View Modes

Every view renders as a **clean image** (no axis chrome) — you read the frequency
and time range off the **f-min/f-max** (STFT settings) and **t-min/t-max** fields,
pyspecview-style. Wide arrays are auto-binned to display width so full-shot plots
stay fast.

- **Original** — the raw spectrogram (`gist_heat`), cropped to the f-min/f-max band.
- **Enhanced** — the model mask as a smooth green (coherent) / red (transient)
  overlay; **% Min Clip / % Max Clip** set the transparency window.
- **Mask** — the mask thresholded to a hard binary; **Threshold** sets the cut.
- **Amplitude** — the spectrogram spectrally gated by the mask (shows only the
  power the model flagged).
- **Modespec** — the classic toroidal mode-number analysis (see below).

**Coherent / Transient** checkboxes toggle the two mask channels.

### Live controls

**Threshold**, **% Min Clip**, **% Max Clip**, the **f-min / f-max** band, and the
modespec **Coherence threshold** re-render the picture immediately (on release /
change) — they only re-color or re-crop an already-computed array, so there's no
recompute and no re-inference. (`Decimation`, `n_fft`, `hop`, and clip percentiles
change the actual transform, so they take effect on the next **Load shot** —
`n_fft`/hop/clip also need **Apply Transform Settings** first.)

> **f-min/f-max is display-only for TokEye views.** It crops what you *see* (and
> sets the modespec band); the U-Net still runs on the full band, so mask results
> are unchanged. **Decimation** *does* change the signal fed to the STFT/model
> (fewer samples, lower Nyquist) — like `n_fft`/hop, it's a transform knob (default
> 1 = off).

---

## Modespec + TokEye gating

Set **View Mode → Modespec** and **Analyze**. This runs the vendored
`mode_spectrogram` on the **toroidal Mirnov array** (independent of the single
probe you loaded) and plots the **dominant toroidal mode number `n`** vs frequency
and time, as a **plain discrete-rainbow (`turbo`) image** — one distinct colour per
mode number, dark where the mode coherence is below threshold. The `n` → colour
**legend renders in the page beside the image** (same colours), not baked into the
plot.

Controls (in the Modespec group):

- **n min / n max** — range of toroidal mode numbers to fit (default −5…5).
- **Coherence threshold** — keep bins above this coherence (live slider).
- **Gate with TokEye** — the payoff. TokEye is a strong gate: it takes the loaded
  probe's **coherent** mask (thresholded), resamples it onto the modespec
  frequency×time grid, and keeps a mode number only where **the mask is on AND
  coherence > threshold**. The result is a much cleaner mode plot — the scattered
  low-coherence noise drops out and only mode activity the model confirms remains.

The analysis **frequency band comes from STFT settings → f-min/f-max** (so the
spectrogram and the modes share one band), and **Decimation** applies here too.

> **Why modespec used to be slow, and the fix.** The Mirnov digitizer runs at
> 200 kHz–2 MHz and `mode_spectrogram`'s cost scales with the sample count, so a
> multi-second, full-rate shot means millions of samples/probe → thousands of STFT
> windows. Since MHD modes live below your `f-max`, the extra samples are pure cost.
> The tab now **decimates each probe to just above `2·f_max`** (anti-aliased, no
> in-band loss) before the analysis — a near-linear speedup (≈2–8× depending on the
> probe's rate) with identical modes. Narrowing **t-min/t-max** helps too.

> First Modespec run for a shot fetches all 14 probes (slower); they're cached, so
> re-runs and threshold tweaks are fast. To gate, load a probe first (the gate uses
> that probe's mask), then Analyze in Modespec mode.

---

## DIII-D Offline tab (batch many shots)

Because cluster compute nodes can't reach atlas, the app **prefetches every shot
here on `somega`** (which can), then submits one Slurm job that runs over the
cached data.

1. **Shots** — an inclusive range `150000-150010` and/or a comma list
   `150000,150002,150005` (both can be combined).
2. **Diagnostic / Probe / Model / time window** — as in the online tab.
3. **Analyses** — tick any of **TokEye mask**, **Modespec**, **TokEye-gated
   modespec** (gating implies both of the others). Set **n min/max**, **f min/max**,
   and **Decimation** for the modespec runs.
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
    --tokeye --modespec --gate --decimation 1 --device auto
```

`tokeye diiid-batch` reads prefetched signals from the cache, so it runs on a
no-internet compute node. Its exit code is the number of shots that failed.
`--decimation` floors the automatic f-max-safe decimation for the modespec runs.

---

## Key Concepts

### Caching

Every probe fetch is cached to `$TOKEYE_CACHE` (default
`/cscratch/share/tokeye/cache`) as `{shot}/{shot}_{pointname}.pkl`. A shot fetched
once on `somega` replays instantly anywhere — this is what lets the offline Slurm
job work with no atlas access. Time-window bounds are cached in-process too, so
re-selecting a probe fills the window instantly.

### Plain images + display band

Views render server-side to clean PNGs (fast, no plot chrome). The **f-min/f-max**
band crops the *displayed* spectrogram and sets the modespec analysis band; it does
**not** change the U-Net input (cropping that would push it off the training
distribution). Modespec colours come from a discrete `turbo` map with a matching
in-page legend.

### Decimation & modespec speed

`Decimation` downsamples the signal (anti-aliased) before the STFT — a transform
knob for the spectrogram/model (default 1). Modespec additionally auto-decimates to
just above `2·f_max` for speed; the frequency-bin spacing is unchanged, so the modes
are identical to the full-rate result.

### Why prefetch on `somega`

Login/`somega` nodes reach `atlas.gat.com`; the `gpus` compute nodes do not. So the
app fetches (and downloads the model) on `somega`, then the batch job only does
inference/analysis over the cache.

---

## Troubleshooting

**"Enter a shot number and pick a diagnostic/probe first."** — Shot or probe was
empty. If Shot is 0, MDS was unreachable at page load; type a shot manually.

**t-min/t-max didn't auto-fill.** — The bounds query is best-effort; if atlas was
briefly unreachable the fields stay blank (just type a window, or Load — the fetch
still works). The first read for a probe takes a few seconds (building the time
base server-side); it's cached after.

**CO2 shows nothing.** — Should be fixed: `co2` now reads the real BCI.DPD source
(the `DENVnUF` PTDATA was all-zeros). If a specific shot is still blank, its fast
CO2 may simply not be digitized — try another shot.

**Fetch failed / `TdiABORT` for ECE (`TECEF…`).** — Expected for now. ECE channels
live in a separate `ece` MDSplus tree, not PTDATA, so the generic fetch can't reach
them (the repo's `modespec.fetch_ece` shows the tree path). The channels are listed
for selection; wiring the tree fetch is a follow-up.

**Modespec takes a long time.** — The first run fetches all 14 probes (cached
after). For the compute itself, raise **Decimation** and/or narrow **t-min/t-max**;
the analysis auto-decimates to your **f-max** already.

**Modespec / gated view looks empty or too sparse.** — Modes are gated by
coherence. Lower the **Coherence threshold** (watch the `c95` noise floor — bins
below it are noise) and, for gating, the **Threshold** slider. Some shots/windows
simply have little coherent mode activity.

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
`modespec/classic/*`) is **untouched**; only `app/__main__.py` and `cli/__init__.py`
gain a couple of registration lines. Feature code lives in diiid-only files:

- `sources/presets.py` — diagnostic → pointnames (+ toroidal angles); CO2 chord names.
- `sources/mds.py` — single-probe fetch (routes CO2), `latest_shot()`, `time_bounds()`.
- `sources/co2.py` — real CO2/BCI fetch (DPD node + segmented BCI, non-zero gate).
- `sources/mirnov.py` — cached Mirnov-array fetch, `run_mode_spectrogram` (with
  auto-decimation), `gate_dominant` (the TokEye→modespec grid resample + intersect).
- `sources/viz.py` — gradio-free **plain-image** renderers (`render_view`,
  `render_modespec`) + the shared `turbo` legend (`mode_color_legend_html`); reused
  by the CLI.
- `app/tabs/diiid.py`, `app/tabs/diiid_offline.py` — the two tabs.
- `cli/diiid_batch.py` — the `tokeye diiid-batch` runner + `parse_shots` /
  `build_sbatch_script`.

Tests (`tests/test_sources.py`, `tests/test_app_diiid_tab.py`,
`tests/test_diiid_batch.py`) are all offline / MDSplus-free.
