# DIII-D Tabs - User Guide

Three tabs turn TokEye into a "pyspecview + TokEye" viewer for DIII-D shots read
straight from MDSplus (`atlas.gat.com`), served in your browser over an SSH
tunnel (no X11):

- **DIII-D** — load one shot, see its spectrogram as an **interactive Plotly plot**
  (real kHz/ms axes, zoom/pan), and overlay the ML mode mask.
- **DIII-D Modespec** — the classic toroidal mode-number analysis on its own: an
  interactive discrete-`n` heatmap, optionally **gated by TokEye computed from the
  same array** (band-matched).
- **DIII-D Offline** — batch many shots on the cluster (Slurm) and view the gallery
  of results when the job finishes.

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
   spectrogram as an interactive Plotly plot cropped to your f-min/f-max band.
6. **View Mode** — choose how to see the result (below), then **Analyze**. The
   built-in `big_tf_unet` model downloads on first use (~30 MB, cached).

Slow steps (fetching, mode analysis) show a **"Fetching… / Computing…"**
progress indicator so you know something is happening.

Fetching needs a node that can reach `atlas.gat.com` (a login or `somega` node).

---

## Diagnostics

Selectable in the DIII-D + Offline tabs. Most are fetched as PTDATA on the `D3D`
tree; the probe dropdown also accepts custom pointnames.

| Key | What | Status |
|---|---|---|
| `mag` | Toroidal Mirnov array (14 B-dot probes) — the default; feeds modespec | ✅ verified |
| `mag_pol` | 322° poloidal Mirnov array (31 probes) — single-probe viewing | ✅ verified |
| `mhr` | High-res magnetics `B1`–`B8` (~2 MHz) — the "b1-b8" set (not Mirnov) | ✅ verified |
| `co2` | CO2 / BCI density chords `DENV1_UF`…`DENR0_UF` (~2 MHz) | ✅ verified (real BCI.DPD source) |
| `bes` | Beam-emission `BESFU01`–`BESFU40` (~1 MHz) | ✅ verified (availability varies by shot) |
| `ece` | Electron-cyclotron `TECEF01`–`TECEF48` (~500 kHz) | ✅ verified (real D3D-tree TECEF node) |

> **CO2 fix.** The obvious `DENV1UF`-style PTDATA pointnames resolve to an
> **all-zeros** array (correct size/timebase, no data) — which is why CO2 used to
> show nothing. The real fast-CO2 data is the `\D3D::TOP.ELECTRONS.BCI.DPD.*:DENUF`
> tree node (recent shots) or segmented `BCI`-tree signals (older shots), chosen by
> a non-zero gate. `tokeye.sources.co2` handles this; `MDSSource.fetch` routes the
> `DENVn_UF` pointnames there automatically (online and offline).

> **ECE fix (same pattern).** Fast ECE `TECEFnn` isn't reachable as plain PTDATA
> (the vendored `modespec.fetch_ece` reads a separate `ece` tree). The simplest
> working source — confirmed in `FusionAIHub`'s `co2_check` — is the D3D-tree node
> `\D3D::TOP.ELECTRONS.ECE.TECEF:TECEFnn` (48 fast channels, ~500 kHz), chosen by a
> non-zero gate. `tokeye.sources.ece` handles this; `MDSSource.fetch` routes the
> `TECEFnn` pointnames there automatically. Older shots may have fewer than 48
> channels (a missing one just fails the non-zero gate).

---

## View Modes (DIII-D tab)

Every view is an **interactive Plotly plot** with real **frequency (kHz)** and
**time (ms)** axes — scroll to zoom, drag to pan, double-click to reset. Wide
arrays are auto-binned to display width so full-shot plots stay fast.

- **Original** — the raw spectrogram (`gist_heat`), cropped to the f-min/f-max band.
- **Enhanced** — the model mask as a smooth green (coherent) / red (transient)
  overlay; **% Min Clip / % Max Clip** set the transparency window.
- **Mask** — the mask thresholded to a hard binary; **Threshold** sets the cut.
- **Amplitude** — the spectrogram spectrally gated by the mask (shows only the
  power the model flagged).

**Coherent / Transient** checkboxes toggle the two mask channels. (The classic
toroidal mode-number analysis is now its own **DIII-D Modespec** tab — see below.)

### Live controls

**Threshold**, **% Min Clip**, **% Max Clip**, and the **f-min / f-max** band
re-render the picture immediately (on release / change) — they only re-color or
re-crop an already-computed array, so there's no recompute and no re-inference.
(`Decimation`, `n_fft`, `hop`, and clip percentiles change the actual transform, so
they take effect on the next **Load shot** — `n_fft`/hop/clip also need **Apply
Transform Settings** first.)

> **f-min/f-max is display-only for TokEye views.** It crops what you *see*; the
> U-Net still runs on the full band, so mask results are unchanged. **Decimation**
> *does* change the signal fed to the STFT/model (fewer samples, lower Nyquist) —
> like `n_fft`/hop, it's a transform knob (default 1 = off).

---

## DIII-D Modespec tab (band-matched gating)

Modespec is its own self-contained tab (matching how `modespec` lives in the src).
Enter a **shot** and **Analyze**: it fetches the **toroidal Mirnov array**, runs the
vendored `mode_spectrogram`, and plots the **dominant toroidal mode number `n`** vs
frequency and time as an **interactive discrete-rainbow (`turbo`) heatmap** — one
distinct colour per mode number, an **integer colorbar** on the side (this replaces
the old external HTML legend), and **hover** to read `t / f / n`. Suppressed
(low-coherence) bins are transparent over a dark panel.

Controls:

- **Reference probe** — used to auto-fill the time window, and as the gate reference
  when gate source is *Reference probe*.
- **f min / f max (kHz)** — the modespec analysis band. **Decimation** speeds it up.
- **n min / n max** — range of toroidal mode numbers to fit (default −5…5).
- **Coherence threshold** — keep bins above this coherence (**live** slider:
  re-renders instantly from the cached result, no recompute).
- **Gate with TokEye** + **Gate source** — the payoff.

### The gate is band-matched (the fix)

The old DIII-D-tab gate used the single loaded probe's mask — which could be a
**different frequency band** than modespec (or a single probe whose stationary
pickup/harmonics show up as spurious **horizontal lines**), so the gated ratios were
wrong. The Modespec tab computes the gate from the **same toroidal array** modespec
runs on:

- **Array average** (default) — run TokEye on every probe, average the coherent-mode
  probability across probes, threshold. Averaging **cancels single-probe
  horizontal-line artifacts** — only mode activity most probes agree on survives.
- **Reference probe** — gate from one chosen probe (the reference dropdown).

Either way, TokEye keeps a mode number only where **the (band-matched) mask is on
AND coherence > threshold**, giving a much cleaner mode plot.

> **Why modespec used to be slow, and the fix.** The Mirnov digitizer runs at
> 200 kHz–2 MHz and `mode_spectrogram`'s cost scales with the sample count, so a
> multi-second, full-rate shot means millions of samples/probe → thousands of STFT
> windows. Since MHD modes live below your `f-max`, the extra samples are pure cost.
> The tab **decimates each probe to just above `2·f_max`** (anti-aliased, no in-band
> loss) — a near-linear speedup (≈2–8×) with identical modes. The gate uses the same
> decimation, so it stays cheap (14 inferences) and band-matched. Narrowing
> **t-min/t-max** helps too.

> First Analyze for a shot fetches all 14 probes (slower); they're cached, so re-runs
> and coherence-threshold tweaks are fast. Gating loads the model on first use
> (~30 MB, cached).

---

## DIII-D Offline tab (batch many shots)

Because cluster compute nodes can't reach atlas, the app **prefetches every shot
here on `somega`** (which can), then submits one Slurm job that runs over the
cached data.

1. **Shots** — an inclusive range `150000-150010` and/or a comma list
   `150000,150002,150005` (both can be combined).
2. **Diagnostic / Probe / Model / time window** — as in the online tab.
3. **Analyses** — tick any of **TokEye mask**, **Modespec**, **TokEye-gated
   modespec**. **Gate source** (`average` | `reference`) picks the band-matched gate
   — *average* over the array (default) or the probe above as reference. Set **n
   min/max**, **f min/max**, and **Decimation** for the modespec runs.
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
- `<shot>_modespec_gated.png` (TokEye-gated modespec, band-matched array gate)

Offline images are **matplotlib PNGs** with clean axes (Plotly is online-only: its
static export needs Kaleido/Chromium, absent on compute nodes).

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
    --tokeye --modespec --gate --gate-source average --decimation 1 --device auto
```

`tokeye diiid-batch` reads prefetched signals from the cache, so it runs on a
no-internet compute node. Its exit code is the number of shots that failed.
`--decimation` floors the automatic f-max-safe decimation for the modespec runs;
`--gate-source {average,reference}` (+ `--reference-probe`) picks the band-matched
gate for `--gate`.

---

## Key Concepts

### Caching

Every probe fetch is cached to `$TOKEYE_CACHE` (default
`/cscratch/share/tokeye/cache`) as `{shot}/{shot}_{pointname}.pkl`. A shot fetched
once on `somega` replays instantly anywhere — this is what lets the offline Slurm
job work with no atlas access. Time-window bounds are cached in-process too, so
re-selecting a probe fills the window instantly.

### Interactive plots (Plotly online, matplotlib offline)

Online figures are Plotly (`gr.Plot`): the browser renders them client-side with
real kHz/ms axes, zoom/pan, and (for modespec) hover-to-read `n`. Spectrogram views
are drawn as a compact stretched PNG image over the axes; modespec is a `go.Heatmap`
with a discrete `turbo` colorscale + integer colorbar. The offline batch keeps
**matplotlib** PNGs (Plotly's static export needs Kaleido/Chromium, which compute
nodes lack). The **f-min/f-max** band crops the *displayed* spectrogram (DIII-D tab)
and sets the modespec analysis band (Modespec tab); it does **not** change the U-Net
input.

### Decimation & modespec speed

`Decimation` downsamples the signal (anti-aliased) before the STFT — a transform
knob for the spectrogram/model (default 1). Modespec additionally auto-decimates to
just above `2·f_max` for speed; the frequency-bin spacing is unchanged, so the modes
are identical to the full-rate result. The band-matched gate uses the same
decimation.

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

**ECE (`TECEF…`) shows nothing / fetch fails.** — `ece` now reads the D3D-tree node
`\D3D::TOP.ELECTRONS.ECE.TECEF:TECEFnn` (not plain PTDATA, not the separate `ece`
tree). If a specific channel/shot is blank, that channel may be absent or all-zero
for the shot (older shots have fewer than 48 channels) — try another channel.

**Modespec takes a long time.** — The first run fetches all 14 probes (cached
after), and gating runs the model on each of them. Raise **Decimation** and/or narrow
**t-min/t-max**; the analysis + gate auto-decimate to your **f-max** already. Turn off
**Gate with TokEye** for a fast coherence-only baseline.

**Modespec / gated view looks empty or too sparse.** — Modes are gated by coherence.
Lower the **Coherence threshold** (watch the `c95` noise floor). If gating drops too
much, try **Array average** vs **Reference probe**, or turn gating off.

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
`modespec/classic/*`) is **untouched**; only `pyproject.toml` (the `app` extra gains
`plotly`), `app/__main__.py` (registers the three tabs), and `cli/__init__.py` gain a
couple of registration lines. Feature code lives in diiid-only files:

- `sources/presets.py` — diagnostic → pointnames (+ toroidal angles); CO2 chord names.
- `sources/mds.py` — single-probe fetch (routes CO2 + ECE), `latest_shot()`, `time_bounds()`.
- `sources/co2.py` — real CO2/BCI fetch (DPD node + segmented BCI, non-zero gate).
- `sources/ece.py` — real fast-ECE fetch (D3D-tree TECEF node, non-zero gate; CO2-style).
- `sources/mirnov.py` — cached Mirnov-array fetch, `run_mode_spectrogram` (with
  auto-decimation), `array_gate_mask` (the band-matched TokEye gate from the array),
  and `gate_dominant_mask` / `gate_dominant` (the TokEye→modespec grid resample +
  intersect).
- `sources/viz.py` — Plotly renderers `plotly_view` / `plotly_modespec` (online) +
  matplotlib `render_modespec_png` (offline PNGs); reused by the CLI.
- `app/tabs/diiid.py` — the spectrogram/mask viewer (Plotly).
- `app/tabs/diiid_modespec.py` — the self-contained Modespec tab (band-matched gate).
- `app/tabs/diiid_offline.py` — the batch tab.
- `cli/diiid_batch.py` — the `tokeye diiid-batch` runner + `parse_shots` /
  `build_sbatch_script`.

Tests (`tests/test_sources.py`, `tests/test_app_diiid_tab.py`,
`tests/test_diiid_batch.py`) are all offline / MDSplus-free.
