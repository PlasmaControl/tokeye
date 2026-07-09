# Deploying TokEye on the Omega cluster (DIII-D)

Runbook for the `diiid` build: a self-contained mamba env + a locally loadable
Lmod module, so `module load tokeye && tokeye` opens the **native desktop GUI**
(the "pyspecview + TokEye" viewer — load a DIII-D shot → spectrogram + mode
overlay, plus the toroidal modespec map) directly over **NoMachine / X11**. A
no-X11 **web app** (`tokeye app`, reached over an SSH tunnel) remains available
as an alternative.

Background/reference: `docs/omega-cluster.md` (cluster facts) and `docs/diiid.md`
(the plan). This directory is the concrete recipe.

**Everything here works without GitHub access.** Contributing the module to
`GAmfe/css-omega-modules` is a later, drop-in step (see the end).

Files:
- `environment-omega.yml` — the mamba env spec (Python 3.13 + MDSplus + `tokeye[app]`).
- `modulefiles/tokeye.lua` — the Lmod modulefile (PATH-prepend; sets `TOKEYE_DIR`/`TOKEYE_CACHE`).
- `tokeye-app.sh` — cluster-side launcher: prints the exact tunnel command, then runs `tokeye app`.
- `tokeye-connect.sh` — laptop-side one-step launcher: tunnel + remote app + open browser.

---

## 0. Where to run

Run on **`somega`** (Omega-Shared: `omega14`–`omega17`), the sanctioned home for
heavy interactive apps — **not** the `omega-a`/`omega-b` login nodes (CPU-heavy
work is killed there). `ssh somega.gat.com` lands you on one of them.

Paths used below (change `TOKEYE_DIR` to relocate):
- env  → `/cscratch/share/tokeye/env-x86_64`
- cache → `/cscratch/share/tokeye/cache`  (auto-set by the module as `$TOKEYE_CACHE`)
- module → `/cscratch/share/tokeye/modulefiles/tokeye.lua`

> `/cscratch` is shared and visible on every node, **but not backed up and files
> >32 days old are swept.** Fine for the env + shot cache now; the durable home
> is `/fusion/projects/codes/tokeye` once CSS creates it (see the end).

## 1. Build the env (once)

```bash
cd <repo root>            # e.g. /cscratch/chenn/tokeye
module purge && module load conda          # `conda` is an alias for mamba on Omega
# Python 3.13 + MDSplus from conda-forge:
mamba env create -p /cscratch/share/tokeye/env-x86_64 -f deploy/omega/environment-omega.yml
# TokEye (editable) + its wheels (torch/torchvision/gradio), from this checkout.
# gradio is pinned to the repo's tested version (uv.lock) in the same command, so
# gradio 6 is never installed — the app isn't gradio-6 ready (6.x drops
# show_download_button and moves theme/css to launch()).
/cscratch/share/tokeye/env-x86_64/bin/pip install -e "$PWD[app]" "gradio==5.49.1"
```

(`module load conda` sets up mamba via a shell hook. If that isn't available in
a non-interactive shell, the mambaforge binary works directly:
`/fusion/projects/codes/conda/mambaforge/bin/mamba env create ...`.)

Verify the things TokEye needs import cleanly (`PySide6`/`pyqtgraph` power the
native GUI and come from conda-forge above; `plotly` powers the web app's plots
and rides in the `[app]` extra):

```bash
/cscratch/share/tokeye/env-x86_64/bin/python -c "import tokeye, MDSplus, PySide6, pyqtgraph, gradio, plotly; print('ok')"
```

## 2. Make the module loadable (local, no GitHub)

> **Re-run this whole step after any env rebuild.** `mamba env create` recreates
> `env-x86_64/bin` from scratch, so the manually-copied `tokeye-app` launcher is
> lost (you'll get `tokeye-app: command not found`); re-copy it below. The
> modulefile also prepends `LD_LIBRARY_PATH` to the env's `lib` so the conda
> `libstdc++`/BLAS win over the node's system `/lib64` — required for numpy/torch to
> import on a fresh somega login.

Publish the modulefile next to the env so anyone can `module use` it:

```bash
mkdir -p /cscratch/share/tokeye/modulefiles
cp deploy/omega/modulefiles/tokeye.lua /cscratch/share/tokeye/modulefiles/
# put the tunnel-printing launcher on PATH (env bin) as `tokeye-app`:
cp deploy/omega/tokeye-app.sh /cscratch/share/tokeye/env-x86_64/bin/tokeye-app
chmod +x /cscratch/share/tokeye/env-x86_64/bin/tokeye-app
# publish the laptop-side launcher so teammates can scp it to their own machines:
cp deploy/omega/tokeye-connect.sh /cscratch/share/tokeye/
chmod 755 /cscratch/share/tokeye/tokeye-connect.sh

module use /cscratch/share/tokeye/modulefiles
module load tokeye
tokeye --help          # subcommands incl. `fetch` and `app`
```

(For a quick dev check straight from the checkout, `module use $PWD/deploy/omega/modulefiles`
works too — the modulefile still points `PATH` at the shared env.)

## 3. Run TokEye

### Native desktop GUI over NoMachine / X11 (recommended)

Connect to **somega** with NoMachine (or `ssh -X somega.gat.com`), then:

```bash
module load tokeye
tokeye                       # opens the DIII-D window — no tunnel, renders on the node
# tokeye gui --view modespec # open straight on the toroidal modespec view
```

The **Spectrogram** view: pick a diagnostic + probe (the time window auto-fills),
**Load shot** → **Analyze** to overlay the coherent (green) / transient (red) mode
mask. Drag to pan, scroll to zoom (centred on the cursor), toggle **Zoom box** to
rubber-band a region, double-click to reset; the crosshair read-out (`t / f / value`)
tracks the cursor. The **Modespec** view is the toroidal mode-number map (discrete
`n`, integer colour bar) with optional **Gate with TokEye** (array average or a
reference probe); the coherence slider re-gates instantly.

If the window doesn't appear, name the missing X library:

```bash
QT_DEBUG_PLUGINS=1 tokeye gui --self-test        # lists any .so the xcb plugin can't load
QT_QPA_PLATFORM=offscreen tokeye gui --self-test  # headless sanity check -> exits 0
```

`xcb-util-cursor` (`libxcb-cursor`) is the usual Qt6-on-HPC culprit; it ships in
the conda env above (pulled in by `pyside6`). The modulefile pins
`QT_QPA_PLATFORM=xcb`; unset it or use `offscreen` for a display-less node.

### Web app over an SSH tunnel (alternative, no X11)

#### One step, from your laptop

Copy the laptop-side launcher once, then run it — it opens the tunnel, starts the
app on the cluster over that same connection, and opens your browser when ready:

```bash
scp <you>@somega.gat.com:/cscratch/share/tokeye/tokeye-connect.sh ~/
~/tokeye-connect.sh <you>@somega.gat.com        # Ctrl-C stops the app + tunnel
```

That's it — the **DIII-D** tab opens in your browser. (The cluster app can't open
your local browser itself: it binds 127.0.0.1 on the cluster, so the tunnel +
browser-open must be driven from the laptop — which is all this script does. The
`somega` round-robin is fine: the forward rides the same SSH session that runs
the app, so it always points at the node the app actually landed on.)

#### Manual, two terminals

```bash
# terminal 1 — on somega:
module load tokeye && tokeye-app       # prints the exact tunnel line (or: tokeye app)
# terminal 2 — on your laptop (use the line it printed):
ssh -N -L 7860:localhost:7860 <you>@omega14.gat.com
# then browse to http://localhost:7860
```

Add `LocalForward 7860 localhost:7860` under the host in your laptop `~/.ssh/config`
to make the tunnel automatic (then `ssh <node>` alone forwards the port).

In the **DIII-D** tab: the shot defaults to the latest on MDS. Pick a diagnostic
+ probe — the time window auto-fills to the signal's data range — then **Load
shot** → **Analyze**. The spectrogram is an **interactive Plotly plot** with real
kHz/ms axes (scroll to zoom, drag to pan); crop the band with **STFT settings →
f-min/f-max**. Threshold / clip / band sliders re-render live (on release).

The **DIII-D Modespec** tab is the classic toroidal mode-number analysis on its
own: enter a shot → an interactive discrete-`n` heatmap (integer colorbar, **hover**
to read the mode number, zoom/pan). **Gate with TokEye** keeps only the modes the
model confirms, computed from the **same array** (band-matched) — **Array average**
(default; cancels single-probe horizontal-line artifacts) or a **Reference probe**.
The coherence-threshold slider re-renders instantly; modespec auto-decimates to
your f-max for speed.

The **DIII-D Offline** tab batches many shots (range `a-b` or a comma list): it
prefetches here on somega, then submits one Slurm job (`gpus`) that runs TokEye +
modespec (gated by the band-matched array average, or a reference probe) over the
cached data — **Refresh** shows status and the result gallery. Offline images are
matplotlib PNGs (Plotly is online-only: its static export needs Kaleido/Chromium,
absent on compute nodes).

Do **not** use `tokeye app --share` on GA (relays through Gradio's cloud, off-network).

## 4. CLI: pre-fetch shots (for batch / GPU nodes that can't reach atlas)

```bash
# on a node that can reach atlas.gat.com (login / somega):
tokeye fetch --shot 190904 --diag mag --pointname MPI66M067D --tlim 1000 3000
#   -> writes data/input/190904_MPI66M067D.npy, cached under $TOKEYE_CACHE

# then run inference anywhere (e.g. a batch GPU job) from the .npy:
tokeye run data/input/190904_MPI66M067D.npy --output-dir results
```

GPU is a Slurm partition (`gpus` → `omega08/09`, V100), not a login node:

```bash
srun --partition=gpus --gres=gpu:v100:1 --time=0-02:00:00 --ntasks=1 --pty /bin/bash -l
module load tokeye && tokeye run ... --device auto     # picks the GPU automatically
```

## 5. Diagnostics

Verified live against DIII-D:
- **`mag`** — toroidal Mirnov array (the default; feeds the modespec analysis).
- **`mag_pol`** — 31-probe 322° poloidal Mirnov (single-probe viewing).
- **`mhr`** — high-res magnetics `B1`–`B8` (~2 MHz).
- **`co2`** — CO2/BCI density chords (`DENVn_UF`). Real source is the BCI.DPD tree
  node / segmented BCI tree (`src/tokeye/sources/co2.py`); the plain `DENVnUF`
  PTDATA is all-zeros.
- **`ece`** — fast ECE `TECEF01`–`TECEF48` (~500 kHz). Real source is the D3D-tree
  node `\D3D::TOP.ELECTRONS.ECE.TECEF:TECEFnn` (`src/tokeye/sources/ece.py`); not
  reachable as plain PTDATA — the same fix pattern as CO2.
- **`bes`** — `BESFU` fast channels (availability varies by shot).

Presets live in `src/tokeye/sources/presets.py`; the probe dropdown also accepts
custom names.

---

## Later, when you have access

1. **Durable home.** Email `omega-support-dl@fusion.gat.com` to create
   `/fusion/projects/codes/tokeye` (community-code home, same class as pyspecview).
   Rebuild the env there and set `TOKEYE_DIR=/fusion/projects/codes/tokeye` — no
   modulefile edit needed. Keep the shot *cache* on `/cscratch` (never write job
   data into `/fusion/projects`).
2. **Public conda env** (optional, for early sharing before a module): email
   `environment-omega.yml` to the same address to have it published; users then
   `mamba activate` it.
3. **Contribute the module** to `GAmfe/css-omega-modules` (needs write access —
   ask CSS):
   ```bash
   git clone git@github.com:GAmfe/css-omega-modules.git
   cd css-omega-modules && git checkout -b tokeye_module
   mkdir -p tokeye && cp <repo>/deploy/omega/modulefiles/tokeye.lua tokeye/default.lua
   module purge && module use $PWD && module load tokeye && tokeye --help   # test pre-merge
   git add tokeye/default.lua && git commit -m "Add tokeye module" && git push origin tokeye_module
   # open a PR — not visible to users until CSS merges (repo syncs daily ~3 AM PDT)
   ```
