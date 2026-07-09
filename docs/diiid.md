# TokEye on the DIII-D / GA cluster — `diiid` branch plan

Status: **planning doc, no integration code yet.** This captures the plan and
the cluster facts it rests on, so the actual work can start from a known base.
Cluster mechanics (module workflow, run-location policy, scratch, node specs)
are distilled in the adjacent **`omega-cluster.md`** — this file is the plan;
that file is the reference.

Three goals:

1. **Make `tokeye` a loadable cluster module**, like `module load pyspecview`.
2. **A fast remote UI** — X11/XQuartz forwarding is painfully slow.
3. **A DIII-D build that reads shots from MDSplus**, like pyspecview.

The headline finding from investigating the repo: **two of the three already
largely exist.** TokEye ships a **Gradio web app** (`tokeye app`) — that *is*
the modern, no-X11 answer to goal 2 — and it already **reads DIII-D MDSplus**
(`atlas.gat.com` thin client) inside `src/tokeye/modespec/classic/data_utils.py`.
So the `diiid` branch is mostly **packaging and surfacing what exists**, not
building from scratch.

---

## TL;DR

| Goal | Status today | Plan |
|---|---|---|
| 1. Cluster module | not packaged | mamba env (Python 3.13 + MDSplus) in a requested `/fusion/projects/codes/tokeye`; modulefile contributed via a PR to `GAmfe/css-omega-modules` |
| 2. Fast remote UI | already a web app | run `tokeye app` on **`somega`** (Omega-Shared) + `ssh -L` tunnel; do **not** chase X11, and do **not** run it on the login nodes |
| 3. Read MDS | exists, siloed in `modespec` | lift MDS fetchers into a reusable `sources/` layer with DIII-D diagnostic presets; wire into app + CLI; cache to `/cscratch` |

---

## Cluster facts (confirmed by probing omega + official docs)

See `omega-cluster.md` for detail; the plan-critical points:

- **Module system:** Lmod. Modules are contributed as PRs to the GitHub repo
  **`GAmfe/css-omega-modules`** (synced to `/fusion/usc/c8/modulefiles-git`
  daily ~3 AM PDT); **a module isn't visible to users until CSS merges the PR.**
  You can run an unmerged module yourself with `module use <checkout>`.
- **pyspecview is a PyQt desktop GUI** → that is *why* it needs X11 and feels
  slow over XQuartz. It ships `loaders_DIIID/` (machine-specific MDS loaders) —
  the reference for exact pointnames/trees.
- **Python constraint:** TokEye needs `>=3.13` (`pyproject.toml`). Cluster
  Python modules are only 3.7.9 / 3.11 → the env must bring its own 3.13, which
  conda-forge has. Get mamba via `module purge && module load conda` (don't
  hardcode a mambaforge path).
- **MDSplus:** cleanest to `conda install -c conda-forge mdsplus` **into** the
  env (avoids ABI-matching the cluster `mdsplus/d3d/*` module against 3.13).
  Access is the thin client (`MDSplus.Connection('atlas.gat.com')`), reachable
  from login/somega nodes; often **not** from batch compute nodes (→ cache-first).
- **Where the code lives:** community code belongs in `/fusion/projects/codes`
  (source + binaries, **no data**). You can't create the top-level dir yourself
  — **request `/fusion/projects/codes/tokeye`** from
  `omega-support-dl@fusion.gat.com` (same home class as pyspecview). Home dirs
  are 10 GB and private — the env (GBs) cannot live there.
- **`/cscratch` is scratch:** cluster-wide, shared, but **files >32 days old are
  auto-deleted and it is not backed up.** The current checkout at
  `/cscratch/chenn/tokeye` is on scratch — the durable install must not be.

### Compute topology (omega) — all x86_64, heterogeneous microarch

- Login `omega-a`/`omega-b` (Intel): **light** interactive only — CPU-heavy
  processes get killed. **Not** where the app should run.
- **`somega.gat.com`** (Omega-Shared, `omega14-17`, AMD, 2 TB): the sanctioned
  home for heavy interactive apps (OMFIT/VSCode/Jupyter-class). Per-user cap
  **10 cores / 100 GB**. → **this is where CPU `tokeye app` runs.**
- GPU is a **Slurm partition**, not a login node: `gpus` → `omega[08-09]`
  (Intel + **V100**), via `--gres=gpu:v100:1`. Only **2 V100s** total.

### `hopr` — the H100 system (separate cluster)

- **`hopr1.gat.com` (10.220.24.128)** / **`hopr2.gat.com` (10.220.24.129)** — a
  small (2 named nodes) H100 system on GA's `10.220.24.x` network, alongside
  `iris`, `saga`, `saturn`, `theia`/`theia-poc`.
- Invisible from omega: different subnet (omega is `10.1.1.x`), not
  Slurm-federated, no `/fusion/usc/hopr-related` tree yet (new / POC-adjacent).
- **Architecture is the open question that decides one-env-vs-two.** "H100"
  alone usually means discrete H100 in an **x86_64** server (same arch as omega
  → reuse the env). Only **GH200 "Grace Hopper"** is aarch64 (→ a separate
  native env). Confirm with `uname -m` (see the hopr section).

---

## Goal 1 — Make TokEye a module

**Environment (self-contained mamba env — matches pyspecview's conda approach
and gets MDSplus painlessly). Do this after CSS creates the codes dir:**

```bash
module purge && module load conda
mamba create -p /fusion/projects/codes/tokeye/env-x86_64 python=3.13
/fusion/projects/codes/tokeye/env-x86_64/bin/pip install 'tokeye[app]'   # or -e <checkout> for dev
mamba install -p /fusion/projects/codes/tokeye/env-x86_64 -c conda-forge mdsplus
```

Notes:
- Ship the spec as an `environment.yaml` **in the TokEye repo** (the diiid build
  adds `mdsplus` + `gradio` and pins `python=3.13`). GA's python-module guide
  expects the env file to live with the code.
- The PyTorch wheel **bundles its own CUDA runtime** → you only need the NVIDIA
  *driver* (present on GPU nodes), no `module load cuda`. Use a recent **CUDA
  12.x** build so the wheel carries both V100 (`sm_70`) and H100 (`sm_90`)
  kernels → one wheel spans omega and (x86_64) hopr. Keep the env portable
  (avoid `-march=native` native builds) so it runs on Intel login + AMD compute.

**Modulefile** — base the real one on the repo's template
(`python/module_script.lua.example` in css-omega-modules). Improve on
pyspecview's pattern: it uses `set_alias` because it's *one* GUI script; TokEye
is a multi-subcommand CLI, so prepend the env's `bin/` to `PATH`. Make it
arch-aware so one `module load tokeye` works on x86_64 and (if ever) aarch64:

```lua
-- css-omega-modules/tokeye/default.lua  (sketch — start from module_script.lua.example)
help([[TokEye — ML detection of fluctuating modes in spectrograms.
No-X11 web app:  tokeye app   (run on somega)  then from your laptop:
                 ssh -L 7860:localhost:7860 <you>@somega.gat.com
CLI suite:       tokeye run | modespec | elmspec | alfvenspec | eigspec
diiid build reads DIII-D shots from MDSplus (atlas.gat.com).]])
whatis("Name : tokeye  (diiid)")

local root = "/fusion/projects/codes/tokeye"
setenv("TOKEYE_DIR", root)

-- One x86_64 env serves login/somega CPU + V100 + (x86_64) H100. Only
-- Grace-Hopper (aarch64) would need env-aarch64.
local arch = capture("uname -m"):gsub("%s+", "")
local env  = (arch == "aarch64") and "env-aarch64" or "env-x86_64"
prepend_path("PATH", pathJoin(root, env, "bin"))   -- MDSplus bundled in env
```

**Making it loadable (official flow):**
```bash
git clone git@github.com:GAmfe/css-omega-modules.git
cd css-omega-modules && git checkout -b tokeye_module
mkdir tokeye && cp python/module_script.lua.example tokeye/default.lua   # edit <...>
# test locally (also how anyone runs it pre-merge):
module purge && module use $PWD && module load tokeye && tokeye --help
git add tokeye/default.lua && git commit -m "Add tokeye module" && git push origin tokeye_module
# then open a PR — not visible to users until CSS merges
```
Ask CSS for repo write access if the push is rejected. **Shortcut for early
testing:** email the `environment.yaml` to `omega-support-dl@fusion.gat.com` to
publish it as a public conda env (users `mamba activate` it) before the module
PR lands.

**How it mirrors pyspecview:**

| pyspecview | tokeye `diiid` |
|---|---|
| `loaders_DIIID/` | `sources/mds.py` + diagnostic presets |
| Qt window shows spectrogram | Gradio Analyze tab shows spectrogram **+ model mask** |
| X11 forward | `ssh -L` to web app on somega |
| conda env + `.sh` + `.lua` alias | mamba env + `.lua` `PATH`-prepend |

---

## Goal 2 — Beat X11/XQuartz

**The architecture is already right.** X11 forwarding ships rendering
primitives per redraw over the WAN (Mac→GA) — inherently laggy. A web app ships
HTML/JSON and renders in your *local* browser. So the answer is not "make X11
faster," it's "don't use X11":

```bash
# on somega (NOT a login node — heavy interactive apps belong on Omega-Shared):
ssh <you>@somega.gat.com
module load tokeye
tokeye app                                  # binds 127.0.0.1:7860 by default (good — not exposed)
# on your laptop:
ssh -L 7860:localhost:7860 <you>@somega.gat.com
# then open http://localhost:7860
```

To make it turnkey **for others:**
- Add `LocalForward 7860 localhost:7860` to their `~/.ssh/config` for the GA
  host so the tunnel is automatic.
- Print the exact `ssh -L` line (resolved host/port) on app startup → new user
  copy-pastes one line.
- Keep the **localhost bind** (current default): Gradio has no auth, so the
  per-user SSH tunnel *is* the security model. Don't expose it to the network.
- **Avoid `--share`** on GA: it relays through Gradio's cloud (slow, and
  egresses lab data off-network — likely policy-violating).

**General "modern X11" note** (for apps that must stay GUI, like pyspecview
itself): the modern replacements are **FastX** (browser-based remote X — many
DOE/HPC sites run it), **x2go**, or **NoMachine/NX** — all do compression +
caching raw X11 doesn't. But for *TokEye* the web app sidesteps the problem.

### App-speed tiers (GPU folds in here)

| Tier | Where | Notes |
|---|---|---|
| **Default** | **`somega`** CPU (Omega-Shared) | sanctioned for heavy interactive apps; ~5–10 s/inference; **not** the login nodes |
| **Fast** | `gpus` V100 via `srun`/`salloc` + tunnel | <0.5 s, but holds 1 of only **2** V100s |
| **Fastest** | **hopr H100** (`ssh hopr1`, `tokeye app`, tunnel) | best latency; possibly no scheduler to fight — ideal shared-fast home **if** hopr is stable/available |

GPU-interactive recipe (composes with the tunnel plan):
```bash
# on a login node, grab a GPU shell (lands you on omega08/09):
srun --partition=gpus --gres=gpu:v100:1 --time=0-02:00:00 --ntasks=1 --pty /bin/bash -l
module load tokeye && tokeye app --port 7860
# from your laptop (two-hop tunnel straight to the GPU node):
ssh -N -L 7860:localhost:7860 -J <you>@omega.gat.com <you>@omega08
```

**Recommendation:** default the app to **CPU on `somega`**. Single-shot
interactive use at ~5–10 s is fine, MDS fetch + STFT are CPU-bound anyway, and
there are only 2 V100s — don't have the team holding interactive allocations on
them. Reserve GPU (omega V100 or hopr H100) for latency-sensitive use and batch.

---

## Goal 3 — DIII-D build reads from MDS

**What exists:** `data_utils.py` already does pyspecview-style thin-client reads
— `MDSplus.Connection('atlas.gat.com')`, `openTree`, `PTDATA(...)`,
`DIM_OF(...)`, the `s→ms` time fixups, and a **fetch-or-load cache**
(`fetch_or_load`). It's just siloed in the modespec-classic path and specialized
to transient-mode analysis.

**Plan — lift it into a reusable data-source layer** (also serves the ROADMAP's
machine-agnostic "mode catalogue" goal):

- `src/tokeye/sources/` with a small `SignalSource` protocol:
  `fetch(shot, pointname, tlim) -> (t, x, fs)`.
- `MDSSource` (DIII-D): generalize `fetch_ptdata` / `fetch_tree_nodes` out of
  `data_utils` so **both** the app and modespec use one implementation.
- **Diagnostic presets** mirroring pyspecview's `loaders_DIIID/` and the
  README's "Verified Datatypes": Fast Magnetics / Mirnov (`MPI*`), CO2
  interferometer, ECE, BES — as named pointname sets. Use `loaders_DIIID` as the
  authoritative reference for trees/pointnames.
- **Cache-first, on `/cscratch`:** atlas is reachable from login/somega but
  often **not** from batch nodes → "prefetch on a reachable node → run anywhere
  from cache," like the HF-weights flow in the README. Put the cache on
  **`/cscratch`** (never `/fusion/projects`), and a shared team cache under
  **`/cscratch/share`** so a popular shot is fetched once. **Caveats** (from the
  cluster's IO policy): mind the **32-day scratch deletion** (re-fetch, or
  promote a curated cache to a requested project/results dir), and the "avoid
  many small files" guidance — the current per-shot `{shot}/{key}.pkl` layout
  makes lots of small files; consider consolidating per shot (one file / HDF5).

**Wire-up:**
- *App:* a "DIII-D" input mode in the Analyze tab — shot #, diagnostic dropdown,
  time window → `MDSSource.fetch` → the existing STFT → model → overlay pipeline.
  This literally makes TokEye "pyspecview that also labels the modes."
- *CLI:* `tokeye run --shot 190000 --diag mag --tlim 1000 3000`, plus a
  `tokeye fetch` to pre-cache shots.
- *Batch / crawler (later):* use **toksearch** for multi-shot sweeps (per the
  ROADMAP `modesearch` plan) — separate from the interactive thin-client path.

---

## GPU + CPU architecture — the one real env fork

Two independent axes; only one forces a separate environment:

1. **GPU vs CPU — does NOT need separate envs.** A CUDA-build PyTorch runs fine
   on CPU-only nodes (reports `cuda.is_available()==False`), and TokEye's
   `device="auto"` already picks correctly. So **one x86_64 env serves
   login/somega CPU + V100 + (x86_64) H100.** A recent CUDA 12.x wheel includes
   both `sm_70` (V100) and `sm_90` (H100).

2. **CPU architecture — DOES force a split.** omega is uniformly x86_64.
   **Grace Hopper (GH200) is aarch64**, and x86_64 wheels won't run there → a
   **separate native aarch64 env** (aarch64 Python + sbsa CUDA PyTorch), built
   on the aarch64 node (miniforge-aarch64 or `uv` — uv cleanly fetches aarch64
   Python + the sbsa torch wheel). The arch-aware modulefile already selects
   `env-aarch64`.

**So:** likely end state is *mamba for the x86_64 env, uv (or miniforge-aarch64)
for an aarch64 env if ever needed*, tied together by one arch-aware modulefile.
Plain-H100 hopr (if x86_64) needs **no** second env.

---

## `hopr` — how it folds in

Cases, by `uname -m` on `hopr1` and whether `/fusion` is mounted:

- **x86_64 + `/fusion` mounted (most likely):** best case. The *same* mamba env
  on `/fusion` just works on hopr; **H100 ≫ V100** makes hopr the fastest home
  for the interactive app. No arch-aware split needed — `module load tokeye` +
  `device="auto"` uses the H100 automatically.
- **x86_64 + `/fusion` not mounted:** install a second copy of the same x86_64
  recipe locally on hopr. Trivial.
- **aarch64 (GH200 after all):** the arch-aware modulefile + native aarch64 env
  applies.

**Caveat:** hopr looks new and small (2 nodes, POC-adjacent). Treat it as
"fastest when available," not the primary home, until access/longevity and
`/fusion` integration are confirmed. `somega` (CPU) stays the dependable
default; hopr is the speed upgrade.

### Things to confirm on hopr (run when you can log in)

```bash
ssh <you>@hopr1.gat.com 'uname -m; nvidia-smi -L; (sinfo 2>/dev/null || echo NO-SLURM); ls -ld /fusion/projects ~; module avail 2>&1 | head'
```

The four facts that change the plan:
1. `uname -m` → `x86_64` (reuse env) vs `aarch64` (separate env).
2. `/fusion` mounted? → install once on shared FS vs a local copy on hopr.
3. Slurm or free interactive? → a 2-node box may be just "ssh and run."
4. Can hopr reach `atlas.gat.com`? (for MDS) — likely yes on the GA analysis
   network, but confirm.

---

## Suggested `diiid` branch shape & sequencing

Keep it a thin **overlay on `main`** — the source layer and presets are
additive; the module/env lives outside the package. Avoid forking core logic so
`main` improvements flow into `diiid`.

1. **Request the codes dir** (`omega-support-dl@fusion.gat.com`) — the one
   external dependency; start it early.
2. **Module packaging:** mamba env + `tokeye/default.lua` (from the repo
   template) + a `module use` test, then the css-omega-modules PR.
3. **MDS source layer:** refactor `data_utils` fetchers → `sources/mds.py`;
   add DIII-D diagnostic presets; cache to `/cscratch`.
4. **App shot input:** DIII-D mode → fetch → analyze.
5. **CLI shot input:** `tokeye run --shot`, `tokeye fetch`.
6. **(Later) toksearch crawler** for `modesearch`.

---

## Ideas — "could or should"

- **Unify the two MDS code paths.** MDS logic currently lives only in
  modespec-classic. One `MDSSource` used by both is the highest-value cleanup
  and the foundation the ROADMAP catalogue schema wants.
- **"Show + label" loop.** pyspecview *shows* spectrograms; TokEye can show
  *and* auto-label modes. Overlaying the U-Net mask on a live DIII-D
  spectrogram is a genuinely new capability, not a reimplementation.
- **Shared team cache** under `/cscratch/share` so fetched shots are reused
  across users — the second load of a popular shot becomes instant. (Promote a
  curated cache to a requested project/results dir to survive the 32-day scratch
  sweep.)
- **Inter-shot mode** (ROADMAP): web app + shot input makes a "run the last
  shot, show the mode inventory" control-room view natural.
- **Keep sources machine-agnostic.** pyspecview already has AUG/NSTX loaders;
  source = interface, DIII-D = one impl → keeps the door open for
  C-Mod/NSTX-U without schema surgery.
- **Don't over-serve.** Resist a single always-on shared app instance (no auth,
  contention). Per-user `module load tokeye && tokeye app` on somega + personal
  tunnel scales fine and stays secure.
- **GPU batch throughput** (`tokeye run` over many shots) belongs in a Slurm
  batch job on `gpus` (or on hopr), not the interactive app: prefetch MDS on a
  node that can reach atlas → run from the `/cscratch` cache on the GPU node
  (batch nodes have no internet).

---

## Open decisions (for later — not blockers)

- **Codes dir request:** submit to `omega-support-dl@fusion.gat.com` for
  `/fusion/projects/codes/tokeye` (not a blocker, just lead time).
- **Env tool per arch:** mamba on x86_64 (easy MDSplus); uv or miniforge-aarch64
  if an aarch64 target ever materializes.
- **hopr's role:** primary fast home vs occasional speed upgrade — depends on the
  four facts above.
