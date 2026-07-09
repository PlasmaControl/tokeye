# Omega cluster reference (for deploying TokEye)

Distilled from the official GA docs (mkdocs.gat.com — the Omega cluster + PCS
docs). Only the parts relevant to packaging/running TokEye are kept here; see
`diiid.md` for the actual integration plan. Where a fact changes the plan, a
**→ TokEye** note says how.

**Contacts / repos:**
- `omega-support-dl@fusion.gat.com` — request a `/fusion/projects` directory,
  ask for module-repo write access, submit a public conda env.
- `fus@fusion.gat.com` — request access to the mkdocs documentation repo.
- `git@github.com:GAmfe/css-omega-modules.git` — the modulefiles repo (below).

---

## How software becomes a module (the official workflow)

Omega uses **Lmod** (lua modules). The modulefiles repo is
**`GAmfe/css-omega-modules`** on GitHub; it is synced to
`/fusion/usc/c8/modulefiles-git` **daily ~3 AM PDT**. You contribute via a
branch + pull request; **your module is not visible to users until CSS merges
the PR.**

The end-to-end path for a Python app (this is TokEye's path):

1. **Get a home for the code.** Community codes live in
   `/fusion/projects/codes` (source + binaries only — **no data**). You
   **cannot** create a top-level `/fusion/projects` dir yourself; request one
   from `omega-support-dl@fusion.gat.com` (owner + Linux group control it
   afterward). → **TokEye: request `/fusion/projects/codes/tokeye`** (this is
   the correct home, same as pyspecview — not a personal/scratch dir).
2. **Build a conda env** for the code (see next section) from an
   `environment.yaml` that lives in **your code's repo** (not css-omega-modules).
   → TokEye already ships `environment.yml`; the diiid build adds `mdsplus` +
   `gradio` and pins `python=3.13`.
3. **Author the modulefile:**
   ```bash
   git clone git@github.com:GAmfe/css-omega-modules.git
   cd css-omega-modules
   git checkout -b tokeye_module
   mkdir tokeye && cd tokeye
   cp ../python/module_script.lua.example ./default.lua   # template — edit the <...> fields
   ```
   For multiple versions, one modulefile per version, named by version number.
4. **Test locally before pushing** (this is also how *anyone* can run a module
   that isn't merged yet):
   ```bash
   cd ..              # repo root
   module purge
   module use $PWD    # add this checkout to MODULEPATH
   module load tokeye
   tokeye --help      # or the alias/entrypoint the modulefile defines
   ```
5. **Push + PR:** `git add … && git commit && git push origin tokeye_module`,
   then open a PR. If the push fails, ask CSS for write access.

Docs: modules should be documented on mkdocs.gat.com (repo access via
`fus@fusion.gat.com`).

**Shortcut if you don't need a module yet:** you can have just a *public conda
env* published by emailing its `environment.yaml` to
`omega-support-dl@fusion.gat.com` — users then `mamba activate` it. Good for
early testing before the module PR lands.

---

## Conda / mamba on Omega

- Get it via Lmod (conda is incompatible with the `defaults` env, so purge
  first). No `conda init`/`mamba init` needed — the module handles it:
  ```bash
  module purge
  module load conda
  ```
- **Always use `mamba`** (not `conda`) for `create`/`install` — much faster,
  better solves. They share the same files/format.
- Public envs: `mamba env list` → `mamba activate <env>` → `mamba list`.
  Prefer a public env if one fits before building your own; remove envs you
  stop using (they're large).
- Create from an env file: `mamba env create -n <name>_env -f environment.yaml`.
  Their examples pin `python=3.11`; **TokEye needs `python=3.13`** (fine — it's
  on conda-forge). Relax version pins where possible; `pip:` deps are a last
  resort.

---

## Where to RUN things (usage policy — this shapes Goal 2)

- **Login nodes `omega-a` / `omega-b`** (you land on the least-loaded when you
  ssh `omega.gat.com`): Intel, for **light** interactive use only — editors,
  compiling, GUI-app launching, job submission. **CPU-intensive processes are
  killed indiscriminately** if they hurt the shared node. → **Do NOT run
  `tokeye app` inference here as the default.**
- **`somega.gat.com` (Omega-Shared, nodes `omega14-17`):** the sanctioned home
  for **heavy interactive apps** — OMFIT, VSCode, Jupyter, "any other
  interactive workflows that exceed the login-node limits." Connect with
  `ssh -XY somega.gat.com`. Hard per-user limits: **10 logical cores, 100 GB
  RAM** (app killed if exceeded). → **This is where the CPU `tokeye app` should
  run.** (Use a browser + SSH tunnel; the `-XY` is only needed for real X11
  GUIs, which TokEye is not.)
- **Slurm** for anything heavy/batch or GPU (below). X11 is enabled on Slurm
  (`--x11`), but TokEye prefers web + tunnel.

---

## Slurm partitions & interactive/GPU jobs

Query with `sinfo`. Relevant partitions:

| Partition | Limits / nodes | Use |
|---|---|---|
| `short` | ≤32 CPU, ≤30 min | quick tests |
| `medium` (default) | ≤16 CPU, ≤1 day | interactive/batch |
| `long` | ≤10 CPU, ≤7 days | long jobs |
| `preemptable` | all of `omega[10-13,20-25]`, preemptible | opportunistic |
| `gpus` | `omega08-09`, `--gres=gpu:v100:1` | GPU (V100) |

Interactive CPU job (dedicated resources, unlike the login node):
```bash
srun --partition=medium --nodes=1 --ntasks=1 --cpus-per-task=2 \
     --mem=7GB --time=0-02:30:00 --pty --x11 /bin/bash -l
```
Interactive GPU job (lands you on omega08/09):
```bash
srun --partition=gpus --gres=gpu:v100:1 --time=0-02:00:00 --ntasks=1 --pty /bin/bash -l
```
`--interactive` is additionally needed only for MPI apps (not TokEye).

---

## Scratch & filesystems (this shapes the MDS cache, Goal 3)

- **`/home/$USER`** — 10 GB hard quota, private, backed up. **No envs, no batch
  results, no scratch here.** (Conda envs are GBs → they must live in a
  `/fusion/projects/codes` dir, not home.)
- **`/fusion/projects/…`** — backed up; **do not have jobs write here directly**
  (taxes offsite backups). `codes` = programs only, no data.
- **`/cscratch`** — cluster-wide NFS scratch, visible on all nodes. Your dirs
  are world-readable, writable only by you; `/cscratch/share` is world-rw
  (`chmod -R a+w <dir>` to share your own). **Files >32 days old are deleted;
  not backed up.**
- **`/local-scratch`** — node-local SSD (fast, high-IO), node-only; exported to
  login nodes as `/worker-scratch/<node>`. Slurm CWD can't be in
  `/local-scratch` unless the path exists on the worker.
- **Do NOT use `/tmp`.** Avoid making many small files / frequent small IO —
  it's the main cause of the cluster feeling unresponsive; prefer memory, write
  intermediates to scratch, copy final results to a project/results area, then
  clean scratch.

→ **TokEye:** the MDSplus fetch cache and batch (`tokeye run`) outputs belong on
**`/cscratch`** (a shared cache under `/cscratch/share` lets the team reuse
fetched shots), **not** `/fusion/projects`. Note the tension with "avoid many
small files": the current per-shot pickle cache (`{shot}/{key}.pkl`) makes lots
of small files — consider consolidating per-shot (single file / HDF5) and mind
the 32-day deletion (re-fetch, or promote a curated cache to a project area).

---

## Architecture & node details

**Everything on omega is x86_64**, but heterogeneous microarch: **login + GPU
nodes are Intel**, **most compute nodes are AMD**.

| Nodes | Role | CPU | GPU / RAM |
|---|---|---|---|
| `omega-a`,`omega-b` | login (light interactive) | Intel Xeon Gold 6230 | 748 GB |
| `omega14-17` | **`somega`** Omega-Shared (heavy interactive) | AMD EPYC 7502 | 2 TB |
| `omega08-09` | `gpus` partition | Intel Xeon Gold 6252 | **V100**, 385 GB |
| `omega10-13` | compute | AMD EPYC 7502 (zen2) | 512 GB |
| `omega20-25` | compute | AMD EPYC 7513 (zen3) | 512 GB |

- **Compiled** code can segfault if built with host-specific optimization on
  Intel and run on AMD; the fix is generic flags (`-march=x86-64` / `-tp=px`)
  or compiling on the target. → **TokEye is pure-Python + prebuilt wheels**, so
  this mostly doesn't bite — just avoid any `-march=native` native builds in the
  env so it stays portable across Intel login and AMD compute.
- Omega-specific paths carry **`c8`** in them (Omega began on CentOS 8, now
  RHEL 8).
- Contrast for `diiid.md`'s hopr question: omega is uniformly x86_64, so the
  only architecture split that could matter is a **Grace-Hopper (aarch64)**
  target — plain H100 in x86_64 servers needs no separate env.

---

## Migration gotcha

Do **not** copy old Iris `.bashrc`/`.cshrc`/`.login` to Omega — it breaks the
login environment. Only bring project-specific setting/input files.
