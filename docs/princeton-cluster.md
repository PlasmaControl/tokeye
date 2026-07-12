# Princeton stellar cluster reference (for deploying TokEye)

Everything below was verified empirically on stellar (2026-07). It backs the
choices in `deploy/princeton/` and this branch's defaults.

## Branch topology

- `main` — the generic tool.
- `facility` — shared machine-facing trunk: sources protocol + factory
  (`$TOKEYE_SOURCE`), PySide6 GUI, viz, tab framework, the DIII-D tab modules.
- `diiid`, `princeton` — site branches forked from `facility`; each holds only
  site-specific code (this one: `FoundationSource`, the Princeton tab,
  `princeton-batch`, the stellar deploy kit).

Track by **merge**, never rebase: `main → facility → {diiid, princeton}`.
Site-specific work never lands on `facility`. The one intentional factory
divergence per site branch is `DEFAULT_SOURCE_KIND` (`foundation` here).

## Group modules (Tcl environment-modules, not Lmod)

Group modulefiles live at
`/projects/<NAME>/Modules/modulefiles-shared/<module>/<version>` as Tcl
`#%Module` files. **Gotcha (verified):** the login-shell auto-discovery
(`/etc/profile.d/z_add_group_space.sh`) probes `/projects/<GROUP>` with the
unix group name uppercased — `kolemen` → `/projects/KOLEMEN` — but this
group's space is `/projects/EKOLEMEN`, so discovery never fires. Every user
needs the one-time line in `~/.bashrc`:

```bash
module use --append /projects/EKOLEMEN/Modules/modulefiles-shared
```

`deploy/princeton/setup.sh --bashrc` appends it for you (once).

## GPUs

| where | hardware | notes |
|---|---|---|
| stellar-vis1 / stellar-vis2 | 2× V100S 32 GB (sm_70) | interactive; X11 works; CUDA driver 13.0 |
| Slurm `gpu` partition | A100 (`gpu:a100:2` ×6 nodes, `gpu:a100:8` ×1) | batch jobs (`princeton-batch` default) |

sm_70 (V100S) means the **`torch<2.11` pin stays required** — torch 2.11
dropped sm_70 kernels. The pin already lives in `pyproject.toml`.

## Filesystems

- `/projects/EKOLEMEN` — group space (setgid `kolemen`), **not swept**: the
  shared checkout + venv + modulefile live here. No self-heal machinery needed
  (unlike Omega's 32-day `/cscratch` sweep that motivated `deploy/omega/`'s
  ensure-env).
- `/scratch/gpfs/EKOLEMEN/foundation_model` — the shot archive: ~17k
  `{shot}_processed.h5` (shots ≈185601–204999, ~1.9 GB each). Not backed up.
  Per file: one HDF5 group per signal, `xdata` float32 `(N,)` seconds +
  `ydata` float32 `(C, N)`; no attributes; absent signal = shape `(1,)`
  placeholder; `tangtv` is 4-D camera video (not a channel signal). Channel
  row identity is **not recorded** → pointnames are `group/index`, and
  mode-number (modespec) analysis is not possible from this source.
- `/scratch/gpfs/$USER` — per-user scratch for batch outputs
  (`TOKEYE_RUNS_DIR`, set by the module). Not backed up.

### The float32 time-base trap (measured)

`xdata` is float32; at `|t| ≈ 4 s` its ULP is ~0.5 µs, so per-sample diffs of
a 500 kHz signal quantize to ULP multiples and a median-diff sampling rate
comes out **524288 Hz instead of 500000 Hz** (~4.9 % off). `FoundationSource`
derives `dt` from the float64 endpoints and synthesizes the ms axis;
`tests/test_sources_foundation.py` locks this in. Don't "simplify" it back.

## Compute-node network

Batch (`gpu` partition) nodes have **no internet**. `setup.sh` prefetches the
model weights into a shared `HF_HOME` under `/projects/EKOLEMEN`, and the
`princeton-batch` job body exports `HF_HUB_OFFLINE=1` so a job never stalls on
a Hugging Face call. Login and vis nodes do have internet (uv sync, downloads).

## X11 / Qt

`ssh -X` (or `-Y`) into stellar-vis1/2 works for the PySide6 GUI; the module
sets `QT_QPA_PLATFORM=xcb`. Debug a blank window with
`QT_DEBUG_PLUGINS=1 tokeye gui --self-test`. The anaconda modules export
`PYTHONPATH`, which poisons the uv venv — the tokeye module clears it (reload
your anaconda module after `module unload tokeye` if you need it back).

## Tooling

`uv` (0.11+) is available on the cluster; the deploy uses a plain uv venv
inside the repo (no conda). Python floor is 3.13 (matches `pyproject.toml`).

## Env-var contract (set by the modulefile)

| var | value | consumer |
|---|---|---|
| `TOKEYE_SOURCE` | `foundation` | source factory (redundant with the branch default; explicit for clarity) |
| `TOKEYE_FOUNDATION_DIR` | the archive | `FoundationSource` |
| `HF_HOME` | shared weights cache | model download/load |
| `TOKEYE_RUNS_DIR` | `/scratch/gpfs/$USER/tokeye/runs` | output defaults |
| `TOKEYE_MODULE_DIR` | the group modulefiles root | `princeton-batch` job body |
| `TOKEYE_SLURM_PARTITION/GRES/TIME` | `gpu` / `gpu:a100:1` / `0-02:00:00` | `princeton-batch` defaults |
| `QT_QPA_PLATFORM` | `xcb` | native GUI |
