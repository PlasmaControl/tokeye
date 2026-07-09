# DIII-D Tabs — Performance / Timing Reference

Where the time goes in the DIII-D tabs, so future work can target the slow parts.
All numbers measured on **omega14** (`somega`, CPU-only, atlas thin client),
2026-07, with the `big_tf_unet` model. `(M)` = measured this session, `(E)` =
estimated/extrapolated. Times vary with shot digitizer rate, window length, and
atlas load — treat them as order-of-magnitude.

## Per-step costs

| Step | Time | Cold/warm | Notes |
|---|---|---|---|
| **Fresh probe fetch** (atlas) | **~8 s / probe** (M) | cold | 5 M points @ 500 kHz. Full signal always transferred (see gotcha below). |
| Cached probe fetch (pickle) | **0.17 s / probe** (M) | warm | `$TOKEYE_CACHE` hit — no atlas. |
| **14-probe array fetch** (modespec) | **~2 min** (E) | cold | 14 × fresh fetch, sequential. |
| 14-probe array fetch, cached | ~2–3 s (E) | warm | 14 × cache hit. |
| `time_bounds` (window autofill) | **~8 s** (M) first, **instant** cached (M) | — | Builds the server-side time base once; cached in-process. |
| STFT (`compute_stft`, 1 probe) | **0.02 s** (M) | — | Negligible. |
| Model download (`big_tf_unet`, HF) | ~5–30 s (E) | one-time | ~30 MB, cached in HF cache thereafter. |
| **Model load + 10× warmup** (CPU) | **~19 s** (M) | per process | Warmup dominates; fixed cost once per app/job. |
| `model_infer` (1 probe, CPU) | **~1.1 s** (M) | — | — |
| **TokEye array gate** (14 inferences) | **~15 s** (E) | — | 14 × `model_infer`, sequential (CPU). |
| `create_app` (build 6 tabs) | **0.23 s** (M) | — | No model/network at build. |
| `import torch + gradio` | **~5 s** (M) warm | app startup | Cold (first import after env build) is longer. |
| Modespec compute (`mode_spectrogram`) | ~1–5 s (E) | — | After auto-decimation to ~2·f_max. |
| Render (`plotly_modespec` / `render_modespec_png`) | <1 s (M) | — | Figure build + `to_json` (~1.9 MB) / matplotlib PNG. |

## The two interactive flows

**DIII-D tab — Load shot → Analyze** (single probe):
- Cold: fetch (~8 s) + STFT (instant) + model load/warmup (~19 s, first Analyze) +
  1 inference (~1 s) ≈ **~30 s**.
- Warm (probe cached, model loaded): fetch (0.17 s) + render ≈ **~1 s**. Slider
  re-renders are pure re-color, **instant** (no recompute/re-infer).

**DIII-D Modespec tab — Analyze (gated)**:
- Cold: 14-probe fetch (**~2 min**) + modespec (~few s) + model load/warmup (~19 s)
  + 14-probe gate (~15 s) ≈ **~2.5–3 min**.
- Warm (probes cached, model loaded): **~40 s** (measured 43 s) — dominated by the
  ~19 s warmup + ~15 s gate.
- Coherence-threshold slider: re-renders from the cached result + gate mask,
  **instant** (no recompute).

## Optimization opportunities (roughly ranked)

1. **Probe fetch ignores the time window.** `data_utils.fetch_ptdata` transfers the
   **whole** probe signal, then `MDSSource.fetch` crops to `t-min/t-max` *after*.
   So narrowing the window speeds up compute but **not** the fetch — the ~8 s/probe
   (×14 = ~2 min) is paid regardless. Biggest win: slice server-side in the TDI
   expression (fetch only `[t_min, t_max]`), which would make short windows fetch in
   a fraction of the time. (Would need care to keep the pickle cache keyed by window,
   or cache full + slice.)
2. **Cache is the big lever, and it's volatile.** Warm fetches are ~50× faster
   (0.17 s vs 8 s). But `$TOKEYE_CACHE` is on `/cscratch`, swept after ~32 days — so
   the "cold" path recurs. A durable cache home (`/fusion/projects/...`) would keep
   fetches warm across weeks.
3. **Model warmup is a fixed ~19 s.** 10 iterations on `(1,1,512,512)`. On CPU this
   is pure overhead before the first real inference; fewer warmup iterations (or
   skipping warmup when `device=cpu`) would cut first-Analyze latency.
4. **The gate runs 14 inferences sequentially (~15 s CPU).** They're independent and
   equal-sized — batching them into one forward pass, or running on a GPU
   (`gpus` partition), would cut this substantially. (Per-probe `model_infer`
   normalizes per-image, so batching must preserve per-image normalization.)
5. **App startup imports torch (~5 s warm, more cold).** Unavoidable-ish; only paid
   once per server.

## One-time / setup costs

- **Env build** (`deploy/omega/README.md` step 1): `mamba env create` ~1–2 min;
  `pip install .[app] gradio==5.49.1` (downloads torch ~2 GB + gradio + plotly)
  ~5–8 min. Rebuild is only needed after a `/cscratch` sweep (see gotcha).

## Gotchas

- **`/cscratch` sweep.** Files older than ~32 days are swept — this has already
  destroyed the env once (broken `libstdc++`/BLAS symlinks → numpy import fails) and
  wipes the shot cache. The durable fix is `/fusion/projects/codes/tokeye`
  (see `deploy/omega/README.md`).
- **`LD_LIBRARY_PATH` import order.** The conda env's compiled libs must win over the
  node's system `/lib64` (old `libstdc++`, no `GLIBCXX_3.4.29`). Running the env
  python with `import torch` *first* and no `LD_LIBRARY_PATH` can load the system
  `libstdc++` and then break numpy. Set `LD_LIBRARY_PATH=$TOKEYE_DIR/lib` (what conda
  activation does) for reliability — worth adding to the launcher/modulefile.

## How these were measured

`time.perf_counter()` around each call in the rebuilt env (`LD_LIBRARY_PATH` set to
the env lib, `PYTHONPATH` cleared). Fetch/`time_bounds` used a fresh scratch cache
dir for the cold number and a warm re-call for the cache-hit number. Shot 208270
(`MPI66M067D`, 5 M pts @ 500 kHz) for fetch; shot 190904 for the warm modespec+gate
flow. Re-measure with the same pattern after hardware/model changes.
