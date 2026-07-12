# Princeton Tab — User Guide

The **Princeton** tab in the web app (`tokeye app`) loads DIII-D shots from
the local foundation_model archive on stellar
(`/scratch/gpfs/EKOLEMEN/foundation_model`, ~17k shots) and overlays TokEye's
mode mask — same pipeline as the DIII-D tab on the `diiid` branch, but reading
GPFS directly: no MDSplus, no network, no cache step.

## Quick Start

1. On stellar-vis1/2: `module load tokeye && tokeye app`, then from your
   laptop `ssh -N -L 7860:localhost:7860 <netid>@stellar-vis1.princeton.edu`
   and browse `http://localhost:7860`.
2. The **Shot** field prefills with the newest shot in the archive. Pick a
   **signal group** (mirnov, ece, sxr, …) and a **channel** — the channel list
   refreshes to what that shot's file actually carries.
3. **t-min/t-max** auto-fill to the signal's data window (times can start
   negative, e.g. −4336 ms — that's real, the digitizers run before t=0).
   Narrow the window to load faster.
4. **Load shot** → interactive spectrogram (zoom/pan, real kHz/ms axes).
5. **Analyze** → coherent/transient mask overlay. Threshold / clip / band
   sliders re-render live without recomputing.
6. **Save results (.npz)** → an analysis bundle
   (`<shot>_<group>-<index>_analysis.npz`, `source='princeton'`) with real
   `time_ms`/`freq_khz` axes.

## Pointnames: `group/index` (and why)

The archive stores each signal group's channels as rows of a `(C, N)` array in
sorted-original-name order — but the original names were never recorded. So a
channel is addressed by its **row index**: `mirnov/07` is row 7 of the mirnov
group. Consequences:

- Which physical probe `mirnov/07` is, is unknown. Fine for spectrograms and
  masks; **not** enough for toroidal mode numbers, so there is no Modespec
  tab/view on this branch.
- Channel counts vary by group (mirnov 29, ece 48, sxr 320, …) and
  availability varies by shot (`co2` only ≳197965, `bes` rare). Empty groups
  are stored as 1-sample placeholders — the tab tells you when a signal is
  absent for the shot.

## Signal groups worth knowing

| group | channels | rate | notes |
|---|---|---|---|
| `mirnov` | 29 | 500 kHz | the mode-detection workhorse; always present |
| `mhr` | 8 | ~1 MHz | high-resolution magnetics |
| `ece` | 48 | ~200 kHz | electron cyclotron emission |
| `sxr` | 320 | 10 kHz | soft X-ray |
| `co2` | 4 | — | interferometer; shots ≳197965 only |
| `bes` | 64 | — | rare |

Other groups in a shot's file (bolo, langmuir, i_coil, …) appear in the
dropdown automatically. 4-D video groups (`tangtv`) are excluded.

## Batch (CLI)

```bash
# Submit one Slurm job on the A100 gpu partition (writes outdir/submit.sh):
tokeye princeton-batch --shots 190000-190010 --probe mirnov/07 \
    --outdir /scratch/gpfs/$USER/tokeye/run1

# Same work, right now, on the vis node's V100S:
tokeye princeton-batch --local --shots 190000 --outdir /scratch/gpfs/$USER/tokeye/smoke

# Inspect the job script without submitting:
tokeye princeton-batch --dry-run --shots 190000 --outdir /tmp/x
```

Outputs per shot: `inputs/<shot>_<group>-<index>.npy` (the channel),
`<shot>_<group>-<index>_mask.npy` (float32 `(2,H,W)`), and a preview PNG.
Exit code = number of failed shots.

`tokeye fetch --shot 190000 --diag mirnov --pointname mirnov/07` also works on
this branch (it routes through the same source) if you just want the `.npy`.

## Troubleshooting

- **"no foundation_model file for shot N"** — the shot isn't in the archive
  (range ≈185601–204999, with gaps). The error names the available range.
- **"has no samples (signal absent for this shot)"** — the group is a
  placeholder in this shot's file (common for `co2`/`bes`). Pick another
  group or shot.
- **Wrong-looking sampling rate?** It isn't: the tab derives fs from float64
  time-base endpoints because the stored float32 axis quantizes (see
  `docs/princeton-cluster.md`). Mirnov is 500 kHz.
- **Slow first Analyze** — the model downloads (~30 MB) on first use; with the
  group module loaded it's already in the shared `HF_HOME`.
