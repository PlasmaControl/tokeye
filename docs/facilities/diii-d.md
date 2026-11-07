# DIII-D National Fusion Facility

The [DIII-D National Fusion Facility](https://www.ga.com/magnetic-fusion/diii-d) is a U.S. Department of Energy user facility operated by General Atomics. Its tokamak and diagnostics support research on magnetically confined fusion plasmas.

TokEye was developed for DIII-D diagnostic signals. The project identifies fast magnetics, CO2 interferometry, electron cyclotron emission, and beam emission spectroscopy as DIII-D data types used with TokEye. See the [TokEye preprint](https://arxiv.org/abs/2602.20317) for research context.

## Analyze an exported signal

Obtain and export a diagnostic signal using the facility's data tools. For a one-dimensional signal stored in a supported file, run:

```bash
tokeye run shot.npy --fs 200000 --format npz --output-dir results
```

Replace `200000` with the signal's actual sampling rate in hertz. If the file records its sampling rate, `--fs` can be omitted. The output bundle contains the segmentation and its axes; see [outputs](../concepts/outputs.md) and [input formats](../guides/input-formats.md).

For ELM event detection and Alfvén-eigenmode instance detection, see the [mode-analysis guide](../guides/mode-suite.md). These analyses have model and input requirements; inspect their results in the context of the diagnostic and experiment.

TokEye does not fetch DIII-D shots directly. Follow the facility's data-access procedures to obtain the signal before running it.
