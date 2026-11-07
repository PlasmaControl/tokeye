# Princeton Plasma Physics Laboratory

The [Princeton Plasma Physics Laboratory (PPPL)](https://www.pppl.gov/research) conducts research in plasma science and fusion energy. Its experiments include the National Spherical Torus Experiment-Upgrade (NSTX-U), and PPPL researchers also collaborate on experiments at other facilities, including DIII-D. See [PPPL's experiment overview](https://www.pppl.gov/experiments) for more information.

## Use TokEye with exported data

TokEye accepts one-dimensional time series and two-dimensional spectrograms regardless of their source. Export a diagnostic signal through the appropriate facility data tools and supply the sampling rate in hertz when it is not included in the file:

```bash
tokeye run signal.npy --fs 200000 --format npz --output-dir results
```

Replace `200000` with the actual sampling rate. See [input formats](../guides/input-formats.md) for supported files and [preprocessing](../concepts/preprocessing.md) for the treatment of time series and spectrograms.

TokEye's documented development uses DIII-D diagnostics. The ability to read PPPL data does not by itself establish model performance on a particular PPPL experiment or diagnostic; evaluate results against suitable reference data for the intended use. TokEye does not connect directly to PPPL data systems.
