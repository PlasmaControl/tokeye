# Outputs

The default segmentation model returns sigmoid scores in `[0, 1]`: channel 0 describes coherent activity and channel 1 describes transient activity. A score is not a binary mask; apply a threshold appropriate for the task.

`TokEye.segment()` returns a `Segmentation` with a `(C, H, W)` mask, spectrogram, axes, and metadata. Plot with the lowest frequencies at the bottom (`origin="lower"`). With a sample rate, frequency and time are measured in hertz and seconds; otherwise they are bin and frame indices.

The CLI writes a NumPy mask, a preview, and a JSON provenance file per input. `--format npz` writes a bundle containing the mask, spectrogram, and axes instead of the bare mask. Reload a bundle using `Segmentation.load()`.
