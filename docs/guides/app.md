# Web app guide

`tokeye app` (or `python -m tokeye.app`) launches a Gradio interface with three tabs:
- **Analyze** — load a signal, compute its spectrogram, run a model, and visualize the result. Guided for first-time use: the model dropdown defaults to the bundled `big_tf_unet` model, the STFT transform has working defaults, and "Load Example Signal" generates a synthetic demo signal so a brand-new user needs zero files. "Analyze" runs the whole load-model → infer → visualize pipeline in one click. View modes: Original, Enhanced (percentile-clipped amplitude), Mask (thresholded model output), Amplitude.
- **Annotate** — manually draw and save mask annotations over a read-only backdrop image.
- **Utilities** — audio-format conversion and `.npy` file inspection.

Flags: `tokeye app [--host 127.0.0.1] [--port 7860] [--open | --no-browser] [--workspace DIR] [--share]`. The app binds to `127.0.0.1` (`--host` defaults to `$GRADIO_SERVER_NAME` if set), moves to the next free port if 7860 is busy and says which port it chose, and opens a browser tab when you run it locally. `--share` creates a public Gradio link that anyone with the URL can use.

If you're on a remote server (e.g. an HPC login node), forward the port over SSH instead of using `--share`:
```bash
ssh -L 7860:localhost:7860 user@remote
```
Then open `http://localhost:7860` in your local browser.
