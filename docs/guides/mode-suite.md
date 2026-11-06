# Mode-analysis suite

Beyond segmentation, `tokeye` bundles the analyses DIII-D researchers usually reach for separate tools to get. Each is a subcommand; `--help` on any of them shows the full flags.

| Command | What it does |
| --- | --- |
| `tokeye elmspec INPUTS...` | ELM detection from the segmentation model's transient channel: per-event time intervals (column times from the first sample) plus per-shot count, ELM frequency (with `--fs` or `--dt`), and duty cycle, written to `elm_events.csv` / `elm_summary.csv`. |
| `tokeye alfvenspec INPUTS...` | Alfvén-eigenmode detection with the `ae_tf_maskrcnn` instance model (needs `tokeye[ae]`): per-detection boxes/scores (`ae_detections.csv`) and a per-input instance map (`<stem>_ae_instances.npy`, `i + 1` = detection `i`). Wide spectrograms are processed in training-width windows automatically. Unwindowed inputs also get the deprecated `<stem>_ae_masks.npy` (per-detection soft masks), no longer written from 2.0. |

`elmspec` and `alfvenspec` take `--key` as `tokeye run` does.
