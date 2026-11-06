# Models and weights

| Registry name | HF repo | HF file | Description |
| --- | --- | --- | --- |
| `big_tf_unet` | [`nc1/big_tf_unet`](https://huggingface.co/nc1/big_tf_unet) | `big_tf_unet_251210.pt` | Convolutional U-Net trained on multiscale (multiwindow, multihop) spectrograms. |
| `ae_tf_maskrcnn` | `nc1/ae_tf_maskrcnn` | `ae_tf_maskrcnn_251223.pt` | Mask R-CNN instance detector for Alfvén-eigenmode activity (used by `tokeye alfvenspec`). |

Weights download automatically the first time a registry name is used (cached in `~/.cache/huggingface`). Override the default repo with the `TOKEYE_HF_REPO` environment variable (per-model repos are fixed in the registry).

To use a local checkpoint instead, put `.pt`/`.pt2` files in a `model/` directory (picked up by the app's model dropdown) or pass a path directly via `--model PATH`.

Input should be a tensor that has shape (B, 1, H, W) where B, H, and W can vary
Output will be a tensor of shape (B, 2, H, W)

Best performance when spectrograms are oriented so that when they are plotted with matplotlib, the lowest frequency bin is oriented with the bottom when `origin='lower'`. TokEye standardizes the model input internally. If baseline activity is very strong, clipping the input may help, but is generally not needed.

The first channel of the output will return preferential measurements of coherent activity (useful for most tasks)
The second channel of the output will return preferential measurements of transient activity
