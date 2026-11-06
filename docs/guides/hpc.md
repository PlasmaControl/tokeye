# Batch processing on HPC

On a login node with internet access, cache the weights before submitting a job:

```bash
tokeye download big_tf_unet
```

On the compute node, use the same Hugging Face cache and run the CLI, for example:

```bash
tokeye run "shots/*.npy" --output-dir results --device auto
```

Inputs with the same filename stem must be processed separately. Use `--key` to select a dataset inside a container and `--fs` to specify its sampling rate when it is not recorded in the file. Run `tokeye run --help` for all options. For the app on a remote node, use SSH port forwarding as described in the [app guide](app.md).
