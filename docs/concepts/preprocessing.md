# Preprocessing

A one-dimensional signal is transformed using an STFT, then `log1p`, DC-bin removal, and percentile clipping. The defaults use `n_fft=1024` and `hop=128`, matching the released model's training. Frames are centered at samples 0, hop, 2 × hop, and so on. With the default even FFT size, N samples give `1 + N // hop` columns.

A two-dimensional input is treated as an existing spectrogram. Set `log=True` for linear-scale values to apply `log1p`; the default assumes the input is already log-scaled. Before inference, the whole image is standardized with `(x - mean) / (std + 1e-6)`. Large inputs can be tiled after this global standardization.

When a sample rate is supplied, the axes are in hertz and seconds. Otherwise they are bin and frame indices. See the [Python guide](../guides/python.md) for configuration.
