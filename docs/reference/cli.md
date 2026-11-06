# Command-line reference

Generated from the TokEye argument parser.

## `tokeye`

```text
usage: tokeye [-h] [-v] [--version] COMMAND ...

Automatic classification and localization of fluctuating signals in
spectrograms.

positional arguments:
  COMMAND
    run          Segment one or more inputs (files, directories or globs).
    app          Launch the TokEye web app in your browser.
    download     Download model weights into the Hugging Face cache.
    example      Write a synthetic example signal to a .npy file.
    info         Show versions, the device auto picks, extras and cached
                 models.
    elmspec      Detect ELM events (transient-channel intervals, count,
                 frequency).
    alfvenspec   Detect Alfvén-eigenmode activity (boxes + masks via
                 ae_tf_maskrcnn).

options:
  -h, --help     show this help message and exit
  -v, --verbose  show debug logs on stderr, including tracebacks
  --version      show program's version number and exit
```

## `tokeye run`

```text
usage: tokeye run [-h] [-v] [--model MODEL] [--device DEVICE]
                  [--tile auto|none|N] [--output-dir OUTPUT_DIR]
                  [--png | --no-png] [--fs FS] [--key NAME]
                  [--format {npy,npz}] [--threshold THRESHOLD] [--n-fft N_FFT]
                  [--hop HOP] [--window WINDOW] [--clip-dc | --no-clip-dc]
                  [--clip-low CLIP_LOW] [--clip-high CLIP_HIGH]
                  [--log | --no-log]
                  INPUT [INPUT ...]

Segment each input and write <stem>_mask.npy (or <stem>_tokeye.npz with
--format npz), <stem>_preview.png and <stem>_params.json.

positional arguments:
  INPUT                 files (.npy .npz .wav .flac .ogg .csv .txt .mat .h5),
                        directories, or glob patterns

options:
  -h, --help            show this help message and exit
  -v, --verbose         show debug logs on stderr, including tracebacks
  --model MODEL         registry name or path to a .pt/.pt2 checkpoint
                        (default: big_tf_unet)
  --device DEVICE       cpu, cuda, cuda:N, mps or auto; auto = CUDA, then MPS,
                        then CPU
  --tile auto|none|N    tile side in pixels for long inputs: auto (untiled up
                        to 2^21 pixels), none, or an int >= 512; for
                        big_tf_unet, tiled output matches untiled output to
                        float32 rounding
  --output-dir OUTPUT_DIR
                        directory to write outputs to (default: tokeye_output)
  --png, --no-png       write a mask-overlay preview PNG per input
  --fs FS               sampling rate [Hz] for every input; enables physical
                        time and frequency axes (default: read from the file
                        when recorded)
  --key NAME            array to read from .npz/.mat/.h5 inputs: an entry
                        name, or an HDF5 dataset path or its trailing path
                        components, such as tree/pointname/data (default: the
                        one named data, signal, x, spectrogram, values or y,
                        else the only numeric one)
  --format {npy,npz}    npy = mask only; npz = mask + spectrogram + axes
                        bundle (default: npy)
  --threshold THRESHOLD
                        preview threshold, in [0, 1] (default: 0.5)

spectrogram options:
  --n-fft N_FFT         STFT window length [samples] (default: 1024)
  --hop HOP             STFT hop [samples] (default: 128)
  --window WINDOW       STFT window name; checked before any input is read,
                        even when every input is 2D (default: hann)
  --clip-dc, --no-clip-dc
                        drop the DC row (default: True)
  --clip-low CLIP_LOW   lower clip percentile (default: 1.0)
  --clip-high CLIP_HIGH
                        upper clip percentile (default: 99.0)
  --log, --no-log       log1p 2D inputs stored in linear scale (default:
                        False)
```

## `tokeye app`

```text
usage: tokeye app [-h] [-v] [--host HOST] [--port PORT] [--share] [--open |
                  --no-browser] [--workspace DIR]

Serve the app on http://HOST:PORT (the next free port if PORT is busy). On a
remote machine, forward the port over SSH instead of using --share.

options:
  -h, --help       show this help message and exit
  -v, --verbose    show debug logs on stderr, including tracebacks
  --host HOST      interface to bind; 0.0.0.0 exposes the app to your network
                   (default: $GRADIO_SERVER_NAME if set, else 127.0.0.1)
  --port PORT      first port to try (default: 7860)
  --share          also create a public gradio.live link (anyone with it can
                   use the app)
  --open           open a browser tab (default when not in an SSH session)
  --no-browser     do not open a browser tab
  --workspace DIR  directory the app reads and writes files in (created if
                   missing; default: the current directory)
```

## `tokeye download`

```text
usage: tokeye download [-h] [-v] [MODEL ...]

Download model weights ahead of time, e.g. on an HPC login node before running
on an offline compute node.

positional arguments:
  MODEL          model registry name(s) (default: big_tf_unet)

options:
  -h, --help     show this help message and exit
  -v, --verbose  show debug logs on stderr, including tracebacks
```

## `tokeye example`

```text
usage: tokeye example [-h] [-v] [--output OUTPUT] [--duration DURATION]
                      [--fs FS] [--seed SEED]

options:
  -h, --help           show this help message and exit
  -v, --verbose        show debug logs on stderr, including tracebacks
  --output OUTPUT      output path (default: tokeye_example_sr<FS>.npy)
  --duration DURATION  length [s] (default: 2.0)
  --fs FS              sampling rate [Hz] (default: 200000.0)
  --seed SEED          random seed
```

## `tokeye info`

```text
usage: tokeye info [-h] [-v]

Print what tokeye sees on this machine. Paste the output into bug reports.
Exits 1 if a check fails (a broken torch, say); a missing extra or an uncached
model is information, not a failure.

options:
  -h, --help     show this help message and exit
  -v, --verbose  show debug logs on stderr, including tracebacks
```

## `tokeye elmspec`

```text
usage: tokeye elmspec [-h] [-v] [--model MODEL] [--device DEVICE]
                      [--tile auto|none|N] [--output-dir OUTPUT_DIR]
                      [--png | --no-png] [--fs FS] [--key NAME] [--dt DT]
                      [--threshold THRESHOLD] [--activity-min ACTIVITY_MIN]
                      [--min-gap-cols MIN_GAP_COLS]
                      [--min-duration-cols MIN_DURATION_COLS] [--n-fft N_FFT]
                      [--hop HOP] [--window WINDOW] [--clip-dc | --no-clip-dc]
                      [--clip-low CLIP_LOW] [--clip-high CLIP_HIGH]
                      [--log | --no-log]
                      INPUT [INPUT ...]

Segment each input, then turn broadband stripes in the transient channel into
ELM events. Writes elm_events.csv and elm_summary.csv (plus
<stem>_elm_preview.png with --png). Event times are column times: column j is
j*hop/fs seconds after the first sample (j*dt with --dt).

positional arguments:
  INPUT                 files, directories, or glob patterns

options:
  -h, --help            show this help message and exit
  -v, --verbose         show debug logs on stderr, including tracebacks
  --model MODEL         registry name or path to a .pt/.pt2 checkpoint
                        (default: big_tf_unet)
  --device DEVICE       cpu, cuda, cuda:N, mps or auto; auto = CUDA, then MPS,
                        then CPU
  --tile auto|none|N    tile side in pixels for long inputs: auto (untiled up
                        to 2^21 pixels), none, or an int >= 512; for
                        big_tf_unet, tiled output matches untiled output to
                        float32 rounding
  --output-dir OUTPUT_DIR
                        directory to write outputs to (default: tokeye_elms)
  --png, --no-png       write a mask-overlay preview PNG per input
  --fs FS               sampling rate [Hz] for every input; enables physical
                        time and frequency axes (default: read from the file
                        when recorded)
  --key NAME            array to read from .npz/.mat/.h5 inputs: an entry
                        name, or an HDF5 dataset path or its trailing path
                        components, such as tree/pointname/data (default: the
                        one named data, signal, x, spectrogram, values or y,
                        else the only numeric one)
  --dt DT               seconds per spectrogram column; overrides --hop/--fs
                        (use it for 2D inputs whose columns are not --hop
                        samples apart) and sets the duration used for the ELM
                        frequency (columns x dt)
  --threshold THRESHOLD
                        mask binarization threshold, in [0, 1] (default: 0.5)
  --activity-min ACTIVITY_MIN
                        minimum fraction of active frequency bins for a time
                        column to belong to an ELM, in [0, 1] (default: 0.1)
  --min-gap-cols MIN_GAP_COLS
                        merge events separated by at most this many columns,
                        >= 0 (default: 3)
  --min-duration-cols MIN_DURATION_COLS
                        drop events shorter than this many columns, >= 1
                        (default: 1)

spectrogram options:
  --n-fft N_FFT         STFT window length [samples] (default: 1024)
  --hop HOP             STFT hop [samples] (default: 128)
  --window WINDOW       STFT window name; checked before any input is read,
                        even when every input is 2D (default: hann)
  --clip-dc, --no-clip-dc
                        drop the DC row (default: True)
  --clip-low CLIP_LOW   lower clip percentile (default: 1.0)
  --clip-high CLIP_HIGH
                        upper clip percentile (default: 99.0)
  --log, --no-log       log1p 2D inputs stored in linear scale (default:
                        False)
```

## `tokeye alfvenspec`

```text
usage: tokeye alfvenspec [-h] [-v] [--model MODEL] [--device DEVICE]
                         [--output-dir OUTPUT_DIR] [--key NAME]
                         [--score-min SCORE_MIN] [--window-cols WINDOW_COLS]
                         [--mean MEAN] [--std STD] [--masks | --no-masks]
                         [--n-fft N_FFT] [--hop HOP] [--window WINDOW]
                         [--clip-dc | --no-clip-dc] [--clip-low CLIP_LOW]
                         [--clip-high CLIP_HIGH] [--log | --no-log]
                         INPUT [INPUT ...]

Run the ae_tf_maskrcnn instance model (needs the 'ae' extra: pip install
"tokeye[ae]"). Writes ae_detections.csv and, unless --no-masks,
<stem>_ae_instances.npy: an (H, W) int32 map where i + 1 marks detection i of
that input. Unwindowed inputs with detections also get <stem>_ae_masks.npy,
the per-detection soft masks (deprecated; no longer written from 2.0).

positional arguments:
  INPUT                 files, directories, or glob patterns

options:
  -h, --help            show this help message and exit
  -v, --verbose         show debug logs on stderr, including tracebacks
  --model MODEL         registry name or path to a .pt/.pt2 checkpoint
                        (default: ae_tf_maskrcnn)
  --device DEVICE       cpu, cuda, cuda:N, mps or auto; auto = CUDA, then MPS,
                        then CPU
  --output-dir OUTPUT_DIR
                        directory to write outputs to (default: tokeye_ae)
  --key NAME            array to read from .npz/.mat/.h5 inputs: an entry
                        name, or an HDF5 dataset path or its trailing path
                        components, such as tree/pointname/data (default: the
                        one named data, signal, x, spectrogram, values or y,
                        else the only numeric one)
  --score-min SCORE_MIN
                        keep detections with at least this score, in [0, 1]
                        (default: 0.5)
  --window-cols WINDOW_COLS
                        process wide spectrograms in windows of this many
                        columns (the default is the training width): 0
                        disables windowing, else >= 32 (default: 710)
  --mean MEAN           standardization mean, a finite number (default: per-
                        input statistics)
  --std STD             standardization std (default: per-input statistics)
  --masks, --no-masks   write <stem>_ae_instances.npy per input, and the
                        deprecated <stem>_ae_masks.npy for unwindowed inputs
                        with detections (default: on)

spectrogram options:
  --n-fft N_FFT         STFT window length [samples] (default: 1024)
  --hop HOP             STFT hop [samples] (default: 128)
  --window WINDOW       STFT window name; checked before any input is read,
                        even when every input is 2D (default: hann)
  --clip-dc, --no-clip-dc
                        drop the DC row (default: True)
  --clip-low CLIP_LOW   lower clip percentile (default: 1.0)
  --clip-high CLIP_HIGH
                        upper clip percentile (default: 99.0)
  --log, --no-log       log1p 2D inputs stored in linear scale (default:
                        False)
```
