# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TokEye is a Python-based application for automatic classification and localization of fluctuating signals in plasma physics. It uses deep learning (UNet models) to detect and segment patterns in spectrograms derived from tokamak plasma diagnostic signals.

**Key Capabilities:**
- Real-time spectrogram analysis using pre-trained UNet models
- Interactive web-based UI (Gradio) for signal analysis and annotation
- Support for multiple signal transforms (STFT, wavelet)
- Fast inference: <0.5s on A100 GPU after warmup

## Development Commands

### Setup and Installation
```bash
# Install dependencies (recommended)
uv sync

# Install with dev dependencies
uv sync --extra dev

# Install with evaluation dependencies
uv sync --extra eval
```

### Running the Application
```bash
# Start the Gradio web app (default port 7860, falls back if occupied)
python -m TokEye.app

# With custom port
python -m TokEye.app --port 8888

# With Gradio share link
python -m TokEye.app --share

# Auto-open in browser
python -m TokEye.app --open

# SSH port forwarding for remote servers
ssh -L 8888:localhost:8888 user@remote_server
```

### Testing and Linting
```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src/TokEye --cov-report=html

# Run tests in parallel
pytest -n auto

# Run specific test file
pytest tests/test_specific.py

# Lint with ruff
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Format code
ruff format .

# Type checking
mypy src/TokEye
```

## Architecture

### High-Level Structure

TokEye has three main components that operate independently:

1. **Gradio Web App** (`src/TokEye/app/`) - Interactive UI with three tabs:
   - **Analyze Tab**: Load signals, apply transforms (STFT/wavelet), run model inference, visualize predictions
   - **Annotate Tab**: Manual annotation of spectrograms for creating training data
   - **Utilities Tab**: Miscellaneous tools

2. **Processing Pipeline** (`src/TokEye/processing/`) - Signal processing and inference utilities:
   - Signal transforms (STFT, wavelet decomposition)
   - Tiling/stitching for UNet processing (handles arbitrary-width spectrograms)
   - Model loading and batch inference
   - Post-processing (thresholding, object filtering, visualization)
   - LRU caching system for performance

3. **Autoprocess Pipeline** (`src/TokEye/autoprocess/`) - Batch processing for dataset creation:
   - Multi-step pipeline for generating training data from raw signals
   - Steps include: timeseries → spectrogram → filtering → thresholding → augmentation

### Key Design Patterns

**Data Flow in Analyze Tab:**
```
Raw Signal (.npy) → STFT/Wavelet Transform → Spectrogram
    → Tiling (for UNet) → Model Inference → Stitching
    → Post-processing → Visualization
```

**Tiling Strategy:**
- Spectrograms are split into height-sized square tiles (since UNet expects square inputs)
- Width is divided into overlapping/non-overlapping tiles
- Predictions are stitched back together with optional blending at overlaps
- Metadata tracks original dimensions and padding for exact reconstruction

**Model Loading:**
- Supports multiple formats: `.pt` (TorchScript), `.pt2` (PyTorch 2.0 export), `.ckpt` (Lightning checkpoints)
- Auto-detects device (CUDA/CPU)
- Performs warmup inference (10 iterations) for CUDA kernel compilation
- Models stored in `model/` directory

**State Management in Gradio:**
- Uses Gradio `gr.State()` to maintain session state (loaded model, transform parameters)
- Tabs are independent but share the same app instance
- Refresh button reloads available models/shots from disk

### Module Responsibilities

**`src/TokEye/models/`**
- `unet.py`: UNet architecture with configurable depth and channels
- `modules/nn.py`: Reusable building blocks (ConvBlock, DownBlock, UpBlock)

**`src/TokEye/processing/`**
- `transforms.py`: Signal preprocessing (preemphasis, STFT, wavelet)
- `tiling.py`: Split/stitch spectrograms for UNet processing
- `inference.py`: Model loading and batch inference
- `postprocess.py`: Thresholding, object filtering, overlay visualization
- `cache.py`: LRU-based caching system with size limits

**`src/TokEye/app/analyze/`**
- `analyze.py`: Gradio UI definition for Analyze tab
- `load.py`: File I/O, model loading, signal loading, directory scanning
- `visualize.py`: Image display and visualization helpers

**`src/TokEye/app/tabs/`**
- `annotate.py`: Gradio UI for manual annotation of spectrograms
- `utilities.py`: Miscellaneous utility tools

**`src/TokEye/autoprocess/`**
- Multi-step batch processing pipeline for dataset generation
- Each step is a separate module (e.g., `step_1a_make_timeseries.py`)
- Used for creating training data, not interactive analysis

## Data Conventions

### Directory Structure
```
TokEye/
├── data/                    # Input signals (1D numpy arrays as .npy)
│   ├── input/              # Raw signals organized by shot/signal
│   └── eval/               # Evaluation datasets
├── model/                   # Pre-trained models (.pt, .pt2, .ckpt)
├── annotations/             # Manual annotations from Annotate tab
└── src/TokEye/             # Source code
```

### Signal Format
- All signals must be **1D numpy float arrays** stored as `.npy` files
- No preprocessing required (normalization happens during transform)
- Typical length: 10k-100k samples depending on diagnostic

### Spectrogram Format
- 2D numpy arrays: `(freq_bins, time_frames)`
- STFT: log-compressed magnitude with mean-std normalization
- Wavelet: log-compressed coefficients with shape `(2^level, coeffs_per_node)`

### Model Input/Output
- **Input**: `(batch_size, 1, height, width)` - single-channel spectrograms
- **Output**: `(batch_size, 1, height, width)` - sigmoid probabilities for segmentation
- Height and width must be divisible by 2^(num_layers-1) for UNet compatibility

### Annotation Files
- Saved in `annotations/` directory
- Naming convention: `{original_filename}_mask.npy` or `.png`
- Binary masks (0/1) or multi-class labels

## Important Implementation Notes

### STFT Processing
The STFT pipeline in `transforms.py` applies these steps in order:
1. Compute STFT using scipy.signal.stft
2. Take magnitude
3. Apply log1p compression: `log(1 + x)`
4. Optionally clip DC component (bottom frequency bin)
5. Normalize: `(x - mean) / std`

This differs from standard STFT and must be matched during training/inference.

### UNet Architecture
- Symmetric encoder-decoder with skip connections
- Configurable depth (`num_layers`) and base filters (`first_layer_size`)
- Filters double at each downsampling level
- Uses LeakyReLU activation and batch normalization
- Returns tuple `(logits,)` for compatibility with loss functions

### Port Selection Strategy
The app tries the specified port (default 7860) and falls back by decrementing if occupied, attempting up to 10 times. This allows multiple instances on the same machine.

### Caching System
- LRU eviction based on both total size (MB) and entry count
- Separate cache types: 'spectrogram', 'inference', 'wavelet', 'general'
- Cache keys generated via SHA256 hash of data + parameters
- Metadata stored in JSON for persistence across sessions

### Model Warmup
Models undergo 10 warmup iterations on dummy input to:
- Compile CUDA kernels (first-run overhead on GPU)
- Allocate memory pools
- Optimize runtime performance

This is critical for achieving <0.5s inference times.

## Remote Development

This project is designed for HPC/remote server usage:

1. **SSH Port Forwarding**: Use `-L` flag to access Gradio UI locally
2. **No Browser Required**: App runs headless by default
3. **Shared Models**: Models in `model/` can be shared across users
4. **Data Management**: Keep large datasets in `data/` directory structure

## Pre-trained Models

Download from: https://drive.google.com/drive/folders/1rXllPXB3eWhMvSIlp0CDSFx68lJOQG1u?usp=drive_link

Place in `model/` directory. Supported formats:
- `.pt` - TorchScript (recommended for production)
- `.pt2` - PyTorch 2.0 export format
- `.ckpt` - Lightning checkpoints (requires specific Module class)

## Citation

This is a work in progress. For questions, contact: nathaniel@princeton.edu

**Poster**: See `assets/aps_dpp_2025.pdf` for technical details from APS DPP 2025.
