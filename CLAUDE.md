# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TokEye is an open-source Python application for automatic classification and localization of fluctuating signals in plasma physics. It uses deep learning models (primarily U-Net variants) to detect coherent and transient events in time-frequency spectrograms.

The application provides:
- A Gradio-based web interface for signal analysis
- Pre-trained models for event detection
- Training pipeline for custom models
- Support for multiple plasma physics diagnostic datatypes (Fast Magnetics, CO2 Interferometer, ECE, BES)

## Development Commands

### Package Management
This project uses `uv` for dependency management (Python 3.13+):

```bash
# Install dependencies
uv sync

# Install with dev dependencies (includes pytest, ruff, mypy, jupyter)
uv sync --all-extras

# Install with training dependencies
uv sync --extra train
```

### Running the Application
```bash
# Start the web app (default port 7860)
python -m tokeye.app

# Start with custom port
python -m tokeye.app --port 8888

# Start with sharing enabled
python -m tokeye.app --share

# Auto-open in browser
python -m tokeye.app --open
```

The app will try multiple ports if the default is occupied, decrementing from the specified port.

### Testing and Quality

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/tokeye

# Run tests in parallel
pytest -n auto

# Lint with ruff
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Type checking
mypy src/tokeye
```

### Training Pipeline Scripts

Training scripts are located in `src/tokeye/training/big_tf_unet/` and follow a numbered sequence:

```bash
# Example: Run a training step with config
python -m tokeye.training.big_tf_unet.step_1a_make_timeseries path/to/config.yaml
```

Each step processes data sequentially with configurable input/output directories.

## Architecture

### Application Structure

The codebase is organized into several major components:

**1. Web Application (`src/tokeye/app/`)**
- `__main__.py`: Entry point for Gradio web interface
- `tabs/`: UI tabs (annotate, utilities)
- `analyze/`: Main analysis tab with signal loading, model inference, and visualization
  - `load.py`: Signal/model loading and inference orchestration
  - `visualize.py`: Spectrogram visualization with multiple view modes
  - `transforms.py`: STFT and signal preprocessing
- `processing/`: Core inference engine
  - `inference.py`: Model loading and batch inference
  - `tiling.py`: Breaks large spectrograms into tiles for processing
  - `postprocess.py`: Reassembles tiled predictions

**2. Models (`src/tokeye/models/`)**
- `big_tf_unet/`: Primary U-Net model architecture
- `ae_tf_maskrcnn/`, `ae_tf_boxrcnn/`: Alternative detection architectures
- `modules/`: Reusable neural network components

Models expect input tensors of shape `(B, 1, H, W)` and output `(B, 2, H, W)`:
- Channel 0: Coherent activity detection
- Channel 1: Transient activity detection

**3. Training Pipeline (`src/tokeye/training/big_tf_unet/`)**
Sequential data processing steps (step_0x, step_1x, etc.):
- Data conversion and filtering
- Spectrogram generation
- Feature extraction
- Model training
- Prediction conversion

Each step uses:
- `utils/configuration.py`: YAML config loading and directory setup
- `utils/parmap.py`: Parallel processing utilities
- `utils/hdf5_io.py`: HDF5 data handling

**4. Analysis Tools (`src/tokeye/analysis/`)**
- `batch_analysis.py`: Batch processing capabilities

**5. Extra Utilities (`src/tokeye/extra/`)**
- `eval/`: Evaluation scripts for different datasets (DIII-D, etc.)
- `D3D/`: DIII-D specific utilities

### Key Data Flow

1. **Signal Loading**: Raw 1D numpy arrays → STFT transform → normalized spectrogram
2. **Preprocessing**: Spectrograms are standardized (mean=0, std=1) with optional clipping
3. **Tiling**: Large spectrograms split into overlapping tiles for inference
4. **Inference**: Model processes tiles, outputs detection masks
5. **Postprocessing**: Tiles reassembled, results visualized in multiple modes (Enhanced, Mask, Amplitude)

### Important Conventions

- **Spectrogram Orientation**: Lowest frequency at bottom when plotted with `origin='lower'`
- **Data Format**: Keep signals as 1D numpy float arrays in `data/` directory
- **Models**: Pre-trained models go in `model/` directory (download separately)
- **Multi-scale Processing**: big_mode_v2.pt supports variable window/hop sizes

## Configuration

### Ruff Linting Rules
The project enforces several linting categories:
- Import sorting (I)
- Python upgrade suggestions (UP)
- Comprehension best practices (C4)
- Future annotations (FA)
- Return statement practices (RET)
- Type checking imports (TC)
- Pathlib over os.path (PTH)
- NumPy-specific conventions (NPY)

Exceptions:
- E501 (line length) ignored - relying on formatter
- `__init__.py` files can have unused imports (F401)

### Coverage Configuration
Coverage runs on `src/tokeye` with exclusions for:
- Test files
- `__pycache__`
- `autoprocess/` directory

## Notes for Development

- The app uses Gradio state management extensively for passing data between UI components
- Model inference supports both JIT-compiled (.pt) and exported (.pt2) PyTorch models
- Signal directory structure expects subdirectories for different shots/experiments
- The training pipeline is designed for reproducible, step-by-step data processing
- All training steps support configuration via YAML files passed as command-line arguments
