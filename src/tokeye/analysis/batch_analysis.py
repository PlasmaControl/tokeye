import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

# Import model loading function from the analyze module
from TokEye.app.analyze.load import model_load

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default settings
default_settings = {
    "model_path": "data/models/big_mode_v1.pth",
    "input_path": "data/batch_inputs",
    "output_path": "data/batch_outputs",
    "analysis_mode": "amplitude",
    "threshold": 0.5,
    "polling_interval": 5,  # seconds between directory scans
}


def load_spectrogram(filepath: Path) -> np.ndarray | None:
    """Load a 2D spectrogram from a numpy file.

    Args:
        filepath: Path to the .npy file containing a 2D spectrogram

    Returns:
        2D numpy array or None if loading fails
    """
    try:
        spec = np.load(filepath)
        if spec.ndim != 2:
            logger.error(f"Expected 2D array, got {spec.ndim}D: {filepath.name}")
            return None
        if spec.size == 0:
            logger.error(f"Empty array: {filepath.name}")
            return None
        return spec
    except Exception as e:
        logger.error(f"Failed to load {filepath.name}: {e}")
        return None


def run_inference(
    spec: np.ndarray,
    model: nn.Module | torch.export.ExportedProgram,
) -> np.ndarray | None:
    """Run inference on a 2D spectrogram.

    Args:
        spec: 2D numpy array (H, W)
        model: Loaded PyTorch model

    Returns:
        3D numpy array (2, H, W) with two output channels or None if inference fails
    """
    try:
        device = next(model.parameters()).device

        # Normalize input
        spec_norm = (spec - spec.mean()) / (spec.std() + 1e-6)

        # Convert to tensor: (1, 1, H, W)
        inp_tensor = torch.from_numpy(spec_norm)
        inp_tensor = inp_tensor.unsqueeze(0).unsqueeze(0).float()
        inp_tensor = inp_tensor.to(device)

        # Run inference
        with torch.no_grad():
            out_tensor = model(inp_tensor)

        # Remove batch dimension: (1, 2, H, W) -> (2, H, W)
        out_tensor = out_tensor[0]

        # Apply sigmoid activation
        out_tensor = torch.sigmoid(out_tensor)

        # Convert to numpy
        return out_tensor.cpu().numpy()

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return None


def process_file(
    input_path: Path,
    output_path: Path,
    model: nn.Module | torch.export.ExportedProgram,
) -> bool:
    """Process a single file: load, infer, save.

    Args:
        input_path: Path to input .npy file
        output_path: Path to output .npy file
        model: Loaded PyTorch model

    Returns:
        True if successful, False otherwise
    """
    # Load spectrogram
    spec = load_spectrogram(input_path)
    if spec is None:
        return False

    # Run inference
    result = run_inference(spec, model)
    if result is None:
        return False

    # Save result
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, result)
        return True
    except Exception as e:
        logger.error(f"Failed to save {output_path.name}: {e}")
        return False


def scan_directory(input_dir: Path) -> set[str]:
    """Scan directory for .npy files.

    Args:
        input_dir: Directory to scan

    Returns:
        Set of filenames (not full paths)
    """
    if not input_dir.exists():
        return set()

    npy_files = input_dir.glob("*.npy")
    return {f.name for f in npy_files}


def run_batch_analysis(settings: dict):
    """Main batch analysis loop.

    Continuously monitors input directory for new spectrogram files,
    runs inference, and saves results. Runs until interrupted with Ctrl+C.

    Args:
        settings: Dictionary with configuration:
            - model_path: Path to model file
            - input_path: Input directory to monitor
            - output_path: Output directory for results
            - polling_interval: Seconds between directory scans
    """
    # Extract settings
    model_path = Path(settings["model_path"])
    input_dir = Path(settings["input_path"])
    output_dir = Path(settings["output_path"])
    polling_interval = settings.get("polling_interval", 5)

    # Ensure directories exist
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("=" * 60)
    logger.info("Starting Batch Analysis System")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Polling interval: {polling_interval}s")
    logger.info("=" * 60)

    try:
        model = model_load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Initialize tracking
    processed_files = set()

    logger.info("Monitoring for new files... (Press Ctrl+C to stop)")
    logger.info("")

    # Main loop
    try:
        while True:
            # Scan for all files in directory
            current_files = scan_directory(input_dir)

            # Update tracking: remove deleted files
            deleted_files = processed_files - current_files
            if deleted_files:
                for fname in deleted_files:
                    processed_files.discard(fname)
                    logger.info(f"File deleted, will reprocess if added again: {fname}")

            # Find new files to process
            new_files = current_files - processed_files

            if new_files:
                logger.info(f"Found {len(new_files)} new file(s)")

                # Process files with progress bar
                new_files_list = sorted(new_files)
                for fname in tqdm(new_files_list, desc="Processing", unit="file"):
                    input_path = input_dir / fname
                    output_path = output_dir / fname

                    # Process the file
                    success = process_file(input_path, output_path, model)

                    if success:
                        processed_files.add(fname)
                        logger.info(f"✓ Processed: {fname}")
                    else:
                        logger.warning(f"✗ Failed: {fname}")

                logger.info("")

            # Sleep before next scan
            time.sleep(polling_interval)

    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Shutting down gracefully...")
        logger.info(f"Total files processed: {len(processed_files)}")
        logger.info("=" * 60)


if __name__ == "__main__":
    run_batch_analysis(default_settings)
