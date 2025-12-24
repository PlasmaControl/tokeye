import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL.Image import Resampling
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import resize

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

L.seed_everything(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from TokEye.autoprocess.utils.augmentations import get_augmentation
from TokEye.autoprocess.utils.losses import (
    dice_coefficient,
    get_loss_function,
    iou_score,
)
from TokEye.models.unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_NFFT = 1024
BASE_HOP = 128
TRAIN_CHUNK_SAMPLES = 65536

# (nfft, hop)
SCALE_CONFIGS = [
    (128, 64),
    (256, 64),
    (256, 128),
    (512, 128),
    (512, 256),
    (1024, 128),
    (1024, 256),
    (1024, 512),
    (2048, 256),
    (2048, 512),
]

NORMALIZATION_CONFIGS = {
    "bes": {
        "mean": 0.4*2,
        "std": 0.5*2,
    },
    "co2": {
        "mean": 27.5*2,
        "std": 1.3*2,
    },
    "ece": {
        "mean": 0.38**2,
        "std": 0.5**2,
    },
    "mhr": {
        "mean": 0.5*2,
        "std": 1,
    },
}

default_settings = {
    # Data and training
    "batch_size": 8,
    "num_workers": 6,
    "prefetch_factor": 2,
    "num_layers": 5,
    "first_layer_size": 32,
    "max_epochs": 100,
    "precision": "bf16-mixed",
    "devices": 1,
    "enable_progress_bar": True,
    "ckpt_path": None,
    "log_every_n_steps": 10,
    "data_dir": Path("/scratch/gpfs/nc1514/TokEye/data/.cache/multiscale_prep"),
    "model_dir": Path("/scratch/gpfs/nc1514/TokEye/model/multiscale"),
    "pretrained_path": Path("/scratch/gpfs/nc1514/TokEye/model/big_mode_v1.ckpt"),
    "fast_dev_run": False,
    "early_stopping_patience": 10,
    # Optimizer settings
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    # Learning rate scheduler
    "lr_scheduler": "reduce_on_plateau",
    "lr_factor": 0.5,
    "lr_patience": 5,
    "lr_min": 1e-6,
    # Dropout
    "dropout_rate": 0.2,
    # Augmentation
    "augmentation": False,
    # Loss function
    "loss_type": "label_smooth_bce", # Options: 'bce', 'label_smooth_bce', 'symmetric_bce', 'dice', 'dice_bce', 'symmetric_bce_dice', 'focal', 'focal_dice', 'iou'
    "label_smoothing": 0.1,
    "symmetric_alpha": 0.1,
    "symmetric_beta": 1.0,
    "symmetric_weight": 0.5,
    "dice_weight": 0.5,
    "bce_weight": 0.5,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "focal_weight": 0.5,
    # Multiscale settings
    "scale_configs": SCALE_CONFIGS,
    "base_scale_weight": 0.3,  # Weight for base scale samples in batch
    "chunk_samples": TRAIN_CHUNK_SAMPLES,  # Samples per training chunk
}


def make_spectrogram(
    data: torch.Tensor,
    window_size: int,
    hop_size: int,
    window: torch.Tensor,
) -> torch.Tensor:
    Sxx = torch.stft(
        data,
        n_fft=window_size,
        window=window,
        hop_length=hop_size,
        return_complex=True,
    )
    Sxx = Sxx.abs() ** 2
    Sxx = Sxx.log1p()

    if Sxx.shape[0] % 2 == 1:
        Sxx = Sxx[:-1]

    return Sxx


def normalize_spectrogram(Sxx: torch.Tensor, norm_config: dict) -> torch.Tensor:
    # vmin = norm_config["mean"] - norm_config["std"] * 3
    # vmax = norm_config["mean"] + norm_config["std"] * 3
    # Sxx = torch.clip(Sxx, vmin, vmax)
    Sxx = (Sxx - norm_config["mean"]) / (norm_config["std"] + 1e-8)
    return Sxx


class MultiscaleDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        scale_configs: list[tuple[int, int]],
        base_scale_weight: float = 0.3,
        chunk_samples: int = TRAIN_CHUNK_SAMPLES,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.scale_configs = scale_configs
        self.base_scale_weight = base_scale_weight
        self.chunk_samples = chunk_samples
        self.transform = transform

        # Find all sample files
        self.samples = []
        for diag_dir in self.data_dir.iterdir():
            for shot_dir in diag_dir.iterdir():
                for ts_file in shot_dir.glob("*_timeseries.npy"):
                    label_file = ts_file.with_name(
                        ts_file.name.replace("_timeseries.npy", "_label.npy")
                    )
                    if label_file.exists():
                        self.samples.append((ts_file, label_file))

        logger.info(f"Found {len(self.samples)} samples across all diagnostics")

        self.nfft_sizes = sorted(set(cfg[0] for cfg in scale_configs))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ts_path, label_path = self.samples[idx]

        diag_name = ts_path.parent.parent.name
        norm_config = NORMALIZATION_CONFIGS[diag_name]

        # Load data
        timeseries = np.load(ts_path) # (n_samples,)
        base_label = np.load(label_path)  # (2, H_base, W_base)
        n_samples = len(timeseries)
        base_label_width = base_label.shape[2]  # W dimension

        nfft, hop = self.scale_configs[
            np.random.randint(len(self.scale_configs))
            ]

        chunk_size = min(self.chunk_samples, n_samples)
        chunk_size = max(chunk_size, nfft + hop)

        start_sample = np.random.randint(0, n_samples - chunk_size)
        end_sample = start_sample + chunk_size

        ts_chunk = timeseries[start_sample:end_sample]
        label_start_frame = start_sample // BASE_HOP
        label_end_frame = min(end_sample // BASE_HOP, base_label_width)

        # Handle edge cases
        if label_start_frame >= base_label_width:
            label_start_frame = max(0, base_label_width - 10)
            label_end_frame = base_label_width

        if label_end_frame <= label_start_frame:
            label_end_frame = min(label_start_frame + 10, base_label_width)

        # Extract label chunk
        label_chunk = base_label[:, :, label_start_frame:label_end_frame]

        # Create spectrogram at selected scale
        timeseries_t = torch.from_numpy(ts_chunk).float()
        window = torch.hann_window(nfft)

        Sxx = make_spectrogram(timeseries_t, nfft, hop, window)
        Sxx_norm = normalize_spectrogram(Sxx, norm_config)

        # Resize label to match new spectrogram size
        target_size = [Sxx_norm.shape[0], Sxx_norm.shape[1]]
        base_label_t = torch.from_numpy(label_chunk).float()  # (2, H, W_chunk)

        # Resize each channel separately using bicubic interpolation
        label_resized = torch.zeros((2, *target_size))
        for ch in range(2):
            label_resized[ch] = resize(
                base_label_t[ch].unsqueeze(0),
                target_size,
                antialias=True,
                interpolation=Resampling.BICUBIC,
            ).squeeze(0)

        # Convert to binary labels (threshold at 0)
        # label_resized = torch.sigmoid(label_resized)
        label_resized = (label_resized > 0.25).float()

        # Add channel dimension to image
        x = Sxx_norm.unsqueeze(0)  # (1, H, W)
        y = label_resized  # (2, H, W)

        # Apply augmentation if provided
        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized spectrograms.
    Pads all samples in the batch to the maximum size.
    """
    # Find max dimensions
    max_h = max(x.shape[1] for x, y in batch)
    max_w = max(x.shape[2] for x, y in batch)

    # Ensure dimensions are divisible by 32 (for 5-layer UNet)
    max_h = ((max_h + 31) // 32) * 32
    max_w = ((max_w + 31) // 32) * 32

    # Pad and stack
    padded_x = []
    padded_y = []

    for x, y in batch:
        # x: (1, H, W), y: (2, H, W)
        h, w = x.shape[1], x.shape[2]
        pad_h = max_h - h
        pad_w = max_w - w

        # Pad on right and bottom
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
        y_padded = F.pad(y, (0, pad_w, 0, pad_h), mode="constant", value=0)

        padded_x.append(x_padded)
        padded_y.append(y_padded)

    return torch.stack(padded_x), torch.stack(padded_y)


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        settings: dict = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.settings = settings if settings is not None else default_settings

    def setup(self, stage=None):
        # Get augmentation (if enabled)
        transform = (
            get_augmentation(self.settings)
            if self.settings.get("augmentation", False)
            else None
        )

        chunk_samples = self.settings.get("chunk_samples", TRAIN_CHUNK_SAMPLES)

        # Create dataset (we'll use same data for train/val with different random scales)
        self.train_dataset = MultiscaleDataset(
            data_dir=self.data_dir,
            scale_configs=self.settings.get("scale_configs", SCALE_CONFIGS),
            base_scale_weight=self.settings.get("base_scale_weight", 0.3),
            chunk_samples=chunk_samples,
            transform=transform,
        )

        # Validation uses only base scale for consistent metrics
        self.val_dataset = MultiscaleDataset(
            data_dir=self.data_dir,
            scale_configs=[(BASE_NFFT, BASE_HOP)],  # Only base scale
            base_scale_weight=1.0,  # Always use base scale
            chunk_samples=chunk_samples,
            transform=None,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=custom_collate_fn,
        )


class MultiscaleModule(L.LightningModule):
    def __init__(
        self,
        num_layers: int = 5,
        first_layer_size: int = 32,
        settings: dict = None,
    ):
        super().__init__()
        self.settings = settings if settings is not None else default_settings

        self.unet = UNet(
            in_channels=1,
            out_channels=2,  # 2 channels: normal (ch0) and baseline (ch1)
            num_layers=num_layers,
            first_layer_size=first_layer_size,
            dropout_rate=self.settings.get("dropout_rate", 0.2),
        )

        # Note: torch.compile() is called after loading pretrained weights in main()
        self.loss_fn = get_loss_function(self.settings)

    def forward(self, x):
        return self.unet(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        y_hat = self(x)[0]

        channel_names = ["normal", "baseline"]
        losses = []
        dice_scores = []
        iou_scores = []

        for ch_idx, ch_name in enumerate(channel_names):
            y_hat_ch = y_hat[:, ch_idx : ch_idx + 1, ...]
            y_ch = y[:, ch_idx : ch_idx + 1, ...]

            loss_ch = self.loss_fn(y_hat_ch, y_ch)
            dice_ch = dice_coefficient(y_hat_ch, y_ch)
            iou_ch = iou_score(y_hat_ch, y_ch)

            losses.append(loss_ch)
            dice_scores.append(dice_ch)
            iou_scores.append(iou_ch)

            self.log(f"{stage}_loss_{ch_name}", loss_ch, batch_size=x.shape[0])
            self.log(f"{stage}_dice_{ch_name}", dice_ch, batch_size=x.shape[0])
            self.log(f"{stage}_iou_{ch_name}", iou_ch, batch_size=x.shape[0])

        total_loss = sum(losses) / len(losses)
        avg_dice = sum(dice_scores) / len(dice_scores)
        avg_iou = sum(iou_scores) / len(iou_scores)

        self.log(f"{stage}_loss", total_loss, prog_bar=True, batch_size=x.shape[0])
        self.log(f"{stage}_dice", avg_dice, prog_bar=True, batch_size=x.shape[0])
        self.log(f"{stage}_iou", avg_iou, batch_size=x.shape[0])

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.settings.get("learning_rate", 1e-4),
            weight_decay=self.settings.get("weight_decay", 1e-5),
        )

        lr_scheduler_type = self.settings.get("lr_scheduler", None)

        if lr_scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.settings.get("lr_factor", 0.5),
                patience=self.settings.get("lr_patience", 5),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.settings.get("max_epochs", 100),
                eta_min=self.settings.get("lr_min", 1e-6),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return optimizer


def main(config_path=None):
    """Main training entry point."""
    # python train_multiscale.py /scratch/gpfs/nc1514/TokEye/dev/notebooks/evaluation/settings/multiscale_settings.yaml

    if config_path is not None:
        from TokEye.autoprocess.utils.configuration import load_settings

        settings = load_settings(config_path, default_settings)
    else:
        settings = default_settings.copy()

    data_dir = settings["data_dir"]
    model_dir = settings["model_dir"]
    pretrained_path = settings.get("pretrained_path")

    # Setup model directory
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create data module
    datamodule = DataModule(
        data_dir=data_dir,
        batch_size=settings["batch_size"],
        num_workers=settings["num_workers"],
        prefetch_factor=settings["prefetch_factor"],
        settings=settings,
    )

    # Create model
    model = MultiscaleModule(
        num_layers=settings["num_layers"],
        first_layer_size=settings["first_layer_size"],
        settings=settings,
    )

    # Load pretrained weights if available (same approach as notebook)
    if pretrained_path is not None and pretrained_path.exists():
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        from TokEye.autoprocess.step_6d_final import Module as PretrainedModule

        pretrained = PretrainedModule.load_from_checkpoint(
            pretrained_path,
            num_layers=settings["num_layers"],
            first_layer_size=settings["first_layer_size"],
        )
        # Copy UNet weights directly
        model.unet.load_state_dict(pretrained.unet.state_dict())
        logger.info("Pretrained weights loaded successfully")

    # Compile model for faster training (after loading weights)
    if not settings.get("fast_dev_run", False):
        model.unet = torch.compile(model.unet)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename="best_multiscale",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=settings["early_stopping_patience"],
        mode="min",
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=settings["max_epochs"],
        precision=settings["precision"],
        devices=settings["devices"],
        log_every_n_steps=settings["log_every_n_steps"],
        enable_progress_bar=settings["enable_progress_bar"],
        fast_dev_run=settings["fast_dev_run"],
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    # Train
    logger.info(f"\n{'=' * 60}")
    logger.info("Starting Multiscale UNet Training")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Scale configurations: {len(settings['scale_configs'])} scales")
    logger.info(f"{'=' * 60}\n")

    trainer.fit(model, datamodule)

    # Export best model
    best_model_path = model_dir / "best_multiscale.ckpt"
    if best_model_path.exists():
        logger.info(f"Best model saved to: {best_model_path}")

    logger.info("\nMultiscale training complete!")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path)
