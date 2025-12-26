import logging
import shutil
from pathlib import Path

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def load_settings(
    config_path: Path | str | None,
    default_settings: dict = None,
) -> dict:
    """Load settings from YAML file or use defaults."""

    if default_settings is None:
        default_settings = {}
    if config_path is None:
        cfg = default_settings
    else:
        config_path = Path(config_path)
        if not config_path.exists():
            logger.error("Config file not found")

        cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.to_container(cfg, resolve=True)

        for key, value in cfg.items():
            if (key.endswith("_dir") or key.endswith("_path")) and value is not None:
                cfg[key] = Path(value)

    return cfg


def load_input_paths(
    input_dir: Path,
) -> list[Path]:
    """Load input paths from input directory."""
    input_paths = list(input_dir.glob("*.joblib"))
    input_paths.sort(key=lambda x: int(x.stem))
    return input_paths


def setup_directory(
    path: Path | str,
    overwrite: bool = True,
) -> Path:
    """Setup output directory."""

    path = Path(path)
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
