"""Shared utilities for training and evaluation."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml


# =========================================================================== #
# Config                                                                        #
# =========================================================================== #

def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return it as a plain dict."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_config(config: dict, path: str | Path) -> None:
    """Persist config dict to a YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.dump(config, fh, default_flow_style=False, allow_unicode=True)


# =========================================================================== #
# File-system helpers                                                           #
# =========================================================================== #

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist; return the Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# =========================================================================== #
# Reproducibility                                                               #
# =========================================================================== #

def set_seed(seed: int) -> None:
    """Set Python / NumPy / PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================================== #
# Device                                                                        #
# =========================================================================== #

def resolve_device(device: str = "auto") -> torch.device:
    """Return the appropriate torch.device.

    Args:
        device: ``"auto"`` to pick CUDA when available, else CPU;
                or an explicit string like ``"cuda"`` / ``"cpu"`` / ``"cuda:1"``.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# =========================================================================== #
# Model helpers                                                                 #
# =========================================================================== #

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================================== #
# Checkpoint I/O                                                                #
# =========================================================================== #

def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: dict,
    seed: int = 42,
    extra: dict | None = None,
) -> None:
    """Save a training checkpoint.

    The checkpoint contains everything needed to resume training or run
    evaluation:
        - ``model_state_dict``
        - ``optimizer_state_dict``
        - ``epoch``
        - ``best_val_loss``
        - ``config``  (the full YAML config as a dict)
        - ``seed``
        - any additional key-value pairs from ``extra``
    """
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": config,
        "seed": seed,
    }
    if extra:
        payload.update(extra)
    ensure_dir(Path(path).parent)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    """Load a checkpoint and inject state into model (and optionally optimizer).

    Returns:
        The full checkpoint dict (contains epoch, best_val_loss, config, seed …).
    """
    if device is None:
        device = torch.device("cpu")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


# =========================================================================== #
# CSV logging                                                                   #
# =========================================================================== #

def save_metrics_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, object]],
    mode: str = "w",
) -> None:
    """Write a list of metric dicts to a CSV file.

    Args:
        path:  Output CSV path.
        rows:  Iterable of dicts where all dicts share the same keys.
        mode:  ``"w"`` to overwrite, ``"a"`` to append.
    """
    rows = list(rows)
    if not rows:
        return
    output_path = Path(path)
    ensure_dir(output_path.parent)
    write_header = mode == "w" or not output_path.exists()
    with output_path.open(mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_metrics_csv(path: str | Path, row: Mapping[str, object]) -> None:
    """Append a single row to a CSV file (creates file/header if needed)."""
    save_metrics_csv(path, [row], mode="a")


# =========================================================================== #
# Logging helpers                                                               #
# =========================================================================== #

def log_experiment_start(config: dict, checkpoint_dir: Path) -> None:
    """Print experiment header and persist the config to checkpoint_dir."""
    exp_name = config.get("experiment", {}).get("name", "unnamed")
    print("=" * 60)
    print(f"  Experiment : {exp_name}")
    print(f"  Seed       : {config.get('experiment', {}).get('seed', 42)}")
    print(f"  Device     : {config.get('training', {}).get('device', 'auto')}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print("=" * 60)
    save_config(config, checkpoint_dir / "config.yaml")
