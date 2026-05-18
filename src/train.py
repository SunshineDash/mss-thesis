"""Training entry point for Conv-TasNet ablation experiments.

Usage:
    python src/train.py --config configs/baseline.yaml
    python src/train.py --config configs/dsc5.yaml --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running directly from within the src/ directory
sys.path.insert(0, str(Path(__file__).parent))

from models.conv_tasnet import ConvTasNet
from models.dsc_conv_tasnet import DSCConvTasNet
from data.musdb_dataset import MUSDBDataset, worker_init_fn
from losses import si_sdr_loss
from metrics import compute_si_sdr
from utils import (
    append_metrics_csv,
    count_parameters,
    ensure_dir,
    load_checkpoint,
    load_config,
    log_experiment_start,
    resolve_device,
    save_checkpoint,
    set_seed,
)


# =========================================================================== #
# Model factory                                                                 #
# =========================================================================== #

def build_model(model_config: dict) -> torch.nn.Module:
    """Instantiate the correct model class from the config dict."""
    model_type = model_config.get("type", "conv_tasnet")
    # Pass everything except 'type' as kwargs
    kwargs = {k: v for k, v in model_config.items() if k != "type"}

    if model_type == "conv_tasnet":
        return ConvTasNet(**kwargs)
    if model_type == "dsc_conv_tasnet":
        return DSCConvTasNet(**kwargs)

    raise ValueError(
        f"Unknown model type '{model_type}'. "
        "Valid options: 'conv_tasnet', 'dsc_conv_tasnet'."
    )


# =========================================================================== #
# Training & validation loops                                                   #
# =========================================================================== #

def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    pit: bool = False,
    grad_clip: float = 5.0,
    desc: str = "Train",
) -> tuple[float, float]:
    """Run one training or validation epoch.

    Args:
        model:      The separation model.
        loader:     DataLoader providing (mixture, sources) pairs.
        optimizer:  If None the function runs in eval mode (no gradients).
        device:     Computation device.
        pit:        Use permutation-invariant training.
        grad_clip:  Max gradient norm for clipping (ignored if optimizer=None).
        desc:       Description for the tqdm progress bar.

    Returns:
        (mean_loss, mean_si_sdr) for the epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)
    context = torch.enable_grad() if is_train else torch.no_grad()

    total_loss = 0.0
    total_si_sdr = 0.0
    n_batches = 0

    with context:
        bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for mixture, sources in bar:
            # mixture: [B, C, T]   sources: [B, S, C, T]
            mixture = mixture.to(device)
            sources = sources.to(device)

            estimate = model(mixture)  # [B, S, C, T]

            # Flatten channel dim for loss/metric: [B, S, C*T] → use last dim
            # For SI-SDR we treat each (source, channel) pair independently
            B, S, C, T = estimate.shape
            est_flat = estimate.reshape(B, S * C, T)
            ref_flat = sources.reshape(B, S * C, T)

            loss = si_sdr_loss(est_flat, ref_flat, permutation_invariant=pit)
            si_sdr_val = compute_si_sdr(est_flat.detach(), ref_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            total_si_sdr += si_sdr_val
            n_batches += 1
            bar.set_postfix(loss=f"{loss.item():.3f}", si_sdr=f"{si_sdr_val:.2f}")

    return total_loss / max(n_batches, 1), total_si_sdr / max(n_batches, 1)


# =========================================================================== #
# Main training function                                                        #
# =========================================================================== #

def train(config_path: str | Path, resume: bool = False) -> None:
    """Full training pipeline for one experiment."""
    config = load_config(config_path)

    # ------------------------------------------------------------------ #
    # Experiment setup                                                     #
    # ------------------------------------------------------------------ #
    exp_cfg = config.get("experiment", {})
    exp_name = exp_cfg.get("name", "unnamed")
    seed = int(exp_cfg.get("seed", 42))
    set_seed(seed)

    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    device = resolve_device(str(train_cfg.get("device", "auto")))
    checkpoint_dir = ensure_dir(train_cfg.get("checkpoint_dir", f"checkpoints/{exp_name}"))
    log_experiment_start(config, checkpoint_dir)

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    # Propagate audio_channels from data config if not in model config
    if "audio_channels" not in model_cfg:
        model_cfg = dict(model_cfg)
        model_cfg["audio_channels"] = int(data_cfg.get("audio_channels", 2))

    model = build_model(model_cfg).to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # ------------------------------------------------------------------ #
    # Optimizer & scheduler                                                #
    # ------------------------------------------------------------------ #
    lr = float(train_cfg.get("learning_rate", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    epochs = int(train_cfg.get("epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )


    # ------------------------------------------------------------------ #
    # Optionally resume                                                    #
    # ------------------------------------------------------------------ #
    start_epoch = 0
    best_val_loss = float("inf")
    latest_path = checkpoint_dir / "latest.pt"
    if resume and latest_path.exists():
        ckpt = load_checkpoint(latest_path, model, optimizer, device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        print(f"Resumed from {latest_path} — starting at epoch {start_epoch}")

    # ------------------------------------------------------------------ #
    # Datasets                                                             #
    # ------------------------------------------------------------------ #
    data_root = data_cfg.get("root", "data/musdb18")
    sources = list(data_cfg.get("sources", ["vocals", "drums", "bass", "other"]))
    sample_rate = int(data_cfg.get("sample_rate", 44100))
    segment_seconds = float(data_cfg.get("segment_seconds", 4.0))
    batch_size = int(data_cfg.get("batch_size", 4))
    num_workers = int(data_cfg.get("num_workers", 2))

    train_dataset = MUSDBDataset(
        root=data_root,
        sources=sources,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        split="train",
        seed=seed,
    )
    # Use last 10 % of training tracks as a quick validation proxy
    n_val = max(1, len(train_dataset) // 10)
    n_train = len(train_dataset) - n_val
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn,
    )

    print(f"Train samples: {len(train_subset)}  Val samples: {len(val_subset)}")

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    grad_clip = float(train_cfg.get("grad_clip_norm", 5.0))
    pit = str(train_cfg.get("loss", "si_sdr")) == "pit_si_sdr"
    log_csv = Path(exp_cfg.get("output_dir", "results")) / f"{exp_name}_train_log.csv"

    for epoch in range(start_epoch, epochs):
        print(f"\n[Epoch {epoch + 1}/{epochs}]")

        train_subset.dataset.epoch = epoch
        train_loss, train_si_sdr = run_epoch(
            model, train_loader, optimizer, device,
            pit=pit, grad_clip=grad_clip, desc=f"Train E{epoch+1}"
        )
        val_loss, val_si_sdr = run_epoch(
            model, val_loader, optimizer=None, device=device, desc=f"Val   E{epoch+1}"
        )

        scheduler.step()
        lr_current = optimizer.param_groups[0]["lr"]

        print(
            f"  train_loss={train_loss:.4f}  train_si_sdr={train_si_sdr:.2f} dB"
            f"  |  val_loss={val_loss:.4f}  val_si_sdr={val_si_sdr:.2f} dB"
            f"  |  lr={lr_current:.2e}"
        )

        # Log to CSV
        append_metrics_csv(log_csv, {
            "experiment": exp_name,
            "seed": seed,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_si_sdr": train_si_sdr,
            "val_loss": val_loss,
            "val_si_sdr": val_si_sdr,
            "lr": lr_current,
        })

        # Save latest checkpoint
        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model, optimizer, epoch, best_val_loss, config, seed,
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model, optimizer, epoch, best_val_loss, config, seed,
            )
            print(f"  ✓ New best val_loss={best_val_loss:.4f} — saved best.pt")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Training log: {log_csv}")


# =========================================================================== #
# CLI                                                                           #
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Conv-TasNet ablation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the YAML experiment config (e.g. configs/baseline.yaml)."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint if it exists."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config, resume=args.resume)
