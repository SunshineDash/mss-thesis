"""Evaluation entry point for Conv-TasNet ablation experiments.

Runs full-track inference over the MUSDB18-HQ test split and computes
BSS-Eval (SDR / SIR / SAR / ISR via museval) and SI-SDR metrics.

Usage:
    python src/evaluate.py \\
        --config  configs/baseline.yaml \\
        --checkpoint checkpoints/baseline/best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import math
import numpy as np
import torch
from tqdm import tqdm

# Allow running directly from within the src/ directory
sys.path.insert(0, str(Path(__file__).parent))

from train import build_model
from data.musdb_dataset import MUSDBDataset
from inference import separate_track
from metrics import aggregate_track_scores, compute_si_sdr, evaluate_track
from utils import (
    count_parameters,
    ensure_dir,
    load_checkpoint,
    load_config,
    resolve_device,
    save_metrics_csv,
    set_seed,
)


# =========================================================================== #
# Main evaluation function                                                      #
# =========================================================================== #

def evaluate(config_path: str | Path, checkpoint_path: str | Path) -> None:
    """Evaluate a trained model on the MUSDB18-HQ test split.

    Results are saved to two CSV files:
    - ``<output_dir>/<experiment>_per_track.csv``   — per-track, per-source metrics
    - ``<output_dir>/<experiment>_metrics.csv``     — aggregate metrics (mean/median)
    """
    config = load_config(config_path)

    exp_cfg = config.get("experiment", {})
    exp_name = exp_cfg.get("name", "unnamed")
    seed = int(exp_cfg.get("seed", 42))
    set_seed(seed)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    eval_cfg = config.get("evaluation", {})
    train_cfg = config.get("training", {})

    device = resolve_device(str(train_cfg.get("device", "auto")))

    # ------------------------------------------------------------------ #
    # Build and load model                                                 #
    # ------------------------------------------------------------------ #
    if "audio_channels" not in model_cfg:
        model_cfg = dict(model_cfg)
        model_cfg["audio_channels"] = int(data_cfg.get("audio_channels", 2))

    model = build_model(model_cfg).to(device)
    n_params = count_parameters(model)

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        ckpt = load_checkpoint(checkpoint_path, model, device=device)
        trained_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {trained_epoch})")
    else:
        print(f"⚠  Checkpoint not found: {checkpoint_path}  — evaluating untrained model.")
        trained_epoch = 0

    model.eval()
    print(f"Model parameters: {n_params:,}")

    # ------------------------------------------------------------------ #
    # Test dataset (full tracks)                                           #
    # ------------------------------------------------------------------ #
    data_root = data_cfg.get("root", "data/musdb18")
    sources = list(data_cfg.get("sources", ["vocals", "drums", "bass", "other"]))
    sample_rate = int(data_cfg.get("sample_rate", 44100))

    test_dataset = MUSDBDataset(
        root=data_root,
        sources=sources,
        sample_rate=sample_rate,
        segment_seconds=None,   # full tracks
        split="test",
        full_track=True,
    )
    print(f"Test tracks: {len(test_dataset)}")

    # ------------------------------------------------------------------ #
    # Chunked inference settings                                           #
    # ------------------------------------------------------------------ #
    chunk_size = int(eval_cfg.get("chunk_size", 65536))
    hop_size   = int(eval_cfg.get("hop_size", chunk_size // 2))

    # ------------------------------------------------------------------ #
    # Evaluate track by track                                              #
    # ------------------------------------------------------------------ #
    output_dir = ensure_dir(exp_cfg.get("output_dir", "results"))
    per_track_csv = output_dir / f"{exp_name}_per_track.csv"
    per_track_rows: list[dict] = []
    per_track_scores: list[dict] = []
    total_inference_ms = 0.0

    for idx in tqdm(range(len(test_dataset)), desc="Evaluating", unit="track"):
        track_dir = test_dataset.index[idx]
        track_name = track_dir.name

        mixture, sources_ref = test_dataset[idx]
        # mixture:      [C, T]
        # sources_ref:  [S, C, T]

        # Overlap-add inference
        separated, elapsed_ms = separate_track(
            model=model,
            mixture=mixture,
            chunk_size=chunk_size,
            hop_size=hop_size,
            device=device,
        )
        # separated: [S, C, T]  (may have different length; align)
        total_inference_ms += elapsed_ms
        T = sources_ref.shape[-1]
        separated = separated[..., :T]

        # ---- SI-SDR (PyTorch, tensor) — computed per source below --- #
        sep_pt = separated.unsqueeze(0)    # [1, S, C, T]
        ref_pt = sources_ref.unsqueeze(0).to(sep_pt.device)  # [1, S, C, T]
        B, S, C, T_ = sep_pt.shape

        # ---- museval BSS-Eval (numpy) -------------------------------- #
        sep_np = separated.cpu().numpy()   # [S, C, T]
        ref_np = sources_ref.cpu().numpy() # [S, C, T]
        try:
            track_scores = evaluate_track(sep_np, ref_np, sample_rate=sample_rate)
        except Exception as exc:
            print(f"  ⚠  museval failed for track '{track_name}': {exc}")
            track_scores = {"sdr": [float("nan")] * len(sources),
                            "sir": [float("nan")] * len(sources),
                            "sar": [float("nan")] * len(sources),
                            "isr": [float("nan")] * len(sources)}

        per_track_scores.append(track_scores)

        # ---- Per-track rows (one row per source)
        for s_idx, src_name in enumerate(sources):
            # Per-source SI-SDR
            si_sdr_src = compute_si_sdr(
                sep_pt[:, s_idx, :, :].reshape(1, C, T_),
                ref_pt[:, s_idx, :, :].reshape(1, C, T_),
            )
            # Guard against inf SIR (common when there is no interferer energy)
            sir_val = track_scores["sir"][s_idx]
            if sir_val is not None and not math.isnan(float(sir_val)) and math.isinf(float(sir_val)):
                sir_val = float("nan")
            per_track_rows.append({
                "experiment":        exp_name,
                "seed":              seed,
                "checkpoint":        str(checkpoint_path),
                "track":             track_name,
                "source":            src_name,
                "si_sdr":            si_sdr_src,
                "sdr":               track_scores["sdr"][s_idx],
                "sir":               sir_val,
                "sar":               track_scores["sar"][s_idx],
                "isr":               track_scores["isr"][s_idx],
                "params":            n_params,
                "inference_time_ms": elapsed_ms,
            })

    # ------------------------------------------------------------------ #
    # Save per-track CSV                                                   #
    # ------------------------------------------------------------------ #
    save_metrics_csv(per_track_csv, per_track_rows)
    print(f"Per-track metrics saved: {per_track_csv}")

    # ------------------------------------------------------------------ #
    # Aggregate metrics                                                    #
    # ------------------------------------------------------------------ #
    agg = aggregate_track_scores(per_track_scores)
    mean_si_sdr = float(
        np.mean([r["si_sdr"] for r in per_track_rows if not np.isnan(r["si_sdr"])])
        if per_track_rows else float("nan")
    )
    mean_inference_ms = total_inference_ms / max(len(test_dataset), 1)

    aggregate_row = {
        "experiment":        exp_name,
        "seed":              seed,
        "checkpoint":        str(checkpoint_path),
        "epoch":             trained_epoch,
        "si_sdr":            mean_si_sdr,
        "sdr":               agg["sdr"],
        "sir":               agg["sir"],
        "sar":               agg["sar"],
        "isr":               agg["isr"],
        "params":            n_params,
        "inference_time_ms": mean_inference_ms,
    }

    agg_csv = output_dir / f"{exp_name}_metrics.csv"
    # Also honour the eval.save_csv config key for backward compatibility
    legacy_csv = eval_cfg.get("save_csv")
    if legacy_csv:
        save_metrics_csv(legacy_csv, [aggregate_row])

    save_metrics_csv(agg_csv, [aggregate_row])
    print(f"Aggregate metrics saved: {agg_csv}")

    # ------------------------------------------------------------------ #
    # Summary print                                                        #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(f"  Experiment : {exp_name}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  SI-SDR     : {mean_si_sdr:.2f} dB")
    print(f"  SDR        : {agg['sdr']:.2f} dB")
    print(f"  SIR        : {agg['sir']:.2f} dB")
    print(f"  SAR        : {agg['sar']:.2f} dB")
    print(f"  Params     : {n_params:,}")
    print(f"  Avg infer  : {mean_inference_ms:.1f} ms/track")
    print("=" * 60)


# =========================================================================== #
# CLI                                                                           #
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Conv-TasNet ablation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the YAML experiment config (e.g. configs/baseline.yaml)."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the model checkpoint .pt file."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.config, args.checkpoint)
