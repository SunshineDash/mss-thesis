#!/usr/bin/env bash
# =============================================================================
# run_ablation.sh — full ablation training + evaluation pipeline
#
# Usage (from project root):
#   bash run_ablation.sh
#
# Requirements:
#   - Python virtualenv activated, or conda env with requirements.txt installed
#   - MUSDB18-HQ placed at data/musdb18hq/  (or update root in configs/)
#
# The script will:
#   1. Train all 5 models sequentially (baseline, dsc5, dsc10, dsc20, dsc_full)
#   2. Evaluate each model from its best.pt checkpoint
#   3. Save per-track and aggregate CSV results to results/
# =============================================================================

set -euo pipefail

EXPERIMENTS=(baseline dsc5 dsc10 dsc20 dsc_full)

echo "============================================================"
echo "  AudioSep Ablation Study — Training + Evaluation Pipeline  "
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# 1. Training
# ---------------------------------------------------------------------------
for EXP in "${EXPERIMENTS[@]}"; do
    echo "------------------------------------------------------------"
    echo "  Training: ${EXP}"
    echo "------------------------------------------------------------"
    python src/train.py --config "configs/${EXP}.yaml"
    echo "  ✓ Done training ${EXP}"
    echo ""
done

# ---------------------------------------------------------------------------
# 2. Evaluation
# ---------------------------------------------------------------------------
for EXP in "${EXPERIMENTS[@]}"; do
    CKPT="checkpoints/${EXP}/best.pt"
    echo "------------------------------------------------------------"
    echo "  Evaluating: ${EXP}  (checkpoint: ${CKPT})"
    echo "------------------------------------------------------------"
    if [ -f "${CKPT}" ]; then
        python src/evaluate.py \
            --config  "configs/${EXP}.yaml" \
            --checkpoint "${CKPT}"
        echo "  ✓ Done evaluating ${EXP}"
        rclone copy results/ gdrive:mss-results/
        echo "  ✓ Results synced to gdrive:mss-results/"
    else
        echo "  ⚠  Checkpoint not found: ${CKPT} — skipping evaluation."
    fi
    echo ""
done

echo "============================================================"
echo "  All experiments complete!"
echo "  Results saved to: results/"
echo "============================================================"

# Final sync to make sure everything is uploaded
rclone copy results/ gdrive:mss-results/
echo "  ✓ Final sync to gdrive:mss-results/ complete"
