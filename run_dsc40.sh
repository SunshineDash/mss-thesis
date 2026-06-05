#!/bin/bash
set -e

source .venv/bin/activate

# =============================================================
# DSC-40 (num_repeats=5 → 8×5=40 temporal blocks)
# =============================================================
echo "========================================================="
echo "  DSC-40 — Training"
echo "  Starting: $(date)"
echo "========================================================="
python src/train.py --config configs/dsc40.yaml
echo ""

echo "========================================================="
echo "  DSC-40 — Evaluation"
echo "========================================================="
python src/evaluate.py --config configs/dsc40.yaml --checkpoint checkpoints/dsc40/best.pt
echo ""

echo "========================================================="
echo "  DSC-40 — Sync to Google Drive"
echo "========================================================="
rclone copy results/ gdrive:mss-results/ --progress -P
rclone copy checkpoints/ gdrive:mss-checkpoints/ --progress -P
echo ""

# =============================================================
# DSC-48 (num_repeats=6 → 8×6=48 temporal blocks)
# =============================================================
echo "========================================================="
echo "  DSC-48 — Training"
echo "  Starting: $(date)"
echo "========================================================="
python src/train.py --config configs/dsc48.yaml
echo ""

echo "========================================================="
echo "  DSC-48 — Evaluation"
echo "========================================================="
python src/evaluate.py --config configs/dsc48.yaml --checkpoint checkpoints/dsc48/best.pt
echo ""

echo "========================================================="
echo "  DSC-48 — Sync to Google Drive"
echo "========================================================="
rclone copy results/ gdrive:mss-results/ --progress -P
rclone copy checkpoints/ gdrive:mss-checkpoints/ --progress -P
echo ""

echo "========================================================="
echo "  All experiments complete!"
echo "  Finished: $(date)"
echo "========================================================="