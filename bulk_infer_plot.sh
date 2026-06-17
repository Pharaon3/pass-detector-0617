#!/usr/bin/env bash
# Bulk infer all clips in data/, then plot pass-probability curves.
set -euo pipefail

CHECKPOINT="${1:-checkpoints/best.pt}"

echo "=== Step 1/2: Inference on all clips in data/ ==="
python infer.py --checkpoint "$CHECKPOINT"

echo ""
echo "=== Step 2/2: Plot all clips ==="
python plot_probs.py --all --probs-dir outputs --output-dir outputs/plots

echo ""
echo "Done."
echo "  JSON : outputs/*_frame_probs.json"
echo "  Plots: outputs/plots/*_pass_probs.png"
