#!/usr/bin/env bash
# 1) Build grayscale copies of all clips
# 2) Infer pass probabilities on grayscale videos
# 3) Plot probability curves (uses labels copied into data_grayscale/)
set -euo pipefail

CHECKPOINT="${1:-checkpoints/best.pt}"

echo "=== Step 1/3: Grayscale videos ==="
bash make_grayscale_clips.sh data data_grayscale

echo ""
echo "=== Step 2/3: Inference (grayscale) ==="
python infer.py \
  --checkpoint "$CHECKPOINT" \
  --video-dir data_grayscale \
  --output-dir outputs_grayscale \
  --skip-existing

echo ""
echo "=== Step 3/3: Plot (grayscale) ==="
python plot_probs.py \
  --all \
  --data-root data_grayscale \
  --probs-dir outputs_grayscale \
  --output-dir outputs_grayscale/plots \
  --skip-existing

echo ""
echo "Done."
echo "  Videos : data_grayscale/clip_XXX/224p.mp4"
echo "  JSON   : outputs_grayscale/*_frame_probs.json"
echo "  Plots  : outputs_grayscale/plots/*_pass_probs.png"
