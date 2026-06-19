#!/usr/bin/env bash
# Bulk infer all clips with the track model, then plot pass-probability curves.
set -euo pipefail

CHECKPOINT="${1:-checkpoints_tracks/best.pt}"

echo "=== Step 1/2: Track inference on all clips in data/ ==="
python infer_tracks.py --checkpoint "$CHECKPOINT" --skip-existing

echo ""
echo "=== Step 2/2: Plot all clips ==="
python plot_track_probs.py --all \
  --checkpoint "$CHECKPOINT" \
  --probs-dir outputs_tracks \
  --output-dir outputs_tracks/plots \
  --skip-existing

echo ""
echo "Done."
echo "  JSON : outputs_tracks/*_frame_probs.json"
echo "  Events: outputs_tracks/*_events.json"
echo "  Plots: outputs_tracks/plots/*_pass_probs.png"
