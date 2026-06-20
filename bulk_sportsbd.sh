#!/usr/bin/env bash
# Bulk SportSBD shot boundary detection + timeline plots for all clips in data/
set -euo pipefail

DATA_ROOT="${1:-data}"
THRESHOLD="${2:-0.7}"

echo "=== SportSBD on all clips in ${DATA_ROOT} ==="
python detect_shots_sportsbd.py \
  --all \
  --data-root "$DATA_ROOT" \
  --threshold "$THRESHOLD" \
  --plot \
  --skip-existing

echo ""
echo "Done."
echo "  JSON : outputs/sportsbd/*_shots.json"
echo "  Plots: outputs/sportsbd/plots/*_shots.png"
