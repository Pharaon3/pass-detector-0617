#!/usr/bin/env bash
# Rotate ±5°, infer all, plot — left and right variants.
#
# Usage:
#   bash bulk_rotated_infer_plot.sh checkpoints/best.pt
#   FORCE=1 bash bulk_rotated_infer_plot.sh checkpoints/best.pt   # rebuild videos + re-infer all
set -euo pipefail

CHECKPOINT="${1:-checkpoints/best.pt}"
ANGLE="${2:-5}"
FORCE="${FORCE:-0}"

SKIP_ARGS=()
if [[ "$FORCE" != "1" ]]; then
  SKIP_ARGS=(--skip-existing)
fi

run_variant() {
  local name="$1"
  local data_dir="$2"
  local out_dir="$3"

  echo ""
  echo "=== Infer: $name ==="
  python infer.py \
    --checkpoint "$CHECKPOINT" \
    --video-dir "$data_dir" \
    --output-dir "$out_dir" \
    "${SKIP_ARGS[@]}"

  echo ""
  echo "=== Plot: $name ==="
  python plot_probs.py \
    --all \
    --data-root "$data_dir" \
    --probs-dir "$out_dir" \
    --output-dir "$out_dir/plots" \
    "${SKIP_ARGS[@]}"
}

echo "=== Step 1/3: Rotate clips ±${ANGLE}° ==="
FORCE="$FORCE" bash make_rotated_clips.sh data data_rotated_left data_rotated_right "$ANGLE"

echo ""
echo "=== Step 2/3: Left rotation (-${ANGLE}°) ==="
run_variant "left (-${ANGLE}°)" data_rotated_left outputs_rotated_left

echo ""
echo "=== Step 3/3: Right rotation (+${ANGLE}°) ==="
run_variant "right (+${ANGLE}°)" data_rotated_right outputs_rotated_right

echo ""
echo "Done."
echo "  Left  JSON/plots : outputs_rotated_left/"
echo "  Right JSON/plots: outputs_rotated_right/"
