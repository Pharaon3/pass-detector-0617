#!/usr/bin/env bash
# Build 1s-offset clips, infer all, plot probability curves.
#
# Video model:
#   bash bulk_shifted_infer_plot.sh checkpoints/best.pt
#
# Track model (pass checkpoints_tracks/best.pt):
#   MODEL=tracks bash bulk_shifted_infer_plot.sh checkpoints_tracks/best.pt
set -euo pipefail

CHECKPOINT="${1:-checkpoints/best.pt}"
MODEL="${MODEL:-video}"   # video | tracks
DATA_DST="${DATA_DST:-data_shifted_1s}"
OUT_DIR="${OUT_DIR:-outputs_shifted_1s}"

echo "=== Step 1/3: Build 1s-shifted videos ==="
bash make_shifted_clips.sh data "$DATA_DST"

if [[ "$MODEL" == "tracks" ]]; then
  echo ""
  echo "=== Step 2/3: Track inference (shifted) ==="
  python extract_tracks.py --data-root "$DATA_DST" --force
  python infer_tracks.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$OUT_DIR" \
    --skip-existing

  echo ""
  echo "=== Step 3/3: Plot (shifted, track model) ==="
  python plot_track_probs.py \
    --all \
    --data-root "$DATA_DST" \
    --checkpoint "$CHECKPOINT" \
    --probs-dir "$OUT_DIR" \
    --output-dir "$OUT_DIR/plots" \
    --skip-existing
else
  echo ""
  echo "=== Step 2/3: Inference (shifted, video model) ==="
  python infer.py \
    --checkpoint "$CHECKPOINT" \
    --video-dir "$DATA_DST" \
    --output-dir "$OUT_DIR" \
    --skip-existing

  echo ""
  echo "=== Step 3/3: Plot (shifted, video model) ==="
  python plot_probs.py \
    --all \
    --data-root "$DATA_DST" \
    --probs-dir "$OUT_DIR" \
    --output-dir "$OUT_DIR/plots" \
    --skip-existing
fi

echo ""
echo "Done."
echo "  Videos : ${DATA_DST}/*/224p.mp4"
echo "  JSON   : ${OUT_DIR}/*_frame_probs.json"
echo "  Plots  : ${OUT_DIR}/plots/*_pass_probs.png"
