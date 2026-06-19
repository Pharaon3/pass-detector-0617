#!/usr/bin/env bash
# Build all video modifications for one clip, infer, and plot pass probabilities.
#
# Usage:
#   bash batch_clip_mods.sh clip_239
#   bash batch_clip_mods.sh clip_239 checkpoints/best.pt
set -euo pipefail

CLIP="${1:-clip_239}"
CHECKPOINT="${2:-checkpoints/best.pt}"
ANGLE="${3:-5}"

SRC="data/${CLIP}/224p.mp4"
LABEL="data/${CLIP}/label.json"
PLOT_DIR="outputs/plots/${CLIP}_mods"

if [[ ! -f "$SRC" ]]; then
  echo "Missing source video: $SRC"
  exit 1
fi
if [[ ! -f "$LABEL" ]]; then
  echo "Warning: no label.json — plots will have no GT lines"
  mkdir -p "data/${CLIP}"
  echo '{"observation":[],"anticipation":[]}' > "$LABEL"
fi

W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of csv=p=0 "$SRC")
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$SRC")

rotate_to() {
  local dst="$1"
  local angle="$2"
  python3 - "$SRC" "$dst" "$angle" "$W" "$H" <<'PY'
import math, sys, subprocess
src, dst, angle, w, h = sys.argv[1:6]
w, h = int(w), int(h)
a = math.radians(abs(float(angle)))
zoom_w = 1.0 / (math.cos(a) - math.sin(a) * h / w)
zoom_h = 1.0 / (math.cos(a) - math.sin(a) * w / h)
zoom = max(zoom_w, zoom_h) * 1.02
sw = int(math.ceil(w * zoom))
sh = int(math.ceil(h * zoom))
sw += sw % 2
sh += sh % 2
vf = f"rotate={angle}*PI/180:fillcolor=black,scale={sw}:{sh},crop={w}:{h}:(iw-{w})/2:(ih-{h})/2"
subprocess.run(
    ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", vf,
     "-c:v", "libx264", "-pix_fmt", "yuv420p", dst],
    check=True,
)
PY
}

run_variant() {
  local name="$1"
  local data_root="$2"
  local out_dir="$3"
  local video_path="${data_root}/${CLIP}/224p.mp4"
  local label_dest="${data_root}/${CLIP}/label.json"

  mkdir -p "${data_root}/${CLIP}" "$out_dir" "$PLOT_DIR"
  if [[ "$label_dest" != "$LABEL" ]]; then
    cp "$LABEL" "$label_dest"
  fi

  echo "  Infer: $name"
  python infer.py --checkpoint "$CHECKPOINT" \
    --video "$video_path" \
    --output-dir "$out_dir"

  echo "  Plot:  $name"
  python plot_probs.py --clip "$CLIP" \
    --data-root "$data_root" \
    --probs-json "${out_dir}/${CLIP}_frame_probs.json" \
    --output "${PLOT_DIR}/${name}.png"
}

build_if_missing() {
  local path="$1"
  shift
  mkdir -p "$(dirname "$path")"
  if [[ ! -f "$path" ]]; then
    echo "  Build: $path"
    "$@"
  else
    echo "  Exists: $path"
  fi
}

echo "=== Batch clip mods: ${CLIP} ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Plots:      ${PLOT_DIR}/"
echo ""

# Original (already in data/)
run_variant "original" "data" "outputs"

# Grayscale
build_if_missing "data_grayscale/${CLIP}/224p.mp4" \
  ffmpeg -y -loglevel error -i "$SRC" -vf "hue=s=0" \
  -c:v libx264 -pix_fmt yuv420p "data_grayscale/${CLIP}/224p.mp4"
run_variant "grayscale" "data_grayscale" "outputs_grayscale"

# Rotated left / right
build_if_missing "data_rotated_left/${CLIP}/224p.mp4" \
  rotate_to "data_rotated_left/${CLIP}/224p.mp4" "-${ANGLE}"
run_variant "rotated_left" "data_rotated_left" "outputs_rotated_left"

build_if_missing "data_rotated_right/${CLIP}/224p.mp4" \
  rotate_to "data_rotated_right/${CLIP}/224p.mp4" "${ANGLE}"
run_variant "rotated_right" "data_rotated_right" "outputs_rotated_right"

# Hue
build_if_missing "data_hue_plus/${CLIP}/224p.mp4" \
  ffmpeg -y -loglevel error -i "$SRC" -vf "hue=h=15*PI/180" \
  -c:v libx264 -pix_fmt yuv420p "data_hue_plus/${CLIP}/224p.mp4"
run_variant "hue_plus15" "data_hue_plus" "outputs_hue_plus"

build_if_missing "data_hue_minus/${CLIP}/224p.mp4" \
  ffmpeg -y -loglevel error -i "$SRC" -vf "hue=h=-15*PI/180" \
  -c:v libx264 -pix_fmt yuv420p "data_hue_minus/${CLIP}/224p.mp4"
run_variant "hue_minus15" "data_hue_minus" "outputs_hue_minus"

# Zoom 1.1x
build_if_missing "data_zoom_1.1/${CLIP}/224p.mp4" \
  ffmpeg -y -loglevel error -i "$SRC" \
  -vf "scale=${W}*1.1:${H}*1.1,crop=${W}:${H}:(iw-${W})/2:(ih-${H})/2" \
  -c:v libx264 -pix_fmt yuv420p "data_zoom_1.1/${CLIP}/224p.mp4"
run_variant "zoom_1.1" "data_zoom_1.1" "outputs_zoom_1.1"

echo ""
echo "Done."
echo "  Plots: ${PLOT_DIR}/"
ls -1 "${PLOT_DIR}/"*.png
