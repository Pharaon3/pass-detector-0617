#!/usr/bin/env bash
# Rotate all clips by ±5° and crop/scale back to original resolution.
#
# Usage:
#   bash make_rotated_clips.sh
#   bash make_rotated_clips.sh data data_rotated_left data_rotated_right 5

set -euo pipefail

SRC="${1:-data}"
DST_LEFT="${2:-data_rotated_left}"
DST_RIGHT="${3:-data_rotated_right}"
ANGLE="${4:-5}"

rotate_clip() {
  local src_video="$1"
  local dst_video="$2"
  local angle="$3"   # degrees: negative = left (CCW), positive = right (CW)

  local W H
  W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of csv=p=0 "$src_video")
  H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$src_video")

  # rotate → scale up to cover → center-crop back to original WxH
  ffmpeg -y -loglevel error -i "$src_video" \
    -vf "rotate=${angle}*PI/180:fillcolor=black,scale=${W}:${H}:force_original_aspect_ratio=increase,crop=${W}:${H}" \
    -c:v libx264 -pix_fmt yuv420p \
    "$dst_video"
}

echo "Source : $SRC"
echo "Left   : $DST_LEFT  (-${ANGLE} deg)"
echo "Right  : $DST_RIGHT (+${ANGLE} deg)"

for clip_dir in "$SRC"/clip_*; do
  [[ -d "$clip_dir" ]] || continue
  clip="$(basename "$clip_dir")"
  src_video="$clip_dir/224p.mp4"
  [[ -f "$src_video" ]] || continue

  mkdir -p "$DST_LEFT/$clip" "$DST_RIGHT/$clip"
  left_video="$DST_LEFT/$clip/224p.mp4"
  right_video="$DST_RIGHT/$clip/224p.mp4"

  if [[ ! -f "$left_video" ]]; then
    echo "Rotate left  (-${ANGLE}°): $clip"
    rotate_clip "$src_video" "$left_video" "-${ANGLE}"
  else
    echo "Skip left  $clip (exists)"
  fi

  if [[ ! -f "$right_video" ]]; then
    echo "Rotate right (+${ANGLE}°): $clip"
    rotate_clip "$src_video" "$right_video" "${ANGLE}"
  else
    echo "Skip right $clip (exists)"
  fi

  if [[ -f "$clip_dir/label.json" ]]; then
    cp "$clip_dir/label.json" "$DST_LEFT/$clip/label.json"
    cp "$clip_dir/label.json" "$DST_RIGHT/$clip/label.json"
  fi
done

echo "Done."
echo "  $DST_LEFT"
echo "  $DST_RIGHT"
