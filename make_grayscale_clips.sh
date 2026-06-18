#!/usr/bin/env bash
# Convert all clips in data/ to grayscale (no color), keeping size/SAR and labels.
#
# Usage:
#   bash make_grayscale_clips.sh
#   bash make_grayscale_clips.sh data data_grayscale

set -euo pipefail

SRC="${1:-data}"
DST="${2:-data_grayscale}"

echo "Source: $SRC"
echo "Output: $DST"

for clip_dir in "$SRC"/clip_*; do
  [[ -d "$clip_dir" ]] || continue
  clip="$(basename "$clip_dir")"
  src_video="$clip_dir/224p.mp4"
  [[ -f "$src_video" ]] || continue

  mkdir -p "$DST/$clip"
  dst_video="$DST/$clip/224p.mp4"

  if [[ -f "$dst_video" ]]; then
    echo "Skip $clip (already exists)"
  else
    echo "Grayscale: $clip"
    ffmpeg -y -loglevel error -i "$src_video" \
      -vf "hue=s=0" \
      -c:v libx264 -pix_fmt yuv420p \
      "$dst_video"
  fi

  if [[ -f "$clip_dir/label.json" ]]; then
    cp "$clip_dir/label.json" "$DST/$clip/label.json"
  fi
done

echo "Done. Grayscale clips in $DST"
