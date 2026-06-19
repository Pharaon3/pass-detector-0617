#!/usr/bin/env bash
# Build 1s-offset test clips: skip first 1s of source, 29s content + 1s black tail (30s total).
# Keeps original width, height, SAR, and copies label.json.
#
# Usage:
#   bash make_shifted_clips.sh
#   bash make_shifted_clips.sh data data_shifted_1s 1 29 1
set -euo pipefail

SRC="${1:-data}"
DST="${2:-data_shifted_1s}"
OFFSET_SEC="${3:-1}"
MAIN_SEC="${4:-29}"
BLACK_SEC="${5:-1}"

echo "Source : $SRC"
echo "Output : $DST"
echo "Layout : ${OFFSET_SEC}s skip + ${MAIN_SEC}s content + ${BLACK_SEC}s black"

for clip_dir in "$SRC"/*; do
  [[ -d "$clip_dir" ]] || continue
  clip="$(basename "$clip_dir")"
  src_video="$clip_dir/224p.mp4"
  [[ -f "$src_video" ]] || continue

  mkdir -p "$DST/$clip"
  dst_video="$DST/$clip/224p.mp4"

  if [[ -f "$dst_video" ]]; then
    echo "Skip $clip (already exists)"
  else
    echo "Shift +${OFFSET_SEC}s: $clip"
    W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$src_video")
    H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$src_video")
    SAR=$(ffprobe -v error -select_streams v:0 -show_entries stream=sample_aspect_ratio -of csv=p=0 "$src_video")
    if [[ -z "$SAR" || "$SAR" == "N/A" || "$SAR" == "0:1" ]]; then
      SAR_FILTER="1"
    else
      SAR_FILTER="${SAR/:/\/}"
    fi
    ffmpeg -y -loglevel error \
      -i "$src_video" \
      -f lavfi -i "color=c=black:s=${W}x${H}:r=25:d=${BLACK_SEC}" \
      -filter_complex \
      "[0:v]trim=start=${OFFSET_SEC}:duration=${MAIN_SEC},setpts=PTS-STARTPTS,format=yuv420p,setsar=${SAR_FILTER}[vmain]; \
       [1:v]format=yuv420p,setsar=${SAR_FILTER}[vblack]; \
       [vmain][vblack]concat=n=2:v=1:a=0,setsar=${SAR_FILTER}[vout]" \
      -map "[vout]" -c:v libx264 -pix_fmt yuv420p -r 25 \
      "$dst_video"
  fi

  if [[ -f "$clip_dir/label.json" ]]; then
    cp "$clip_dir/label.json" "$DST/$clip/label.json"
  fi
done

echo "Done. Shifted clips in $DST"
