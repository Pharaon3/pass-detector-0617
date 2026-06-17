#!/usr/bin/env bash
# Build a 30s test clip: clip content from 1s offset + 1s black tail.
# Preserves original width, height, and sample aspect ratio.
#
# Usage:
#   bash make_test_video.sh
#   bash make_test_video.sh data/clip_13/224p.mp4 data/test_videos/clip_13_from1s_black1s.mp4

set -euo pipefail

SRC="${1:-data/clip_4/224p.mp4}"
OUT="${2:-data/test_videos/clip_4_from1s_black1s.mp4}"
OFFSET_SEC="${3:-1}"
MAIN_SEC="${4:-29}"
BLACK_SEC="${5:-1}"

mkdir -p "$(dirname "$OUT")"

W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$SRC")
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$SRC")
SAR=$(ffprobe -v error -select_streams v:0 -show_entries stream=sample_aspect_ratio -of csv=p=0 "$SRC")

if [[ -z "$SAR" || "$SAR" == "N/A" || "$SAR" == "0:1" ]]; then
  SAR_FILTER="1"
else
  SAR_FILTER="${SAR/:/\/}"
fi

echo "Source : $SRC"
echo "Output : $OUT"
echo "Size   : ${W}x${H}"
echo "SAR    : $SAR (filter: $SAR_FILTER)"
echo "Layout : ${OFFSET_SEC}s skip + ${MAIN_SEC}s content + ${BLACK_SEC}s black"

ffmpeg -y \
  -i "$SRC" \
  -f lavfi -i "color=c=black:s=${W}x${H}:r=25:d=${BLACK_SEC}" \
  -filter_complex \
  "[0:v]trim=start=${OFFSET_SEC}:duration=${MAIN_SEC},setpts=PTS-STARTPTS,format=yuv420p,setsar=${SAR_FILTER}[vmain]; \
   [1:v]format=yuv420p,setsar=${SAR_FILTER}[vblack]; \
   [vmain][vblack]concat=n=2:v=1:a=0,setsar=${SAR_FILTER}[vout]" \
  -map "[vout]" -c:v libx264 -pix_fmt yuv420p -r 25 \
  "$OUT"

ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,sample_aspect_ratio,r_frame_rate,duration \
  -of default=noprint_wrappers=1 "$OUT"

echo "Done: $OUT"
