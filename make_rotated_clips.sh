#!/usr/bin/env bash
# Rotate all clips by ±N° and zoom/crop to remove black corners (keeps original WxH).
#
# Usage:
#   bash make_rotated_clips.sh
#   FORCE=1 bash make_rotated_clips.sh   # overwrite existing rotated clips

set -euo pipefail

SRC="${1:-data}"
DST_LEFT="${2:-data_rotated_left}"
DST_RIGHT="${3:-data_rotated_right}"
ANGLE="${4:-5}"
FORCE="${FORCE:-0}"

# Zoom after rotation so a center crop at WxH has no black wedges.
# Must satisfy BOTH width and height (use max of the two factors).
compute_zoom_dims() {
  local W="$1" H="$2" DEG="$3"
  python3 - "$W" "$H" "$DEG" <<'PY'
import math, sys
w, h, deg = map(float, sys.argv[1:4])
a = math.radians(abs(deg))
# Largest axis-aligned content rect inside WxH after rotate by a:
#   w_in = w*cos(a) - h*sin(a),  h_in = h*cos(a) - w*sin(a)
# Zoom so S*w_in >= w AND S*h_in >= h:
zoom_w = 1.0 / (math.cos(a) - math.sin(a) * h / w)
zoom_h = 1.0 / (math.cos(a) - math.sin(a) * w / h)
zoom = max(zoom_w, zoom_h) * 1.02  # 2% safety for rounding
sw = int(math.ceil(w * zoom))
sh = int(math.ceil(h * zoom))
sw += sw % 2
sh += sh % 2
print(f"{zoom:.6f} {sw} {sh}")
PY
}

rotate_clip() {
  local src_video="$1"
  local dst_video="$2"
  local angle="$3"   # degrees: negative = left (CCW), positive = right (CW)

  local W H ZOOM SW SH
  W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of csv=p=0 "$src_video")
  H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$src_video")
  read -r ZOOM SW SH <<< "$(compute_zoom_dims "$W" "$H" "$angle")"

  # 1) rotate (black fill in original canvas)
  # 2) zoom in so rotated content covers the frame
  # 3) center-crop back to original resolution
  ffmpeg -y -loglevel error -i "$src_video" \
    -vf "rotate=${angle}*PI/180:fillcolor=black,scale=${SW}:${SH},crop=${W}:${H}:(iw-${W})/2:(ih-${H})/2" \
    -c:v libx264 -pix_fmt yuv420p \
    "$dst_video"
}

echo "Source : $SRC"
echo "Left   : $DST_LEFT  (-${ANGLE} deg)"
echo "Right  : $DST_RIGHT (+${ANGLE} deg)"
[[ "$FORCE" == "1" ]] && echo "FORCE  : overwrite existing"

for clip_dir in "$SRC"/clip_*; do
  [[ -d "$clip_dir" ]] || continue
  clip="$(basename "$clip_dir")"
  src_video="$clip_dir/224p.mp4"
  [[ -f "$src_video" ]] || continue

  mkdir -p "$DST_LEFT/$clip" "$DST_RIGHT/$clip"
  left_video="$DST_LEFT/$clip/224p.mp4"
  right_video="$DST_RIGHT/$clip/224p.mp4"

  if [[ ! -f "$left_video" || "$FORCE" == "1" ]]; then
    echo "Rotate left  (-${ANGLE}°): $clip"
    rotate_clip "$src_video" "$left_video" "-${ANGLE}"
  else
    echo "Skip left  $clip (exists, use FORCE=1 to rebuild)"
  fi

  if [[ ! -f "$right_video" || "$FORCE" == "1" ]]; then
    echo "Rotate right (+${ANGLE}°): $clip"
    rotate_clip "$src_video" "$right_video" "${ANGLE}"
  else
    echo "Skip right $clip (exists, use FORCE=1 to rebuild)"
  fi

  if [[ -f "$clip_dir/label.json" ]]; then
    cp "$clip_dir/label.json" "$DST_LEFT/$clip/label.json"
    cp "$clip_dir/label.json" "$DST_RIGHT/$clip/label.json"
  fi
done

echo "Done."
echo "  $DST_LEFT"
echo "  $DST_RIGHT"
