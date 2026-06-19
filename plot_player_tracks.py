"""Track players across a clip and render trajectory paths on one image."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from detect_ball_players import (
    load_yolo,
    parse_class_filter,
    resolve_model_path,
    resolve_video_path,
)
from utils import clip_id_from_video_path, ensure_dir, get_video_frame_count

TRACK_CLASSES = {"player", "goalkeeper"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot per-player movement tracks on one image")
    p.add_argument("--video", type=str, required=True, help="Video path or clip folder")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG (default: outputs/tracks/{clip_id}_tracks.png)",
    )
    p.add_argument("--model", type=str, default=None, help="YOLO weights path")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument(
        "--classes",
        type=str,
        default="player,goalkeeper",
        help="Classes to track (default: player,goalkeeper)",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--background",
        type=str,
        default="last",
        choices=("last", "first", "dim", "none"),
        help="Background frame: last/first/dimmed first/black canvas",
    )
    p.add_argument(
        "--trail-alpha",
        type=float,
        default=0.35,
        help="Line opacity on background (0-1)",
    )
    p.add_argument(
        "--min-track-len",
        type=int,
        default=8,
        help="Skip tracks shorter than this many detections",
    )
    p.add_argument("--json", type=str, default=None, help="Optional track JSON output")
    return p.parse_args()


def track_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    hue = int(rng.integers(0, 180))
    color = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(color[0]), int(color[1]), int(color[2])


def pick_background(frames: list[np.ndarray], mode: str) -> np.ndarray:
    if not frames:
        raise RuntimeError("No frames read from video")
    if mode == "first":
        return frames[0].copy()
    if mode == "last":
        return frames[-1].copy()
    if mode == "dim":
        bg = frames[0].copy().astype(np.float32)
        bg = np.clip(bg * 0.35, 0, 255).astype(np.uint8)
        return bg
    h, w = frames[0].shape[:2]
    return np.zeros((h, w, 3), dtype=np.uint8)


def draw_tracks(
    canvas: np.ndarray,
    tracks: dict[int, list[tuple[int, int, int]]],
    min_track_len: int,
    trail_alpha: float,
) -> np.ndarray:
    """Draw colored polylines; each point is (frame_idx, x, y)."""
    out = canvas.copy()
    overlay = canvas.copy()

    for track_id, points in sorted(tracks.items(), key=lambda kv: kv[0]):
        if len(points) < min_track_len:
            continue
        color = track_color(track_id)
        xy = np.array([(x, y) for _, x, y in points], dtype=np.int32)

        for i in range(1, len(xy)):
            cv2.line(overlay, tuple(xy[i - 1]), tuple(xy[i]), color, 2, cv2.LINE_AA)

        cv2.circle(overlay, tuple(xy[0]), 5, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(xy[-1]), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(xy[-1]), 5, color, 2, cv2.LINE_AA)

        lx, ly = int(xy[-1, 0]), int(max(xy[-1, 1] - 8, 12))
        cv2.putText(
            overlay,
            f"id {track_id}",
            (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, trail_alpha, out, 1.0 - trail_alpha, 0, out)
    return out


def run_tracking(
    video_path: Path,
    model,
    conf: float,
    class_filter: set[str],
    device: str | None,
    max_frames: int | None,
    imgsz: int,
) -> tuple[dict[int, list[tuple[int, int, int]]], list[np.ndarray], float, int]:
    model_names = {int(k): v.lower() for k, v in model.names.items()}
    track_classes = {c for c in class_filter if c in TRACK_CLASSES or c in model_names.values()}
    if not track_classes:
        track_classes = TRACK_CLASSES & set(model_names.values())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = get_video_frame_count(video_path)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    tracks: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    frames: list[np.ndarray] = []
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc=f"Track {video_path.name}")
    while frame_idx < total_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame.copy())

        results = model.track(
            source=frame,
            persist=True,
            conf=conf,
            imgsz=imgsz,
            verbose=False,
            device=device,
            tracker="bytetrack.yaml",
        )
        result = results[0]

        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = model_names[cls_id]
                if class_name not in track_classes:
                    continue
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = int(0.5 * (x1 + x2)), int(0.5 * (y1 + y2))
                tracks[track_id].append((frame_idx, cx, cy))

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    return tracks, frames, fps, frame_idx


def tracks_to_json(
    tracks: dict[int, list[tuple[int, int, int]]],
    fps: float,
    video_path: Path,
) -> dict:
    out_tracks = []
    for track_id, points in sorted(tracks.items()):
        out_tracks.append(
            {
                "track_id": track_id,
                "num_points": len(points),
                "points": [
                    {
                        "frame": fr,
                        "time_ms": round(1000.0 * fr / fps, 2),
                        "x": x,
                        "y": y,
                    }
                    for fr, x, y in points
                ],
            }
        )
    return {
        "video": str(video_path.resolve()),
        "fps": fps,
        "num_tracks": len(out_tracks),
        "tracks": out_tracks,
    }


def main() -> None:
    args = parse_args()
    video_path = resolve_video_path(args.video)
    clip_id = clip_id_from_video_path(video_path)

    output_png = Path(args.output) if args.output else Path("outputs/tracks") / f"{clip_id}_tracks.png"
    output_json = Path(args.json) if args.json else output_png.with_suffix(".json")

    model = load_yolo(resolve_model_path(args.model), args.device)
    class_filter = parse_class_filter(args.classes, model.names)

    tracks, frames, fps, num_frames = run_tracking(
        video_path=video_path,
        model=model,
        conf=args.conf,
        class_filter=class_filter,
        device=args.device,
        max_frames=args.max_frames,
        imgsz=args.imgsz,
    )

    background = pick_background(frames, args.background)
    image = draw_tracks(
        background,
        tracks,
        min_track_len=args.min_track_len,
        trail_alpha=args.trail_alpha,
    )

    cv2.putText(
        image,
        f"{clip_id} | {num_frames} frames | {len(tracks)} tracks",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    ensure_dir(output_png.parent)
    cv2.imwrite(str(output_png), image)

    track_data = tracks_to_json(tracks, fps, video_path)
    ensure_dir(output_json.parent)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(track_data, f, indent=2)

    kept = sum(1 for pts in tracks.values() if len(pts) >= args.min_track_len)
    print(f"Wrote track image: {output_png}")
    print(f"Wrote track JSON:  {output_json}")
    print(f"Tracks kept (len >= {args.min_track_len}): {kept}/{len(tracks)}")


if __name__ == "__main__":
    main()
