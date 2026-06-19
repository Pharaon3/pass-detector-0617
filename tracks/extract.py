"""Run YOLO + ByteTrack and save smoothed track cache per clip."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from tracks.processing import TrackPoint, smooth_ball_series, smooth_track_points
from utils import ClipRecord, ensure_dir, get_video_frame_count

TRACKABLE_CLASSES = {"player", "goalkeeper"}


def extract_clip_tracks(
    clip: ClipRecord,
    model,
    conf: float = 0.3,
    imgsz: int = 640,
    device: str | None = None,
    median_kernel: int = 5,
    max_speed: float = 60.0,
    ball_max_speed: float = 80.0,
    max_frames: int | None = None,
) -> dict[str, Any]:
    model_names = {int(k): v.lower() for k, v in model.names.items()}

    cap = cv2.VideoCapture(str(clip.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {clip.video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = get_video_frame_count(clip.video_path)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    raw_tracks: dict[int, list[TrackPoint]] = defaultdict(list)
    raw_track_class: dict[int, str] = {}
    ball_raw: dict[int, tuple[float, float]] = {}

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc=f"Track {clip.clip_id}", leave=False)
    while frame_idx < total_frames:
        ok, frame = cap.read()
        if not ok:
            break

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
            best_ball_conf = -1.0
            best_ball_xy: tuple[float, float] | None = None

            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = model_names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

                if class_name == "ball":
                    conf_val = float(box.conf[0])
                    if conf_val > best_ball_conf:
                        best_ball_conf = conf_val
                        best_ball_xy = (float(cx), float(cy))
                    continue

                if class_name not in TRACKABLE_CLASSES:
                    continue
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                raw_track_class[track_id] = class_name
                raw_tracks[track_id].append(TrackPoint(frame=frame_idx, x=cx, y=cy))

            if best_ball_xy is not None:
                ball_raw[frame_idx] = best_ball_xy

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    smoothed_tracks = []
    for track_id, points in sorted(raw_tracks.items()):
        smooth_pts = smooth_track_points(
            points,
            median_kernel=median_kernel,
            max_speed=max_speed,
            fill_gaps=True,
        )
        if len(smooth_pts) < 2:
            continue
        smoothed_tracks.append(
            {
                "track_id": int(track_id),
                "class": raw_track_class.get(track_id, "player"),
                "points": [
                    {"frame": p.frame, "x": round(p.x, 2), "y": round(p.y, 2)} for p in smooth_pts
                ],
            }
        )

    ball_smooth = smooth_ball_series(
        ball_raw,
        num_frames=total_frames,
        median_kernel=max(3, median_kernel - 2),
        max_speed=ball_max_speed,
    )
    ball_track = [
        {"frame": fr, "x": round(xy[0], 2), "y": round(xy[1], 2)}
        for fr, xy in sorted(ball_smooth.items())
    ]

    return {
        "clip_id": clip.clip_id,
        "video": str(clip.video_path.resolve()),
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": total_frames,
        "tracks": smoothed_tracks,
        "ball_track": ball_track,
    }


def save_track_cache(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_track_cache(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cache_path_for_clip(cache_dir: Path, clip_id: str) -> Path:
    return cache_dir / f"{clip_id}.json"


def extract_and_cache_clip(
    clip: ClipRecord,
    cache_dir: Path,
    model,
    force: bool = False,
    **kwargs: Any,
) -> Path:
    out_path = cache_path_for_clip(cache_dir, clip.clip_id)
    if out_path.is_file() and not force:
        return out_path

    data = extract_clip_tracks(clip, model, **kwargs)
    save_track_cache(data, out_path)
    return out_path
