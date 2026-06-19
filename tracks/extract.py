"""
Run YOLO + ByteTrack on frame sequences; save smoothed track cache (full clip or 7s window).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from dataset.sliding_window import generate_sliding_windows
from tracks.processing import TrackPoint, smooth_ball_series, smooth_track_points
from utils import ClipRecord, ensure_dir, get_video_frame_count

TRACKABLE_CLASSES = {"player", "goalkeeper"}


def _run_tracking_on_frames(
    frames: list[np.ndarray],
    model,
    model_names: dict[int, str],
    conf: float,
    imgsz: int,
    device: str | None,
) -> tuple[dict[int, list[TrackPoint]], dict[int, str], dict[int, tuple[float, float]]]:
    raw_tracks: dict[int, list[TrackPoint]] = defaultdict(list)
    raw_track_class: dict[int, str] = {}
    ball_raw: dict[int, tuple[float, float]] = {}

    for frame_idx, frame in enumerate(frames):
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

        if result.boxes is None or not len(result.boxes):
            continue

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

    return raw_tracks, raw_track_class, ball_raw


def _finalize_track_cache(
    raw_tracks: dict[int, list[TrackPoint]],
    raw_track_class: dict[int, str],
    ball_raw: dict[int, tuple[float, float]],
    *,
    clip_id: str,
    video: str,
    width: int,
    height: int,
    fps: float,
    num_frames: int,
    median_kernel: int,
    max_speed: float,
    ball_max_speed: float,
    window_start: int | None = None,
    window_frames: int | None = None,
) -> dict[str, Any]:
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
        num_frames=num_frames,
        median_kernel=max(3, median_kernel - 2),
        max_speed=ball_max_speed,
    )
    ball_track = [
        {"frame": fr, "x": round(xy[0], 2), "y": round(xy[1], 2)}
        for fr, xy in sorted(ball_smooth.items())
    ]

    out: dict[str, Any] = {
        "clip_id": clip_id,
        "video": video,
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": num_frames,
        "tracks": smoothed_tracks,
        "ball_track": ball_track,
    }
    if window_start is not None:
        out["window_start"] = int(window_start)
        out["window_frames"] = int(window_frames if window_frames is not None else num_frames)
    return out


def read_video_segment_bgr(
    video_path: Path,
    start_frame: int,
    num_frames: int,
) -> tuple[list[np.ndarray], int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    for _ in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path} at {start_frame}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return frames[:num_frames], width, height, fps


def extract_tracks_from_frames(
    frames: list[np.ndarray],
    model,
    *,
    clip_id: str = "",
    video: str = "",
    width: int | None = None,
    height: int | None = None,
    fps: float = 25.0,
    conf: float = 0.3,
    imgsz: int = 640,
    device: str | None = None,
    median_kernel: int = 5,
    max_speed: float = 60.0,
    ball_max_speed: float = 80.0,
    window_start: int | None = None,
) -> dict[str, Any]:
    if width is None or height is None:
        height, width = frames[0].shape[:2]

    model_names = {int(k): v.lower() for k, v in model.names.items()}
    raw_tracks, raw_track_class, ball_raw = _run_tracking_on_frames(
        frames, model, model_names, conf, imgsz, device
    )

    return _finalize_track_cache(
        raw_tracks,
        raw_track_class,
        ball_raw,
        clip_id=clip_id,
        video=video,
        width=width,
        height=height,
        fps=fps,
        num_frames=len(frames),
        median_kernel=median_kernel,
        max_speed=max_speed,
        ball_max_speed=ball_max_speed,
        window_start=window_start,
        window_frames=len(frames),
    )


def extract_window_tracks_from_video(
    video_path: Path,
    start_frame: int,
    num_frames: int,
    model,
    clip_id: str,
    **kwargs: Any,
) -> dict[str, Any]:
    frames, width, height, fps = read_video_segment_bgr(video_path, start_frame, num_frames)
    return extract_tracks_from_frames(
        frames,
        model,
        clip_id=clip_id,
        video=str(video_path.resolve()),
        width=width,
        height=height,
        fps=fps,
        window_start=start_frame,
        **kwargs,
    )


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
    total_frames = get_video_frame_count(clip.video_path)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    frames, width, height, fps = read_video_segment_bgr(clip.video_path, 0, total_frames)
    return extract_tracks_from_frames(
        frames,
        model,
        clip_id=clip.clip_id,
        video=str(clip.video_path.resolve()),
        width=width,
        height=height,
        fps=fps,
        conf=conf,
        imgsz=imgsz,
        device=device,
        median_kernel=median_kernel,
        max_speed=max_speed,
        ball_max_speed=ball_max_speed,
    )


def save_track_cache(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_track_cache(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cache_path_for_clip(cache_dir: Path, clip_id: str) -> Path:
    return cache_dir / f"{clip_id}.json"


def window_cache_path(cache_dir: Path, clip_id: str, start_frame: int) -> Path:
    return cache_dir / clip_id / f"w{start_frame:05d}.json"


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


def extract_and_cache_window(
    clip: ClipRecord,
    start_frame: int,
    num_frames: int,
    cache_dir: Path,
    model,
    force: bool = False,
    **kwargs: Any,
) -> Path:
    out_path = window_cache_path(cache_dir, clip.clip_id, start_frame)
    if out_path.is_file() and not force:
        return out_path

    data = extract_window_tracks_from_video(
        clip.video_path,
        start_frame,
        num_frames,
        model,
        clip.clip_id,
        **kwargs,
    )
    save_track_cache(data, out_path)
    return out_path


def extract_and_cache_all_windows(
    clip: ClipRecord,
    cache_dir: Path,
    model,
    window_frames: int = 175,
    stride_frames: int = 25,
    num_frames: int | None = 750,
    force: bool = False,
    **kwargs: Any,
) -> list[Path]:
    clip_frames = clip.num_frames
    if num_frames is not None:
        clip_frames = min(clip.num_frames, num_frames)

    windows = generate_sliding_windows(
        clip_frames,
        window_frames=window_frames,
        stride_frames=stride_frames,
    )

    paths: list[Path] = []
    for w in tqdm(windows, desc=f"Windows {clip.clip_id}", leave=False):
        paths.append(
            extract_and_cache_window(
                clip,
                w.start_frame,
                w.length,
                cache_dir,
                model,
                force=force,
                **kwargs,
            )
        )
    return paths
