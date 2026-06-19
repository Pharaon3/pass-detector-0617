"""Extract smoothed player/ball tracks for all clips (YOLO + ByteTrack)."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from detect_ball_players import load_yolo, resolve_model_path
from tracks.extract import extract_and_cache_clip
from utils import ensure_dir, list_clips, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract track cache JSON per clip")
    p.add_argument("--config", type=str, default="config_tracks.yaml")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--video", type=str, default=None, help="Single clip folder or video path")
    p.add_argument("--force", action="store_true", help="Re-extract even if cache exists")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    track_cfg = cfg["tracks"]
    yolo_cfg = cfg.get("yolo", {})

    cache_dir = ensure_dir(track_cfg["cache_dir"])
    model = load_yolo(resolve_model_path(yolo_cfg.get("model")), args.device)

    extract_kwargs = dict(
        conf=track_cfg.get("yolo_conf", 0.3),
        imgsz=track_cfg.get("yolo_imgsz", 640),
        device=args.device,
        median_kernel=track_cfg.get("median_kernel", 5),
        max_speed=track_cfg.get("max_speed", 60.0),
        ball_max_speed=track_cfg.get("ball_max_speed", 80.0),
    )

    if args.video:
        from detect_ball_players import resolve_video_path
        from utils import clip_id_from_video_path, get_video_frame_count

        video_path = resolve_video_path(args.video)
        clip_id = clip_id_from_video_path(video_path, data_cfg["label_filename"])
        from utils import ClipRecord

        clip = ClipRecord(
            clip_id=clip_id,
            clip_dir=video_path.parent,
            video_path=video_path,
            label_path=video_path.parent / data_cfg["label_filename"],
            num_frames=get_video_frame_count(video_path),
        )
        out = extract_and_cache_clip(clip, cache_dir, model, force=args.force, **extract_kwargs)
        print(f"Wrote {out}")
        return

    data_root = args.data_root or data_cfg["data_root"]
    clips = list_clips(
        data_root,
        video_filename=data_cfg["video_filename"],
        label_filename=data_cfg["label_filename"],
        clip_prefix=data_cfg["clip_prefix"],
    )
    if not clips:
        raise RuntimeError(f"No clips under {data_root}")

    for clip in tqdm(clips, desc="Extract tracks"):
        extract_and_cache_clip(clip, cache_dir, model, force=args.force, **extract_kwargs)

    print(f"Track cache written to {cache_dir} ({len(clips)} clips)")


if __name__ == "__main__":
    main()
