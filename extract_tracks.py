"""
Extract smoothed player/ball tracks for all clips (YOLO + ByteTrack).

Default: 7-second sliding-window track caches for training/inference.
Use --full-clip for legacy full-30s clip caches (plot_player_tracks, etc.).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from detect_ball_players import load_yolo, resolve_model_path
from tracks.extract import extract_and_cache_all_windows, extract_and_cache_clip
from utils import ClipRecord, ensure_dir, list_clips, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract track cache JSON per clip or per 7s window")
    p.add_argument("--config", type=str, default="config_tracks.yaml")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--video", type=str, default=None, help="Single clip folder or video path")
    p.add_argument("--force", action="store_true", help="Re-extract even if cache exists")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="With --windows: skip windows that already have cache files",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract 7s window track caches (default: true). Required for train_tracks.py.",
    )
    p.add_argument(
        "--full-clip",
        action="store_true",
        help="Also write full-clip tracks_cache/{clip_id}.json",
    )
    p.add_argument(
        "--all-folders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include every data subfolder with 224p.mp4 (default: true).",
    )
    p.add_argument(
        "--clip-prefix",
        type=str,
        default=None,
        help="Override folder prefix when --no-all-folders.",
    )
    return p.parse_args()


def _clip_from_video_arg(video_arg: str, data_cfg: dict) -> ClipRecord:
    from detect_ball_players import resolve_video_path
    from utils import clip_id_from_video_path, get_video_frame_count

    video_path = resolve_video_path(video_arg)
    clip_id = clip_id_from_video_path(video_path, data_cfg["label_filename"])
    return ClipRecord(
        clip_id=clip_id,
        clip_dir=video_path.parent,
        video_path=video_path,
        label_path=video_path.parent / data_cfg["label_filename"],
        num_frames=get_video_frame_count(video_path),
    )


def _extract_clip(
    clip: ClipRecord,
    cfg: dict,
    model,
    args: argparse.Namespace,
    extract_kwargs: dict,
) -> None:
    track_cfg = cfg["tracks"]
    sw_cfg = cfg["sliding_window"]
    data_cfg = cfg["data"]

    if args.windows:
        window_dir = ensure_dir(track_cfg["window_cache_dir"])
        extract_and_cache_all_windows(
            clip,
            window_dir,
            model,
            window_frames=sw_cfg["window_frames"],
            stride_frames=sw_cfg["stride_frames"],
            num_frames=data_cfg.get("num_frames"),
            force=args.force and not args.skip_existing,
            **extract_kwargs,
        )

    if args.full_clip:
        clip_dir = ensure_dir(track_cfg["cache_dir"])
        extract_and_cache_clip(
            clip,
            clip_dir,
            model,
            force=args.force,
            **extract_kwargs,
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    track_cfg = cfg["tracks"]
    yolo_cfg = cfg.get("yolo", {})

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
        clip = _clip_from_video_arg(args.video, data_cfg)
        _extract_clip(clip, cfg, model, args, extract_kwargs)
        if args.windows:
            print(f"Window caches: {track_cfg['window_cache_dir']}/{clip.clip_id}/")
        if args.full_clip:
            print(f"Full clip cache: {track_cfg['cache_dir']}/{clip.clip_id}.json")
        return

    data_root = args.data_root or data_cfg["data_root"]
    if args.all_folders:
        clip_prefix: str | None = ""
    elif args.clip_prefix is not None:
        clip_prefix = args.clip_prefix
    else:
        clip_prefix = data_cfg.get("clip_prefix", "clip_")

    clips = list_clips(
        data_root,
        video_filename=data_cfg["video_filename"],
        label_filename=data_cfg["label_filename"],
        clip_prefix=clip_prefix,
    )
    if not clips:
        scope = "all folders" if not clip_prefix else f"folders matching '{clip_prefix}*'"
        raise RuntimeError(f"No clips under {data_root} ({scope})")

    for clip in tqdm(clips, desc="Extract tracks"):
        _extract_clip(clip, cfg, model, args, extract_kwargs)

    if args.windows:
        print(f"Window track caches: {track_cfg['window_cache_dir']}/ ({len(clips)} clips)")
    if args.full_clip:
        print(f"Full clip caches: {track_cfg['cache_dir']}/ ({len(clips)} clips)")
    print(f"Folders: {[c.clip_id for c in clips]}")


if __name__ == "__main__":
    main()
