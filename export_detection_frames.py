"""Export video frames into detection_data/ for YOLO labeling."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
from tqdm import tqdm

from utils import ensure_dir, list_clips


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export frames from clips for YOLO annotation")
    p.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root folder with clip_XXX/224p.mp4 directories",
    )
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help="Single video path (overrides --data-root scan)",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="detection_data",
        help="Dataset root (writes images/train|val)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=25,
        help="Save every Nth frame (25 = 1 frame/sec at 25 FPS)",
    )
    p.add_argument(
        "--max-per-clip",
        type=int,
        default=30,
        help="Max frames exported per clip",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of clips assigned to val split",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def export_video_frames(
    video_path: Path,
    clip_id: str,
    out_dir: Path,
    stride: int,
    max_per_clip: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ensure_dir(out_dir)
    saved = 0
    frame_idx = 0

    while saved < max_per_clip:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            out_name = f"{clip_id}_f{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_dir / out_name), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_root = Path(args.output_root)

    if args.video:
        video_path = Path(args.video)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        clip_id = video_path.parent.name if video_path.name == "224p.mp4" else video_path.stem
        split = "train"
        out_dir = output_root / "images" / split
        n = export_video_frames(
            video_path, clip_id, out_dir, args.stride, args.max_per_clip
        )
        print(f"Exported {n} frames to {out_dir}")
        print("Add matching labels under detection_data/labels/train/")
        return

    clips = list_clips(args.data_root)
    if not clips:
        raise RuntimeError(f"No clips found under {args.data_root}")

    val_count = max(1, int(round(len(clips) * args.val_ratio)))
    val_ids = set(c.clip_id for c in rng.sample(clips, val_count))

    total = 0
    for clip in tqdm(clips, desc="Export frames"):
        split = "val" if clip.clip_id in val_ids else "train"
        out_dir = output_root / "images" / split
        total += export_video_frames(
            clip.video_path,
            clip.clip_id,
            out_dir,
            args.stride,
            args.max_per_clip,
        )

    print(f"Exported {total} frames from {len(clips)} clips")
    print(f"Val clips ({len(val_ids)}): {sorted(val_ids)}")
    print("Next: annotate boxes, save .txt files to labels/train and labels/val")


if __name__ == "__main__":
    main()
