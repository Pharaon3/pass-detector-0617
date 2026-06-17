#!/usr/bin/env python3
"""
Find which videos contain footage matching a segment of a query video.

Compares downscaled grayscale frame fingerprints with a sliding window.

Example:
  python find_matching_segment.py --query-clip clip_235 --start-sec 20 --end-sec 30
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils import list_clips, load_config


@dataclass
class MatchResult:
    clip_id: str
    video_path: Path
    offset_sec: float
    score: float
    end_sec: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search for matching video segments")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--data-root", type=str, default=None, help="Override config data root")
    p.add_argument("--query-clip", type=str, default=None, help="Query clip id, e.g. clip_235")
    p.add_argument("--query-video", type=str, default=None, help="Or direct path to query video")
    p.add_argument("--start-sec", type=float, required=True, help="Query segment start (seconds)")
    p.add_argument("--end-sec", type=float, required=True, help="Query segment end (seconds)")
    p.add_argument("--sample-step-sec", type=float, default=1.0, help="Sample one frame every N seconds")
    p.add_argument("--search-step-sec", type=float, default=0.5, help="Sliding window step in candidates")
    p.add_argument("--fingerprint-size", type=int, default=64, help="Square resize for frame fingerprint")
    p.add_argument("--min-score", type=float, default=0.90, help="Minimum similarity to report")
    p.add_argument("--top-k", type=int, default=15, help="Max results to print")
    p.add_argument("--exclude-self", action="store_true", help="Skip the query clip in search")
    return p.parse_args()


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap


def video_meta(cap: cv2.VideoCapture) -> tuple[float, int]:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, n


def read_frame(cap: cv2.VideoCapture, frame_idx: int, size: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    if not ok:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 = identical, 0.0 = very different."""
    mse = float(np.mean((a - b) ** 2))
    return max(0.0, 1.0 - mse * 4.0)


def extract_segment_fingerprints(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    sample_step_sec: float,
    fingerprint_size: int,
) -> tuple[list[np.ndarray], list[float], float]:
    cap = open_video(video_path)
    fps, n_frames = video_meta(cap)
    duration = n_frames / fps

    if end_sec <= start_sec:
        cap.release()
        raise ValueError("end-sec must be greater than start-sec")

    start_sec = max(0.0, start_sec)
    end_sec = min(end_sec, duration)
    seg_len = end_sec - start_sec

    times = np.arange(start_sec, end_sec, sample_step_sec, dtype=np.float64)
    if len(times) == 0 or times[-1] < end_sec - 1e-6:
        times = np.append(times, end_sec - 1e-3)

    fingerprints: list[np.ndarray] = []
    rel_times: list[float] = []
    for t in times:
        frame_idx = int(round(t * fps))
        fp = read_frame(cap, frame_idx, fingerprint_size)
        if fp is not None:
            fingerprints.append(fp)
            rel_times.append(t - start_sec)

    cap.release()
    if not fingerprints:
        raise RuntimeError(f"No frames extracted from {video_path} [{start_sec}, {end_sec}]")

    return fingerprints, rel_times, seg_len


def score_alignment(
    cap: cv2.VideoCapture,
    fps: float,
    offset_sec: float,
    query_fps: list[np.ndarray],
    rel_times: list[float],
    fingerprint_size: int,
) -> float:
    scores = []
    for fp_q, rel_t in zip(query_fps, rel_times):
        frame_idx = int(round((offset_sec + rel_t) * fps))
        fp_c = read_frame(cap, frame_idx, fingerprint_size)
        if fp_c is None:
            return 0.0
        scores.append(frame_similarity(fp_q, fp_c))
    return float(np.mean(scores))


def search_clip(
    candidate: Path,
    clip_id: str,
    query_fps: list[np.ndarray],
    rel_times: list[float],
    seg_len: float,
    search_step_sec: float,
    fingerprint_size: int,
) -> MatchResult | None:
    cap = open_video(candidate)
    fps, n_frames = video_meta(cap)
    duration = n_frames / fps

    best_score = -1.0
    best_offset = 0.0
    offset = 0.0
    last_offset = max(0.0, duration - seg_len)

    while offset <= last_offset + 1e-6:
        score = score_alignment(cap, fps, offset, query_fps, rel_times, fingerprint_size)
        if score > best_score:
            best_score = score
            best_offset = offset
        offset += search_step_sec

    cap.release()
    return MatchResult(
        clip_id=clip_id,
        video_path=candidate,
        offset_sec=best_offset,
        score=best_score,
        end_sec=best_offset + seg_len,
    )


def resolve_query_video(args: argparse.Namespace, cfg: dict) -> tuple[Path, str]:
    if args.query_video:
        path = Path(args.query_video)
        clip_id = path.parent.name if path.parent.name.startswith("clip_") else path.stem
        return path, clip_id
    if args.query_clip:
        data_root = Path(args.data_root or cfg["data"]["data_root"])
        clips = list_clips(
            data_root,
            video_filename=cfg["data"]["video_filename"],
            label_filename=cfg["data"]["label_filename"],
            clip_prefix=cfg["data"]["clip_prefix"],
        )
        match = next((c for c in clips if c.clip_id == args.query_clip), None)
        if match is None:
            raise RuntimeError(f"Query clip not found: {args.query_clip}")
        return match.video_path, match.clip_id
    raise SystemExit("Provide --query-clip or --query-video")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = Path(args.data_root or cfg["data"]["data_root"])

    query_path, query_id = resolve_query_video(args, cfg)
    print(f"Query: {query_id} ({query_path})")
    print(f"Segment: {args.start_sec:.2f}s - {args.end_sec:.2f}s")

    query_fps, rel_times, seg_len = extract_segment_fingerprints(
        query_path,
        args.start_sec,
        args.end_sec,
        args.sample_step_sec,
        args.fingerprint_size,
    )
    print(f"Fingerprints: {len(query_fps)} frames, segment length {seg_len:.2f}s")

    clips = list_clips(
        data_root,
        video_filename=cfg["data"]["video_filename"],
        label_filename=cfg["data"]["label_filename"],
        clip_prefix=cfg["data"]["clip_prefix"],
    )
    if not clips:
        raise RuntimeError(f"No clips found under {data_root}")

    results: list[MatchResult] = []
    for clip in tqdm(clips, desc="Search clips"):
        if args.exclude_self and clip.clip_id == query_id:
            continue
        hit = search_clip(
            clip.video_path,
            clip.clip_id,
            query_fps,
            rel_times,
            seg_len,
            args.search_step_sec,
            args.fingerprint_size,
        )
        if hit and hit.score >= args.min_score:
            results.append(hit)

    results.sort(key=lambda r: r.score, reverse=True)

    print("\n=== Matches ===")
    if not results:
        print(f"No matches above score {args.min_score:.2f}. Try lowering --min-score.")
        return

    for i, r in enumerate(results[: args.top_k], start=1):
        print(
            f"{i:2d}. {r.clip_id:12s}  score={r.score:.4f}  "
            f"offset={r.offset_sec:6.2f}s - {r.end_sec:6.2f}s  ({r.video_path})"
        )

    top = results[0]
    if top.clip_id != query_id and abs(top.offset_sec - args.start_sec) < args.search_step_sec:
        print("\nNote: best match aligns near the same absolute time in another clip.")
    if top.clip_id == query_id and abs(top.offset_sec - args.start_sec) < args.search_step_sec:
        print("\nSelf-match at expected offset (sanity check OK).")


if __name__ == "__main__":
    main()
