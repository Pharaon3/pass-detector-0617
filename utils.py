"""Shared utilities for soccer pass detection."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml


@dataclass
class ClipRecord:
    clip_id: str
    clip_dir: Path
    video_path: Path
    label_path: Path
    num_frames: int = 750


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_video_frame_count(video_path: str | Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def list_clips(
    data_root: str | Path,
    video_filename: str = "224p.mp4",
    label_filename: str = "label.json",
    clip_prefix: str = "clip_",
) -> list[ClipRecord]:
    """
    Discover clip folders under data_root.

    Expected layout:
      data/clip_4/224p.mp4
      data/clip_4/label.json
    """
    data_root = Path(data_root)
    if not data_root.exists():
        return []

    clips: list[ClipRecord] = []
    for clip_dir in sorted(data_root.iterdir()):
        if not clip_dir.is_dir() or not clip_dir.name.startswith(clip_prefix):
            continue

        video_path = clip_dir / video_filename
        label_path = clip_dir / label_filename
        if not video_path.exists() or not label_path.exists():
            continue

        num_frames = get_video_frame_count(video_path)
        clips.append(
            ClipRecord(
                clip_id=clip_dir.name,
                clip_dir=clip_dir,
                video_path=video_path,
                label_path=label_path,
                num_frames=num_frames,
            )
        )
    return clips


def ms_to_frame(position_ms: int | float, fps: int = 25) -> int:
    """Convert annotation position (milliseconds) to frame index."""
    return int(round(float(position_ms) / 1000.0 * fps))


def frame_to_sec(frame_idx: int, fps: int = 25) -> float:
    return frame_idx / fps


def load_annotation(path: str | Path) -> list[dict[str, Any]]:
    """Load PASS events from the observation section only (in-clip labels)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [e for e in data.get("observation", []) if e.get("label", "").upper() == "PASS"]


def pass_frames_from_annotation(
    annotation_path: str | Path,
    fps: int = 25,
    num_frames: int = 750,
) -> list[int]:
    """Extract pass center frames from annotation JSON."""
    events = load_annotation(annotation_path)
    frames = []
    for ev in events:
        pos = ev.get("position")
        if pos is None:
            continue
        f = ms_to_frame(pos, fps)
        if 0 <= f < num_frames:
            frames.append(f)
    return sorted(frames)


def build_frame_labels(
    pass_frames: list[int],
    num_frames: int,
    radius_sec: float = 0.5,
    fps: int = 25,
    smooth: bool = True,
) -> np.ndarray:
    """
    Build per-frame binary / soft labels.
    Gaussian smoothing ±radius_sec around each pass center.
    """
    labels = np.zeros(num_frames, dtype=np.float32)
    if not pass_frames:
        return labels

    radius_frames = int(radius_sec * fps)
    if radius_frames < 1:
        radius_frames = 1

    sigma = radius_frames / 2.0
    xs = np.arange(-radius_frames, radius_frames + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (xs / max(sigma, 1e-6)) ** 2)

    for center in pass_frames:
        start = max(0, center - radius_frames)
        end = min(num_frames, center + radius_frames + 1)
        k_start = radius_frames - (center - start)
        k_end = k_start + (end - start)
        if smooth:
            labels[start:end] = np.maximum(labels[start:end], kernel[k_start:k_end])
        else:
            labels[start:end] = 1.0

    return np.clip(labels, 0.0, 1.0)


def save_json(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def list_videos(video_dir: str | Path, extensions: tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov")) -> list[Path]:
    """Legacy flat-directory video listing (prefer list_clips for this dataset)."""
    video_dir = Path(video_dir)
    if not video_dir.exists():
        return []
    files = []
    for ext in extensions:
        files.extend(video_dir.glob(f"*{ext}"))
    return sorted(files)


def annotation_path_for_video(video_path: Path, annotation_dir: Path) -> Path | None:
    """Legacy flat-directory annotation lookup."""
    stem = video_path.stem
    for ext in (".json",):
        candidate = annotation_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def get_device(require_cuda: bool = False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError(
            "CUDA is not available. Run:  python check_gpu.py\n"
            "Then reinstall PyTorch with CUDA:  bash setup_gpu.sh"
        )
    return torch.device("cpu")


def print_device_info(device: torch.device) -> None:
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB VRAM)")
        print(f"PyTorch CUDA: {torch.version.cuda}")
    else:
        print("WARNING: running on CPU. For GPU training run:  python check_gpu.py")
