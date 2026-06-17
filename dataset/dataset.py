"""PyTorch dataset for sliding-window pass detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.sliding_window import WindowSpec, generate_sliding_windows
from utils import (
    ClipRecord,
    build_frame_labels,
    list_clips,
    pass_frames_from_annotation,
)


def read_video_frames(
    video_path: Path,
    start_frame: int,
    num_frames: int,
    target_size: int = 224,
) -> np.ndarray:
    """
    Read `num_frames` RGB frames starting at `start_frame`.
    Returns (T, H, W, 3) uint8 array.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from {video_path} at frame {start_frame}")

    # Pad with last frame if video ends early
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.stack(frames[:num_frames], axis=0)


def normalize_frames(frames: np.ndarray, mean: list[float], std: list[float]) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (T, 3, H, W) float tensor."""
    x = frames.astype(np.float32) / 255.0
    x = np.transpose(x, (0, 3, 1, 2))
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
    x = (x - mean_arr) / std_arr
    return torch.from_numpy(x)


class PassWindowSample:
    video_path: Path
    window: WindowSpec
    video_id: str


class SoccerPassDataset(Dataset):
    """
    Each item = one sliding window (175 frames) with frame-level labels.
    """

    def __init__(
        self,
        data_root: str | Path,
        video_filename: str = "224p.mp4",
        label_filename: str = "label.json",
        clip_prefix: str = "clip_",
        window_frames: int = 175,
        stride_frames: int = 25,
        num_frames: int | None = 750,
        fps: int = 25,
        frame_size: int = 224,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        pass_radius_sec: float = 0.5,
        clip_ids: list[str] | None = None,
        transform=None,
    ) -> None:
        self.data_root = Path(data_root)
        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.default_num_frames = num_frames
        self.fps = fps
        self.frame_size = frame_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.pass_radius_sec = pass_radius_sec
        self.transform = transform

        clips = list_clips(
            self.data_root,
            video_filename=video_filename,
            label_filename=label_filename,
            clip_prefix=clip_prefix,
        )
        if clip_ids is not None:
            clip_id_set = set(clip_ids)
            clips = [c for c in clips if c.clip_id in clip_id_set]

        self.samples: list[tuple[ClipRecord, WindowSpec, np.ndarray, list[int]]] = []
        self._build_index(clips)

    def _build_index(self, clips: list[ClipRecord]) -> None:
        for clip in clips:
            clip_frames = clip.num_frames
            if self.default_num_frames is not None:
                clip_frames = min(clip.num_frames, self.default_num_frames)

            pass_frames = pass_frames_from_annotation(clip.label_path, self.fps, clip_frames)
            full_labels = build_frame_labels(
                pass_frames,
                clip_frames,
                radius_sec=self.pass_radius_sec,
                fps=self.fps,
                smooth=True,
            )

            windows = generate_sliding_windows(
                clip_frames,
                window_frames=self.window_frames,
                stride_frames=self.stride_frames,
            )

            for w in windows:
                window_labels = full_labels[w.start_frame : w.end_frame]
                self.samples.append((clip, w, window_labels.copy(), pass_frames))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip, window, window_labels, pass_frames = self.samples[idx]

        frames = read_video_frames(
            clip.video_path,
            start_frame=window.start_frame,
            num_frames=window.length,
            target_size=self.frame_size,
        )
        video = normalize_frames(frames, self.mean, self.std)

        if self.transform is not None:
            video = self.transform(video)

        labels = torch.from_numpy(window_labels).float()

        # Peak time targets for auxiliary MSE loss (normalized 0-1 within window)
        peak_times = []
        for pf in pass_frames:
            if window.start_frame <= pf < window.end_frame:
                local = pf - window.start_frame
                peak_times.append(local / max(window.length - 1, 1))

        if peak_times:
            peak_time_tensor = torch.tensor(peak_times, dtype=torch.float32)
        else:
            peak_time_tensor = torch.zeros(0, dtype=torch.float32)

        return {
            "video": video,  # (T, 3, H, W)
            "labels": labels,  # (T,)
            "peak_times": peak_time_tensor,
            "start_frame": window.start_frame,
            "video_id": clip.clip_id,
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    videos = torch.stack([b["video"] for b in batch], dim=0)
    start_frames = torch.tensor([b["start_frame"] for b in batch], dtype=torch.long)
    video_ids = [b["video_id"] for b in batch]
    out: dict[str, Any] = {
        "video": videos,
        "start_frame": start_frames,
        "video_id": video_ids,
    }
    if "labels" in batch[0]:
        out["labels"] = torch.stack([b["labels"] for b in batch], dim=0)
        out["peak_times"] = [b["peak_times"] for b in batch]
    return out


class SoccerPassVideoDataset(Dataset):
    """Inference dataset: one item per sliding window (no labels required)."""

    def __init__(
        self,
        video_path: str | Path,
        video_id: str | None = None,
        window_frames: int = 175,
        stride_frames: int = 25,
        num_frames: int | None = 750,
        frame_size: int = 224,
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ) -> None:
        self.video_path = Path(video_path)
        if video_id is not None:
            self.video_id = video_id
        elif self.video_path.parent.name.startswith("clip_"):
            self.video_id = self.video_path.parent.name
        else:
            self.video_id = self.video_path.stem
        self.window_frames = window_frames
        self.stride_frames = stride_frames
        if num_frames is None:
            from utils import get_video_frame_count
            self.num_frames = get_video_frame_count(self.video_path)
        else:
            from utils import get_video_frame_count
            self.num_frames = min(get_video_frame_count(self.video_path), num_frames)
        self.frame_size = frame_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

        self.windows = generate_sliding_windows(
            num_frames,
            window_frames=window_frames,
            stride_frames=stride_frames,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        window = self.windows[idx]
        frames = read_video_frames(
            self.video_path,
            start_frame=window.start_frame,
            num_frames=window.length,
            target_size=self.frame_size,
        )
        video = normalize_frames(frames, self.mean, self.std)
        return {
            "video": video,
            "start_frame": window.start_frame,
            "video_id": self.video_id,
        }
