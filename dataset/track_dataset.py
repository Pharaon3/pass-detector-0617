"""Dataset: sliding windows over precomputed track feature matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.sliding_window import WindowSpec, generate_sliding_windows
from tracks.extract import cache_path_for_clip, load_track_cache
from tracks.features import build_clip_feature_matrix
from utils import ClipRecord, build_frame_labels, list_clips, pass_frames_from_annotation


class TrackPassDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        cache_dir: str | Path,
        video_filename: str = "224p.mp4",
        label_filename: str = "label.json",
        clip_prefix: str = "clip_",
        window_frames: int = 175,
        stride_frames: int = 25,
        num_frames: int | None = 750,
        fps: int = 25,
        pass_radius_sec: float = 0.5,
        max_players: int = 22,
        clip_ids: list[str] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.default_num_frames = num_frames
        self.fps = fps
        self.pass_radius_sec = pass_radius_sec
        self.max_players = max_players

        clips = list_clips(
            data_root,
            video_filename=video_filename,
            label_filename=label_filename,
            clip_prefix=clip_prefix,
        )
        if clip_ids is not None:
            wanted = set(clip_ids)
            clips = [c for c in clips if c.clip_id in wanted]

        self.samples: list[tuple[str, WindowSpec, np.ndarray, list[int]]] = []
        self._feature_cache: dict[str, np.ndarray] = {}
        self._build_index(clips)

    def _get_clip_features(self, clip_id: str) -> np.ndarray:
        if clip_id in self._feature_cache:
            return self._feature_cache[clip_id]

        cache_path = cache_path_for_clip(self.cache_dir, clip_id)
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Track cache missing for {clip_id}: {cache_path}\n"
                "Run: python extract_tracks.py"
            )

        cache = load_track_cache(cache_path)
        clip_frames = int(cache["num_frames"])
        if self.default_num_frames is not None:
            clip_frames = min(clip_frames, self.default_num_frames)

        feats = build_clip_feature_matrix(
            cache,
            max_players=self.max_players,
            num_frames=clip_frames,
        )
        self._feature_cache[clip_id] = feats
        return feats

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
                self.samples.append((clip.clip_id, w, window_labels.copy(), pass_frames))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip_id, window, window_labels, pass_frames = self.samples[idx]
        full_feats = self._get_clip_features(clip_id)
        track_feats = full_feats[window.start_frame : window.end_frame]

        labels = torch.from_numpy(window_labels).float()
        peak_times = []
        for pf in pass_frames:
            if window.start_frame <= pf < window.end_frame:
                local = pf - window.start_frame
                peak_times.append(local / max(window.length - 1, 1))
        peak_time_tensor = (
            torch.tensor(peak_times, dtype=torch.float32)
            if peak_times
            else torch.zeros(0, dtype=torch.float32)
        )

        return {
            "tracks": torch.from_numpy(track_feats).float(),
            "labels": labels,
            "peak_times": peak_time_tensor,
            "start_frame": window.start_frame,
            "video_id": clip_id,
        }


class TrackPassVideoDataset(Dataset):
    """Inference: sliding windows from one clip's track cache."""

    def __init__(
        self,
        cache_path: str | Path,
        window_frames: int = 175,
        stride_frames: int = 25,
        max_players: int = 22,
        num_frames: int | None = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.cache = load_track_cache(self.cache_path)
        self.clip_id = self.cache["clip_id"]
        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.max_players = max_players

        clip_frames = int(self.cache["num_frames"])
        if num_frames is not None:
            clip_frames = min(clip_frames, num_frames)

        self.num_frames = clip_frames
        self.full_feats = build_clip_feature_matrix(
            self.cache,
            max_players=max_players,
            num_frames=clip_frames,
        )
        self.windows = generate_sliding_windows(
            clip_frames,
            window_frames=window_frames,
            stride_frames=stride_frames,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        window = self.windows[idx]
        track_feats = self.full_feats[window.start_frame : window.end_frame]
        return {
            "tracks": torch.from_numpy(track_feats).float(),
            "start_frame": window.start_frame,
            "video_id": self.clip_id,
        }


def track_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tracks = torch.stack([b["tracks"] for b in batch], dim=0)
    start_frames = torch.tensor([b["start_frame"] for b in batch], dtype=torch.long)
    video_ids = [b["video_id"] for b in batch]
    out: dict[str, Any] = {
        "tracks": tracks,
        "start_frame": start_frames,
        "video_id": video_ids,
    }
    if "labels" in batch[0]:
        out["labels"] = torch.stack([b["labels"] for b in batch], dim=0)
        out["peak_times"] = [b["peak_times"] for b in batch]
    return out
