"""Smooth player tracks and remove outlier peaks (detection jumps)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackPoint:
    frame: int
    x: float
    y: float


def _interp_series(frames: np.ndarray, values: np.ndarray, target_frames: np.ndarray) -> np.ndarray:
    if len(frames) == 0:
        return np.zeros(len(target_frames), dtype=np.float32)
    if len(frames) == 1:
        return np.full(len(target_frames), values[0], dtype=np.float32)
    return np.interp(target_frames, frames, values).astype(np.float32)


def moving_median(values: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1 or len(values) == 0:
        return values.copy()
    kernel = kernel if kernel % 2 == 1 else kernel + 1
    half = kernel // 2
    out = values.copy().astype(np.float32)
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out[i] = float(np.median(values[lo:hi]))
    return out


def remove_velocity_peaks(
    frames: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    max_speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Drop points where speed to the previous kept point exceeds max_speed (px/frame).
    Remaining gaps are filled later by interpolation.
    """
    if len(frames) <= 1:
        return frames.copy(), xs.copy(), ys.copy()

    keep = [0]
    for i in range(1, len(frames)):
        dt = max(frames[i] - frames[keep[-1]], 1)
        dx = xs[i] - xs[keep[-1]]
        dy = ys[i] - ys[keep[-1]]
        speed = float(np.hypot(dx, dy) / dt)
        if speed <= max_speed:
            keep.append(i)

    return frames[keep], xs[keep], ys[keep]


def smooth_track_points(
    points: list[TrackPoint],
    median_kernel: int = 5,
    max_speed: float = 60.0,
    fill_gaps: bool = True,
) -> list[TrackPoint]:
    """
    Denoise a single track:
      1. Remove velocity peaks (bad detections)
      2. Median smooth surviving points
      3. Optionally interpolate to contiguous frame range
    """
    if not points:
        return []

    order = np.argsort([p.frame for p in points])
    frames = np.array([points[i].frame for i in order], dtype=np.int32)
    xs = np.array([points[i].x for i in order], dtype=np.float32)
    ys = np.array([points[i].y for i in order], dtype=np.float32)

    frames, xs, ys = remove_velocity_peaks(frames, xs, ys, max_speed=max_speed)
    if len(frames) == 0:
        return []

    xs = moving_median(xs, median_kernel)
    ys = moving_median(ys, median_kernel)

    if fill_gaps and len(frames) >= 2:
        target = np.arange(int(frames[0]), int(frames[-1]) + 1, dtype=np.int32)
        xs = _interp_series(frames, xs, target)
        ys = _interp_series(frames, ys, target)
        frames = target

    return [TrackPoint(frame=int(f), x=float(x), y=float(y)) for f, x, y in zip(frames, xs, ys)]


def smooth_ball_series(
    ball_by_frame: dict[int, tuple[float, float]],
    num_frames: int,
    median_kernel: int = 3,
    max_speed: float = 80.0,
) -> dict[int, tuple[float, float]]:
    """Smooth sparse per-frame ball detections."""
    if not ball_by_frame:
        return {}

    frames = np.array(sorted(ball_by_frame.keys()), dtype=np.int32)
    xs = np.array([ball_by_frame[int(f)][0] for f in frames], dtype=np.float32)
    ys = np.array([ball_by_frame[int(f)][1] for f in frames], dtype=np.float32)

    frames, xs, ys = remove_velocity_peaks(frames, xs, ys, max_speed=max_speed)
    if len(frames) == 0:
        return {}

    xs = moving_median(xs, median_kernel)
    ys = moving_median(ys, median_kernel)

    out: dict[int, tuple[float, float]] = {}
    for f, x, y in zip(frames, xs, ys):
        if 0 <= int(f) < num_frames:
            out[int(f)] = (float(x), float(y))
    return out
