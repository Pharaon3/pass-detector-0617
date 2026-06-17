"""Sliding window index generation for temporal clips."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WindowSpec:
    start_frame: int
    end_frame: int  # exclusive

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


def generate_sliding_windows(
    total_frames: int,
    window_frames: int = 175,
    stride_frames: int = 25,
) -> list[WindowSpec]:
    """
    Generate overlapping window specs for a video clip.

    Example (750 frames, window=175, stride=25):
      [0:175], [25:200], [50:225], ... until end covers total_frames.
    """
    if total_frames <= 0:
        return []

    windows: list[WindowSpec] = []
    start = 0
    while start + window_frames <= total_frames:
        windows.append(WindowSpec(start_frame=start, end_frame=start + window_frames))
        start += stride_frames

    # Ensure last window reaches the end if not already covered
    if not windows or windows[-1].end_frame < total_frames:
        last_start = max(0, total_frames - window_frames)
        if not windows or windows[-1].start_frame != last_start:
            windows.append(WindowSpec(start_frame=last_start, end_frame=total_frames))

    return windows


def window_global_indices(window: WindowSpec) -> list[int]:
    """Map local window frame indices to global clip frame indices."""
    return list(range(window.start_frame, window.end_frame))
