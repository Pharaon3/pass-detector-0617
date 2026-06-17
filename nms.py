"""Temporal NMS and event extraction from frame probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils import frame_to_sec


@dataclass
class PassEvent:
    frame: int
    time_sec: float
    probability: float
    event: str = "Pass"


def find_contiguous_segments(mask: np.ndarray, min_length: int = 3) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) for True runs in binary mask."""
    segments = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_length:
                segments.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_length:
        segments.append((start, len(mask)))
    return segments


def extract_events_from_probs(
    frame_probs: np.ndarray,
    threshold: float = 0.5,
    min_segment_frames: int = 3,
    fps: int = 25,
) -> list[PassEvent]:
    """
    Detect pass events from frame probabilities:
      1. Threshold
      2. Find contiguous segments
      3. Peak frame = argmax within segment
    """
    probs = np.asarray(frame_probs, dtype=np.float32)
    mask = probs >= threshold
    segments = find_contiguous_segments(mask, min_length=min_segment_frames)

    events = []
    for start, end in segments:
        seg_probs = probs[start:end]
        peak_local = int(np.argmax(seg_probs))
        peak_frame = start + peak_local
        events.append(
            PassEvent(
                frame=peak_frame,
                time_sec=frame_to_sec(peak_frame, fps),
                probability=float(probs[peak_frame]),
            )
        )
    return events


def temporal_nms(
    events: list[PassEvent],
    window_sec: float = 1.0,
    fps: int = 25,
) -> list[PassEvent]:
    """
    Remove duplicate events within ±window_sec, keeping highest probability.
    """
    if not events:
        return []

    window_frames = int(window_sec * fps)
    sorted_events = sorted(events, key=lambda e: e.probability, reverse=True)
    kept: list[PassEvent] = []

    for ev in sorted_events:
        if all(abs(ev.frame - k.frame) > window_frames for k in kept):
            kept.append(ev)

    return sorted(kept, key=lambda e: e.frame)


def events_to_json(events: list[PassEvent]) -> list[dict]:
    return [
        {"time_sec": round(e.time_sec, 3), "event": e.event, "frame": e.frame, "probability": round(e.probability, 4)}
        for e in events
    ]
