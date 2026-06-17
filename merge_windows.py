"""Merge overlapping window predictions into full-sequence frame probabilities."""

from __future__ import annotations

import numpy as np


def merge_window_predictions(
    window_preds: list[np.ndarray],
    start_frames: list[int],
    total_frames: int = 750,
) -> np.ndarray:
    """
    Stitch overlapping window probability vectors by averaging.

    Args:
        window_preds:  list of (T_window,) arrays, one per window
        start_frames:  global start frame index for each window
        total_frames:  full clip length (default 750)

    Returns:
        frame_probs: (total_frames,) averaged probabilities
    """
    accum = np.zeros(total_frames, dtype=np.float64)
    counts = np.zeros(total_frames, dtype=np.float64)

    for preds, start in zip(window_preds, start_frames):
        preds = np.asarray(preds, dtype=np.float64).flatten()
        end = start + len(preds)
        if end > total_frames:
            preds = preds[: total_frames - start]
            end = total_frames
        accum[start:end] += preds
        counts[start:end] += 1.0

    mask = counts > 0
    frame_probs = np.zeros(total_frames, dtype=np.float32)
    frame_probs[mask] = (accum[mask] / counts[mask]).astype(np.float32)
    return frame_probs


def merge_torch_batches(
    probs_list: list,
    start_frames_list: list,
    total_frames: int = 750,
) -> np.ndarray:
    """Convenience wrapper accepting torch tensors."""
    import torch

    window_preds = []
    start_frames = []
    for probs, starts in zip(probs_list, start_frames_list):
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        if isinstance(starts, torch.Tensor):
            starts = starts.detach().cpu().numpy()
        for i in range(probs.shape[0]):
            window_preds.append(probs[i].squeeze(-1))
            start_frames.append(int(starts[i]))
    return merge_window_predictions(window_preds, start_frames, total_frames)
