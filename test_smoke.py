"""Quick unit tests for window merge and sliding window logic."""

import numpy as np

from dataset.sliding_window import generate_sliding_windows
from merge_windows import merge_window_predictions
from nms import extract_events_from_probs, temporal_nms
from utils import build_frame_labels, ms_to_frame


def test_sliding_windows():
    windows = generate_sliding_windows(750, window_frames=175, stride_frames=25)
    assert len(windows) > 0
    assert windows[0].start_frame == 0
    assert windows[0].length == 175
    assert windows[-1].end_frame == 750


def test_merge_windows():
    preds = [np.ones(175) * 0.8, np.ones(175) * 0.2]
    starts = [0, 25]
    merged = merge_window_predictions(preds, starts, total_frames=750)
    assert len(merged) == 750
    # Overlap region [25:175] should be average of 0.8 and 0.2 = 0.5
    assert abs(merged[50] - 0.5) < 1e-5


def test_labels_and_nms():
    center = ms_to_frame(22280)
    labels = build_frame_labels([center], num_frames=750, radius_sec=0.5, fps=25)
    assert labels[center] > 0.9

    probs = np.zeros(750, dtype=np.float32)
    probs[center - 2 : center + 3] = 0.85
    probs[center] = 0.9
    events = extract_events_from_probs(probs, threshold=0.5)
    events = temporal_nms(events, window_sec=1.0)
    assert len(events) == 1


if __name__ == "__main__":
    test_sliding_windows()
    test_merge_windows()
    test_labels_and_nms()
    print("All tests passed.")
