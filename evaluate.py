"""Evaluate frame-level and event-level metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import SoccerPassDataset, collate_fn
from infer import infer_video, load_model
from nms import extract_events_from_probs, temporal_nms
from utils import (
    build_frame_labels,
    get_device,
    list_clips,
    load_config,
    pass_frames_from_annotation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pass detection")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    return p.parse_args()


def temporal_errors(pred_frames: list[int], gt_frames: list[int], tolerance: int = 25) -> dict:
    """Match predictions to GT within tolerance frames."""
    if not gt_frames:
        return {"matched": 0, "errors": [], "precision": 0.0, "recall": 0.0}

    gt_used = set()
    errors = []
    matched = 0

    for pf in sorted(pred_frames):
        best_gt = None
        best_dist = tolerance + 1
        for gi, gf in enumerate(gt_frames):
            if gi in gt_used:
                continue
            d = abs(pf - gf)
            if d <= tolerance and d < best_dist:
                best_dist = d
                best_gt = gi
        if best_gt is not None:
            gt_used.add(best_gt)
            matched += 1
            errors.append(best_dist)

    precision = matched / max(len(pred_frames), 1)
    recall = matched / len(gt_frames)
    mean_error = float(np.mean(errors)) if errors else float("inf")
    return {
        "matched": matched,
        "errors_sec": [e / 25.0 for e in errors],
        "mean_temporal_error_sec": mean_error / 25.0 if errors else float("inf"),
        "precision": precision,
        "recall": recall,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()
    data_cfg = cfg["data"]
    inf_cfg = cfg["inference"]
    metrics_cfg = cfg["metrics"]

    model = load_model(Path(args.checkpoint), device)
    clips = list_clips(
        data_cfg["data_root"],
        video_filename=data_cfg["video_filename"],
        label_filename=data_cfg["label_filename"],
        clip_prefix=data_cfg["clip_prefix"],
    )

    all_probs = []
    all_labels = []
    event_results = []

    for clip in tqdm(clips, desc="Evaluate"):
        num_frames = clip.num_frames
        if data_cfg.get("num_frames") is not None:
            num_frames = min(num_frames, data_cfg["num_frames"])

        result = infer_video(
            model,
            clip.video_path,
            cfg,
            device,
            video_id=clip.clip_id,
            num_frames=num_frames,
        )
        frame_probs = np.array(result["frame_probs"], dtype=np.float32)

        gt_frames = pass_frames_from_annotation(clip.label_path, data_cfg["fps"], num_frames)
        gt_labels = build_frame_labels(
            gt_frames,
            num_frames,
            radius_sec=cfg["labeling"]["pass_radius_sec"],
            fps=data_cfg["fps"],
        )

        all_probs.append(frame_probs)
        all_labels.append(gt_labels)

        events = extract_events_from_probs(
            frame_probs,
            threshold=inf_cfg["threshold"],
            min_segment_frames=inf_cfg["min_segment_frames"],
            fps=data_cfg["fps"],
        )
        events = temporal_nms(events, window_sec=inf_cfg["nms_window_sec"], fps=data_cfg["fps"])
        pred_frames = [e.frame for e in events]

        er = temporal_errors(pred_frames, gt_frames, tolerance=int(data_cfg["fps"]))
        event_results.append(er)

    if not all_probs:
        print("No annotated videos found.")
        return

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = (y_prob >= metrics_cfg["frame_threshold"]).astype(int)
    y_bin = (y_true >= 0.5).astype(int)

    auc = roc_auc_score(y_bin, y_prob) if len(np.unique(y_bin)) > 1 else 0.0
    f1 = f1_score(y_bin, y_pred, zero_division=0)
    prec = precision_score(y_bin, y_pred, zero_division=0)
    rec = recall_score(y_bin, y_pred, zero_division=0)

    mean_event_prec = np.mean([e["precision"] for e in event_results])
    mean_event_rec = np.mean([e["recall"] for e in event_results])
    valid_errors = [e["mean_temporal_error_sec"] for e in event_results if e["errors_sec"]]
    mean_temp_err = float(np.mean(valid_errors)) if valid_errors else float("inf")

    print("\n=== Frame-level metrics ===")
    print(f"AUC:       {auc:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")

    print("\n=== Event-level metrics ===")
    print(f"Event Precision: {mean_event_prec:.4f}")
    print(f"Event Recall:    {mean_event_rec:.4f}")
    print(f"Mean temporal error (sec): {mean_temp_err:.4f}")


if __name__ == "__main__":
    main()
