"""
Inference: window-local track features -> frame probs -> pass event JSON.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.track_dataset import TrackPassVideoDataset, track_collate_fn
from detect_ball_players import load_yolo, resolve_model_path, resolve_video_path
from merge_windows import merge_window_predictions
from models.track_model import TrackPassModel
from nms import events_to_json, extract_events_from_probs, temporal_nms
from tracks.extract import extract_and_cache_all_windows
from utils import (
    ClipRecord,
    clip_id_from_video_path,
    ensure_dir,
    get_video_frame_count,
    list_clips,
    load_config,
    save_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track-based pass event inference")
    p.add_argument("--config", type=str, default="config_tracks.yaml")
    p.add_argument("--checkpoint", type=str, default="checkpoints_tracks/best.pt")
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--device", type=str, default=None, help="Device for YOLO window track extraction")
    return p.parse_args()


def autocast_context(enabled: bool, device: torch.device):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    from torch.cuda.amp import autocast

    return autocast()


def load_track_model(checkpoint_path: Path, device: torch.device) -> TrackPassModel:
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config") or load_config("config_tracks.yaml")
    model_cfg = cfg["model"]
    track_cfg = cfg["tracks"]

    model = TrackPassModel(
        max_players=track_cfg["max_players"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def ensure_window_caches(
    clip: ClipRecord,
    window_cache_dir: Path,
    cfg: dict,
    yolo_device: str | None,
) -> None:
    track_cfg = cfg["tracks"]
    sw_cfg = cfg["sliding_window"]
    data_cfg = cfg["data"]
    yolo_cfg = cfg.get("yolo", {})

    model = load_yolo(resolve_model_path(yolo_cfg.get("model")), yolo_device)
    extract_and_cache_all_windows(
        clip,
        window_cache_dir,
        model,
        window_frames=sw_cfg["window_frames"],
        stride_frames=sw_cfg["stride_frames"],
        num_frames=data_cfg.get("num_frames"),
        force=False,
        conf=track_cfg.get("yolo_conf", 0.3),
        imgsz=track_cfg.get("yolo_imgsz", 640),
        device=yolo_device,
        median_kernel=track_cfg.get("median_kernel", 5),
        max_speed=track_cfg.get("max_speed", 60.0),
        ball_max_speed=track_cfg.get("ball_max_speed", 80.0),
    )


@torch.no_grad()
def infer_clip_tracks(
    model: TrackPassModel,
    clip: ClipRecord,
    cfg: dict,
    device: torch.device,
    num_frames: int | None = None,
) -> dict:
    data_cfg = cfg["data"]
    sw_cfg = cfg["sliding_window"]
    inf_cfg = cfg["inference"]
    track_cfg = cfg["tracks"]

    ds = TrackPassVideoDataset(
        video_path=clip.video_path,
        window_cache_dir=track_cfg["window_cache_dir"],
        clip_id=clip.clip_id,
        window_frames=sw_cfg["window_frames"],
        stride_frames=sw_cfg["stride_frames"],
        max_players=track_cfg["max_players"],
        num_frames=num_frames,
    )

    loader = DataLoader(
        ds,
        batch_size=inf_cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=0,
        collate_fn=track_collate_fn,
    )

    use_amp = cfg["training"].get("amp", True) and device.type == "cuda"
    window_preds: list[np.ndarray] = []
    start_frames: list[int] = []

    for batch in tqdm(loader, desc=f"Infer {ds.clip_id}", leave=False):
        tracks = batch["tracks"].to(device)
        with autocast_context(use_amp, device):
            out = model(tracks)
        probs = out["frame_probs"].squeeze(-1).cpu().numpy()
        starts = batch["start_frame"]
        if isinstance(starts, torch.Tensor):
            starts = starts.cpu().numpy()
        for i in range(probs.shape[0]):
            window_preds.append(probs[i])
            start_frames.append(int(starts[i]))

    frame_probs = merge_window_predictions(window_preds, start_frames, total_frames=ds.num_frames)
    events = extract_events_from_probs(
        frame_probs,
        threshold=inf_cfg["threshold"],
        min_segment_frames=inf_cfg["min_segment_frames"],
        fps=data_cfg["fps"],
    )
    events = temporal_nms(events, window_sec=inf_cfg["nms_window_sec"], fps=data_cfg["fps"])

    return {
        "video_id": ds.clip_id,
        "frame_probs": frame_probs.tolist(),
        "events": events_to_json(events),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(args.output_dir or cfg["inference"]["output_dir"])
    window_cache_dir = ensure_dir(cfg["tracks"]["window_cache_dir"])

    model = load_track_model(Path(args.checkpoint), device)
    print(f"Track model inference on {device} (7s window-local tracks)")

    if args.video:
        video_path = resolve_video_path(args.video)
        clip_id = clip_id_from_video_path(video_path, cfg["data"]["label_filename"])
        clips = [
            ClipRecord(
                clip_id=clip_id,
                clip_dir=video_path.parent,
                video_path=video_path,
                label_path=video_path.parent / cfg["data"]["label_filename"],
                num_frames=get_video_frame_count(video_path),
            )
        ]
    else:
        data_cfg = cfg["data"]
        clips = list_clips(
            data_cfg["data_root"],
            video_filename=data_cfg["video_filename"],
            label_filename=data_cfg["label_filename"],
            clip_prefix=data_cfg["clip_prefix"],
        )

    if not clips:
        raise RuntimeError("No clips found.")

    for clip in tqdm(clips, desc="Infer clips"):
        out_probs = output_dir / f"{clip.clip_id}_frame_probs.json"
        if args.skip_existing and out_probs.exists():
            continue

        ensure_window_caches(clip, window_cache_dir, cfg, args.device)
        result = infer_clip_tracks(
            model,
            clip,
            cfg,
            device,
            num_frames=clip.num_frames,
        )

        save_json({"frame_probs": result["frame_probs"]}, out_probs)
        save_json(result["events"], output_dir / f"{result['video_id']}_events.json")
        print(f"{result['video_id']}: {len(result['events'])} events")

    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
