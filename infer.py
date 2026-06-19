"""Inference: frame-level pass probabilities + event JSON."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import SoccerPassVideoDataset, collate_fn
from merge_windows import merge_window_predictions
from models.model import PassDetectionModel
from nms import events_to_json, extract_events_from_probs, temporal_nms
from utils import ClipRecord, ensure_dir, get_device, get_video_frame_count, list_clips, load_config, save_json, clip_id_from_video_path


def autocast_context(enabled: bool, device: torch.device):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    from torch.cuda.amp import autocast

    return autocast()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pass detection inference")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--video", type=str, default=None, help="Single video path")
    p.add_argument("--video-dir", type=str, default=None, help="Directory of test videos")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip clips whose outputs/{clip_id}_frame_probs.json already exists",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override config inference output_dir (e.g. outputs_grayscale)",
    )
    return p.parse_args()


@torch.no_grad()
def infer_video(
    model: PassDetectionModel,
    video_path: Path,
    cfg: dict,
    device: torch.device,
    video_id: str | None = None,
    num_frames: int | None = None,
) -> dict:
    data_cfg = cfg["data"]
    sw_cfg = cfg["sliding_window"]
    prep_cfg = cfg["preprocessing"]
    inf_cfg = cfg["inference"]

    if num_frames is None:
        num_frames = get_video_frame_count(video_path)
        if data_cfg.get("num_frames") is not None:
            num_frames = min(num_frames, data_cfg["num_frames"])

    if video_id is None:
        video_id = clip_id_from_video_path(video_path, data_cfg.get("label_filename", "label.json"))

    ds = SoccerPassVideoDataset(
        video_path=video_path,
        video_id=video_id,
        window_frames=sw_cfg["window_frames"],
        stride_frames=sw_cfg["stride_frames"],
        num_frames=num_frames,
        frame_size=prep_cfg["frame_size"],
        mean=prep_cfg["mean"],
        std=prep_cfg["std"],
    )

    use_amp = cfg["training"]["amp"] and device.type == "cuda"
    batch_size = inf_cfg["batch_size"]
    if device.type != "cuda":
        batch_size = inf_cfg.get("cpu_batch_size", 1)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    window_preds: list[np.ndarray] = []
    start_frames: list[int] = []

    model.eval()
    for batch in tqdm(loader, desc=f"Infer {video_path.name}", leave=False):
        video = batch["video"].to(device)
        with autocast_context(use_amp, device):
            out = model(video)
        probs = out["frame_probs"].squeeze(-1).cpu().numpy()

        starts = batch["start_frame"]
        if isinstance(starts, torch.Tensor):
            starts = starts.cpu().numpy()

        for i in range(probs.shape[0]):
            window_preds.append(probs[i])
            start_frames.append(int(starts[i]))

    frame_probs = merge_window_predictions(
        window_preds,
        start_frames,
        total_frames=num_frames,
    )

    events = extract_events_from_probs(
        frame_probs,
        threshold=inf_cfg["threshold"],
        min_segment_frames=inf_cfg["min_segment_frames"],
        fps=data_cfg["fps"],
    )
    events = temporal_nms(events, window_sec=inf_cfg["nms_window_sec"], fps=data_cfg["fps"])

    return {
        "video_id": video_id,
        "frame_probs": frame_probs.tolist(),
        "events": events_to_json(events),
    }


def load_model(checkpoint_path: Path, device: torch.device) -> PassDetectionModel:
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config") or load_config("config.yaml")
    model_cfg = cfg["model"]

    model = PassDetectionModel(
        backbone_name=model_cfg["backbone"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg["dropout"],
        pretrained=False,
        backbone_chunk_size=model_cfg.get("backbone_chunk_size", 16),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()
    output_dir = ensure_dir(args.output_dir or cfg["inference"]["output_dir"])

    model = load_model(Path(args.checkpoint), device)
    print(f"Inference on {device}")

    if args.video:
        video_path = Path(args.video)
        cid = clip_id_from_video_path(video_path, cfg["data"]["label_filename"])
        clips = [
            ClipRecord(
                clip_id=cid,
                clip_dir=video_path.parent,
                video_path=video_path,
                label_path=video_path.parent / cfg["data"]["label_filename"],
                num_frames=get_video_frame_count(video_path),
            )
        ]
    elif args.video_dir:
        clips = list_clips(
            args.video_dir,
            video_filename=cfg["data"]["video_filename"],
            label_filename=cfg["data"]["label_filename"],
            clip_prefix=cfg["data"]["clip_prefix"],
        )
    else:
        data_cfg = cfg["data"]
        clips = list_clips(
            data_cfg["data_root"],
            video_filename=data_cfg["video_filename"],
            label_filename=data_cfg["label_filename"],
            clip_prefix=data_cfg["clip_prefix"],
        )

    if not clips:
        raise RuntimeError("No videos found for inference.")

    skipped = 0
    for clip in tqdm(clips, desc="Infer clips"):
        out_probs = output_dir / f"{clip.clip_id}_frame_probs.json"
        if args.skip_existing and out_probs.exists():
            skipped += 1
            continue

        result = infer_video(
            model,
            clip.video_path,
            cfg,
            device,
            video_id=clip.clip_id,
            num_frames=clip.num_frames,
        )

        save_json({"frame_probs": result["frame_probs"]}, out_probs)

        out_events = output_dir / f"{result['video_id']}_events.json"
        save_json(result["events"], out_events)

        print(f"{result['video_id']}: saved {len(result['frame_probs'])} frame probs, "
              f"{len(result['events'])} events")

    if skipped:
        print(f"Skipped {skipped} clips (already inferred). Use without --skip-existing to re-run.")
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
