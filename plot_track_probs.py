"""Plot frame-level pass probabilities from the track-based model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from infer_tracks import ensure_track_cache, infer_clip_tracks, load_track_model
from plot_probs import load_probs_from_json, plot_one_clip, probs_json_path
from utils import ClipRecord, ensure_dir, get_clip, list_clips, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot track-model pass probability curves")
    p.add_argument("--config", type=str, default="config_tracks.yaml")
    p.add_argument("--checkpoint", type=str, default="checkpoints_tracks/best.pt")
    p.add_argument("--clip", type=str, default=None, help="Single clip id, e.g. clip_4")
    p.add_argument("--all", action="store_true", help="Plot every clip under data/")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--probs-json", type=str, default=None)
    p.add_argument(
        "--probs-dir",
        type=str,
        default="outputs_tracks",
        help="Directory with {clip_id}_frame_probs.json",
    )
    p.add_argument(
        "--infer-missing",
        action="store_true",
        help="With --all: run track inference for clips missing probs JSON",
    )
    p.add_argument("--output-dir", type=str, default="outputs_tracks/plots")
    p.add_argument("--output", type=str, default=None, help="Save path for single --clip PNG")
    p.add_argument("--show", action="store_true")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--yolo-device",
        type=str,
        default=None,
        help="Device for YOLO track extraction when cache/probs missing",
    )
    return p.parse_args()


def resolve_track_probs(
    clip: ClipRecord,
    cfg: dict,
    cache_dir: Path,
    probs_dir: Path,
    probs_json: Path | None,
    model,
    device: torch.device,
    infer_missing: bool,
    yolo_device: str | None,
) -> np.ndarray | None:
    if probs_json is not None:
        return load_probs_from_json(probs_json)

    cached = probs_json_path(probs_dir, clip.clip_id)
    if cached.exists():
        return load_probs_from_json(cached)

    if not infer_missing or model is None:
        return None

    cache_path = ensure_track_cache(clip, cache_dir, cfg, yolo_device)
    result = infer_clip_tracks(model, cache_path, cfg, device, num_frames=clip.num_frames)
    return np.asarray(result["frame_probs"], dtype=np.float32)


def main() -> None:
    args = parse_args()
    if not args.all and not args.clip:
        raise SystemExit("Specify --clip clip_4 or --all")

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    data_root = Path(args.data_root or data_cfg["data_root"])
    threshold = args.threshold if args.threshold is not None else cfg["inference"]["threshold"]
    probs_dir = Path(args.probs_dir)
    output_dir = ensure_dir(args.output_dir)
    cache_dir = Path(cfg["tracks"]["cache_dir"])

    clips = list_clips(
        data_root,
        video_filename=data_cfg["video_filename"],
        label_filename=data_cfg["label_filename"],
        clip_prefix=data_cfg["clip_prefix"],
    )
    if not clips:
        raise RuntimeError("No clips found under data/")

    if args.all:
        need_infer = args.infer_missing or any(
            not probs_json_path(probs_dir, c.clip_id).exists() for c in clips
        )
        model = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if need_infer:
            model = load_track_model(Path(args.checkpoint), device)
            print(f"Track model loaded on {device}")

        skipped = []
        plotted = 0
        for clip in tqdm(clips, desc="Plot clips"):
            out_png = output_dir / f"{clip.clip_id}_pass_probs.png"
            probs_path = probs_json_path(probs_dir, clip.clip_id)
            if args.skip_existing and out_png.exists() and probs_path.exists():
                if out_png.stat().st_mtime >= probs_path.stat().st_mtime:
                    skipped.append(clip.clip_id)
                    continue

            frame_probs = resolve_track_probs(
                clip,
                cfg,
                cache_dir,
                probs_dir,
                None,
                model,
                device,
                infer_missing=args.infer_missing,
                yolo_device=args.yolo_device,
            )
            if frame_probs is None:
                skipped.append(clip.clip_id)
                continue

            if args.infer_missing and model is not None and not probs_path.exists():
                from utils import save_json

                save_json({"frame_probs": frame_probs.tolist()}, probs_path)

            plot_one_clip(clip, cfg, frame_probs, output_dir, threshold)
            plotted += 1

        print(f"Saved {plotted} plots to {output_dir}")
        if skipped:
            print(f"Skipped {len(skipped)} clips (missing probs; use --infer-missing)")
        return

    clip = get_clip(
        data_root,
        args.clip,
        video_filename=data_cfg["video_filename"],
        label_filename=data_cfg["label_filename"],
    )
    if clip is None:
        raise RuntimeError(f"Clip not found: {args.clip} under {data_root}")

    probs_json = Path(args.probs_json) if args.probs_json else None
    if probs_json is None and probs_json_path(probs_dir, clip.clip_id).exists():
        probs_json = probs_json_path(probs_dir, clip.clip_id)

    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if probs_json is None or not probs_json.exists():
        model = load_track_model(Path(args.checkpoint), device)

    frame_probs = resolve_track_probs(
        clip,
        cfg,
        cache_dir,
        probs_dir,
        probs_json,
        model,
        device,
        infer_missing=True,
        yolo_device=args.yolo_device,
    )
    if frame_probs is None:
        raise RuntimeError(f"Could not load probabilities for {clip.clip_id}")

    output = Path(args.output) if args.output else None
    plot_one_clip(
        clip,
        cfg,
        frame_probs,
        output_dir,
        threshold,
        output_path=output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
