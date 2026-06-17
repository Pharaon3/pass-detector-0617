"""Plot frame-level pass probability vs ground-truth pass frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from infer import infer_video, load_model
from utils import (
    ClipRecord,
    build_frame_labels,
    ensure_dir,
    frame_to_sec,
    get_device,
    list_clips,
    load_config,
    pass_frames_from_annotation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot pass probability curve for one or all clips")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--clip", type=str, default=None, help="Single clip id, e.g. clip_4")
    p.add_argument("--all", action="store_true", help="Plot every clip under data/")
    p.add_argument(
        "--probs-json",
        type=str,
        default=None,
        help="Use this JSON for a single --clip run",
    )
    p.add_argument(
        "--probs-dir",
        type=str,
        default="outputs",
        help="Directory with {clip_id}_frame_probs.json files (used with --all)",
    )
    p.add_argument(
        "--infer-missing",
        action="store_true",
        help="With --all: run inference for clips missing probs JSON",
    )
    p.add_argument("--output-dir", type=str, default="outputs/plots", help="Where to save PNGs")
    p.add_argument("--output", type=str, default=None, help="Save path for single --clip PNG")
    p.add_argument("--show", action="store_true", help="Display plot window (single clip only)")
    p.add_argument("--threshold", type=float, default=None, help="Detection threshold line")
    return p.parse_args()


def load_probs_from_json(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.asarray(data["frame_probs"], dtype=np.float32)


def plot_clip(
    clip_id: str,
    frame_probs: np.ndarray,
    pass_frames: list[int],
    fps: int = 25,
    pass_radius_sec: float = 0.5,
    threshold: float = 0.5,
    output_path: Path | None = None,
    show: bool = False,
) -> Path:
    num_frames = len(frame_probs)
    frames = np.arange(num_frames)
    times = frames / fps
    soft_labels = build_frame_labels(pass_frames, num_frames, radius_sec=pass_radius_sec, fps=fps)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(times, frame_probs, color="#2563eb", linewidth=1.5, label="Predicted pass probability")
    ax.fill_between(times, 0, soft_labels, color="#22c55e", alpha=0.18, label="GT label region (±0.5s)")

    for pf in pass_frames:
        t = frame_to_sec(pf, fps)
        ax.axvline(t, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.85)
        ax.scatter([t], [frame_probs[pf]], color="#ef4444", s=60, zorder=5, edgecolors="white", linewidths=0.8)

    ax.axhline(threshold, color="#94a3b8", linestyle=":", linewidth=1.2, label=f"Threshold ({threshold})")

    # Legend for GT vertical lines (only once)
    ax.plot([], [], color="#ef4444", linestyle="--", linewidth=1.2, label="GT pass frame")

    ax.set_xlim(0, times[-1])
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Pass probability")
    ax.set_title(f"{clip_id} — frame-level pass probability")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    for pf in pass_frames:
        t = frame_to_sec(pf, fps)
        ax.annotate(
            f"f{pf}",
            (t, frame_probs[pf]),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=8,
            color="#991b1b",
        )

    fig.tight_layout()

    if output_path is None:
        output_path = Path("outputs") / "plots" / f"{clip_id}_pass_probs.png"
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def probs_json_path(probs_dir: Path, clip_id: str) -> Path:
    return probs_dir / f"{clip_id}_frame_probs.json"


def resolve_frame_probs(
    clip: ClipRecord,
    cfg: dict,
    probs_dir: Path,
    probs_json: Path | None,
    model,
    device,
    infer_missing: bool,
) -> np.ndarray | None:
    if probs_json is not None:
        return load_probs_from_json(probs_json)

    cached = probs_json_path(probs_dir, clip.clip_id)
    if cached.exists():
        return load_probs_from_json(cached)

    if not infer_missing or model is None:
        return None

    result = infer_video(
        model,
        clip.video_path,
        cfg,
        device,
        video_id=clip.clip_id,
        num_frames=clip.num_frames,
    )
    return np.asarray(result["frame_probs"], dtype=np.float32)


def plot_one_clip(
    clip: ClipRecord,
    cfg: dict,
    frame_probs: np.ndarray,
    output_dir: Path,
    threshold: float,
    output_path: Path | None = None,
    show: bool = False,
) -> Path:
    fps = cfg["data"]["fps"]
    pass_frames = pass_frames_from_annotation(clip.label_path, fps, clip.num_frames)
    if output_path is None:
        output_path = output_dir / f"{clip.clip_id}_pass_probs.png"
    return plot_clip(
        clip_id=clip.clip_id,
        frame_probs=frame_probs,
        pass_frames=pass_frames,
        fps=fps,
        pass_radius_sec=cfg["labeling"]["pass_radius_sec"],
        threshold=threshold,
        output_path=output_path,
        show=show,
    )


def main() -> None:
    args = parse_args()
    if not args.all and not args.clip:
        raise SystemExit("Specify --clip clip_4 or --all")

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    threshold = args.threshold if args.threshold is not None else cfg["inference"]["threshold"]
    probs_dir = Path(args.probs_dir)
    output_dir = ensure_dir(args.output_dir)

    clips = list_clips(
        data_cfg["data_root"],
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
        model = device = None
        if need_infer:
            device = get_device()
            model = load_model(Path(args.checkpoint), device)

        skipped = []
        for clip in tqdm(clips, desc="Plot clips"):
            frame_probs = resolve_frame_probs(
                clip, cfg, probs_dir, None, model, device, infer_missing=args.infer_missing
            )
            if frame_probs is None:
                skipped.append(clip.clip_id)
                continue
            plot_one_clip(clip, cfg, frame_probs, output_dir, threshold)

        print(f"Saved {len(clips) - len(skipped)} plots to {output_dir}")
        if skipped:
            print(f"Skipped {len(skipped)} clips (no probs JSON): {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
            print("Run:  python infer.py --checkpoint checkpoints/best.pt")
            print("Then: python plot_probs.py --all")
        return

    clip = next((c for c in clips if c.clip_id == args.clip), None)
    if clip is None:
        raise RuntimeError(f"Clip not found: {args.clip}")

    pass_frames = pass_frames_from_annotation(clip.label_path, data_cfg["fps"], clip.num_frames)
    print(f"{clip.clip_id}: {len(pass_frames)} GT pass frames at {pass_frames}")

    probs_json = Path(args.probs_json) if args.probs_json else None
    if probs_json is None and probs_json_path(probs_dir, clip.clip_id).exists():
        probs_json = probs_json_path(probs_dir, clip.clip_id)

    model = device = None
    if probs_json is None or not probs_json.exists():
        device = get_device()
        model = load_model(Path(args.checkpoint), device)

    frame_probs = resolve_frame_probs(
        clip, cfg, probs_dir, probs_json, model, device, infer_missing=True
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
