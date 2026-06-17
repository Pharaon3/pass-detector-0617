"""Plot frame-level pass probability vs ground-truth pass frames for one clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from infer import infer_video, load_model
from utils import (
    build_frame_labels,
    ensure_dir,
    frame_to_sec,
    get_device,
    list_clips,
    load_config,
    pass_frames_from_annotation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot pass probability curve for one clip")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--clip", type=str, required=True, help="Clip id, e.g. clip_4")
    p.add_argument(
        "--probs-json",
        type=str,
        default=None,
        help="Use existing outputs/{clip}_frame_probs.json instead of running inference",
    )
    p.add_argument("--output", type=str, default=None, help="Save path for PNG")
    p.add_argument("--show", action="store_true", help="Display plot window")
    p.add_argument("--threshold", type=float, default=0.5, help="Detection threshold line")
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    fps = data_cfg["fps"]

    clips = list_clips(
        data_cfg["data_root"],
        video_filename=data_cfg["video_filename"],
        label_filename=data_cfg["label_filename"],
        clip_prefix=data_cfg["clip_prefix"],
    )
    clip = next((c for c in clips if c.clip_id == args.clip), None)
    if clip is None:
        raise RuntimeError(f"Clip not found: {args.clip}. Available: {[c.clip_id for c in clips[:5]]}...")

    pass_frames = pass_frames_from_annotation(clip.label_path, fps, clip.num_frames)
    print(f"{clip.clip_id}: {len(pass_frames)} GT pass frames at {pass_frames}")

    if args.probs_json:
        frame_probs = load_probs_from_json(Path(args.probs_json))
    else:
        device = get_device()
        model = load_model(Path(args.checkpoint), device)
        result = infer_video(
            model,
            clip.video_path,
            cfg,
            device,
            video_id=clip.clip_id,
            num_frames=clip.num_frames,
        )
        frame_probs = np.asarray(result["frame_probs"], dtype=np.float32)

    output = Path(args.output) if args.output else None
    threshold = args.threshold if args.threshold is not None else cfg["inference"]["threshold"]

    plot_clip(
        clip_id=clip.clip_id,
        frame_probs=frame_probs,
        pass_frames=pass_frames,
        fps=fps,
        pass_radius_sec=cfg["labeling"]["pass_radius_sec"],
        threshold=threshold,
        output_path=output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
