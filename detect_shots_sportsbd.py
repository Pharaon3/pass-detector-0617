"""Run SportSBD shot boundary detection on soccer clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from detect_ball_players import resolve_video_path
from utils import clip_id_from_video_path, ensure_dir, get_video_frame_count, list_clips, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SportSBD shot boundary detection")
    p.add_argument("--video", type=str, default=None, help="Video path or clip folder")
    p.add_argument("--data-root", type=str, default="data", help="Bulk mode: scan this folder")
    p.add_argument("--all", action="store_true", help="Run on every clip under --data-root")
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sportsbd",
        help="JSON/plot output directory",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sportsbd/best.pt",
        help="Model weights (auto-downloaded if missing)",
    )
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--t-frames", type=int, default=16)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--device", type=str, default=None, help="cuda, cpu, or auto")
    p.add_argument("--plot", action="store_true", help="Save timeline PNG per clip")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--clip-prefix",
        type=str,
        default=None,
        help="Folder prefix filter (default: all folders with video)",
    )
    return p.parse_args()


def ensure_sportsbd():
    try:
        from sportsbd import download_model, run_video_inference
    except ImportError as exc:
        raise ImportError(
            "SportSBD not installed. Use a separate venv if needed:\n"
            "  pip install sportsbd\n"
            "Note: sportsbd pins numpy==1.26.4 (may conflict with opencv>=4.13)."
        ) from exc
    return download_model, run_video_inference


def resolve_checkpoint(path: str):
    from sportsbd import download_model

    ckpt = Path(path)
    if ckpt.is_file():
        return ckpt
    print(f"Downloading SportSBD model to {ckpt} ...")
    return download_model(destination=ckpt)


def run_on_video(
    video_path: Path,
    clip_id: str,
    output_dir: Path,
    checkpoint: Path,
    threshold: float,
    stride: int,
    t_frames: int,
    fps: int,
    device: str | None,
    make_plot: bool,
) -> dict:
    _, run_video_inference = ensure_sportsbd()

    detections = run_video_inference(
        video_path=video_path,
        checkpoint_path=checkpoint,
        threshold=threshold,
        stride=stride,
        t_frames=t_frames,
        fps=fps,
        device=device,
        progress=True,
    )

    num_frames = get_video_frame_count(video_path)
    duration_sec = num_frames / fps

    summary = {
        "clip_id": clip_id,
        "video": str(video_path.resolve()),
        "num_frames": num_frames,
        "duration_sec": round(duration_sec, 3),
        "fps": fps,
        "threshold": threshold,
        "num_detections": len(detections),
        "detections": detections,
    }

    json_path = output_dir / f"{clip_id}_shots.json"
    ensure_dir(output_dir)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {len(detections)} shot boundaries -> {json_path}")

    if make_plot:
        plot_path = output_dir / "plots" / f"{clip_id}_shots.png"
        plot_shot_timeline(summary, plot_path)

    return summary


def plot_shot_timeline(summary: dict, output_path: Path) -> None:
    detections = summary["detections"]
    duration_sec = summary["duration_sec"]
    clip_id = summary["clip_id"]

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.set_xlim(0, duration_sec)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (seconds)")
    ax.set_yticks([])
    ax.set_title(f"{clip_id} — SportSBD shot boundaries ({len(detections)} detections)")

    colors = {"hard": "#ef4444", "fadein": "#f59e0b", "logo": "#8b5cf6"}
    for det in detections:
        t = det["timestamp_ms"] / 1000.0
        cls = det.get("predicted_class") or "hard"
        color = colors.get(cls, "#64748b")
        ax.axvline(t, color=color, linewidth=1.5, alpha=0.85)
        ax.scatter([t], [0.5], color=color, s=40, zorder=5)

    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], color=colors[k], linewidth=2, label=k) for k in colors
    ]
    ax.legend(handles=legend_items, loc="upper right")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote plot: {output_path}")


def main() -> None:
    args = parse_args()
    ensure_sportsbd()
    checkpoint = resolve_checkpoint(args.checkpoint)
    output_dir = ensure_dir(args.output_dir)

    if args.all or not args.video:
        clip_prefix = args.clip_prefix if args.clip_prefix is not None else ""
        clips = list_clips(
            args.data_root,
            clip_prefix=clip_prefix if clip_prefix else None,
        )
        if not clips:
            raise RuntimeError(f"No clips found under {args.data_root}")

        for clip in tqdm(clips, desc="SportSBD"):
            out_json = output_dir / f"{clip.clip_id}_shots.json"
            if args.skip_existing and out_json.exists():
                continue
            run_on_video(
                clip.video_path,
                clip.clip_id,
                output_dir,
                checkpoint,
                args.threshold,
                args.stride,
                args.t_frames,
                args.fps,
                args.device,
                args.plot,
            )
        return

    video_path = resolve_video_path(args.video)
    clip_id = clip_id_from_video_path(video_path)
    run_on_video(
        video_path,
        clip_id,
        output_dir,
        checkpoint,
        args.threshold,
        args.stride,
        args.t_frames,
        args.fps,
        args.device,
        args.plot,
    )


if __name__ == "__main__":
    main()
