"""Detect ball and players with local YOLO (Hugging Face weights, no Roboflow)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils import clip_id_from_video_path, ensure_dir, get_video_frame_count

DEFAULT_MODEL_URL = (
    "https://huggingface.co/aabyzov/easychamp-player-detection-yolov8/"
    "resolve/main/player_detection_best.pt"
)
DEFAULT_MODEL_PATH = Path("checkpoints/yolo/player_detection_best.pt")

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "player": (0, 220, 0),
    "ball": (0, 220, 255),
    "goalkeeper": (255, 140, 0),
    "referee": (180, 180, 180),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect ball/players with YOLO and write labeled video + optional JSON"
    )
    p.add_argument(
        "--video",
        type=str,
        required=True,
        help="Input video path, or clip folder containing 224p.mp4",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output labeled video path (default: outputs/detections/{clip_id}_labeled.mp4)",
    )
    p.add_argument(
        "--json",
        type=str,
        default=None,
        help="Output JSON path for per-frame boxes (default: same stem as --output with .json)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Local .pt weights (default: download/cache to {DEFAULT_MODEL_PATH})",
    )
    p.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    p.add_argument(
        "--classes",
        type=str,
        default="player,ball",
        help="Comma-separated class names to keep (default: player,ball)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Ultralytics device, e.g. 0 or cpu (default: auto)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most this many frames (default: full video)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference size (longer side)",
    )
    return p.parse_args()


def resolve_video_path(video_arg: str) -> Path:
    path = Path(video_arg)
    if path.is_dir():
        for name in ("224p.mp4", "video.mp4"):
            candidate = path / name
            if candidate.is_file():
                return candidate
        mp4s = sorted(path.glob("*.mp4"))
        if len(mp4s) == 1:
            return mp4s[0]
        raise FileNotFoundError(f"No video found in folder: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def resolve_model_path(model_arg: str | None) -> Path:
    if model_arg:
        path = Path(model_arg)
        if not path.is_file():
            raise FileNotFoundError(f"Model weights not found: {path}")
        return path

    if DEFAULT_MODEL_PATH.is_file():
        return DEFAULT_MODEL_PATH

    ensure_dir(DEFAULT_MODEL_PATH.parent)
    print(f"Downloading YOLO weights to {DEFAULT_MODEL_PATH} ...")
    try:
        from urllib.request import urlretrieve

        urlretrieve(DEFAULT_MODEL_URL, DEFAULT_MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Failed to download model. Pass --model /path/to/player_detection_best.pt "
            f"or download manually from {DEFAULT_MODEL_URL}"
        ) from exc
    return DEFAULT_MODEL_PATH


def load_yolo(model_path: Path, device: str | None):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Install ultralytics: pip install ultralytics"
        ) from exc

    model = YOLO(str(model_path))
    if device is not None:
        model.to(device)
    return model


def parse_class_filter(class_names: str, model_names: dict[int, str]) -> set[str]:
    wanted = {c.strip().lower() for c in class_names.split(",") if c.strip()}
    known = {name.lower() for name in model_names.values()}
    unknown = wanted - known
    if unknown:
        raise ValueError(
            f"Unknown classes {sorted(unknown)}. Model classes: {sorted(known)}"
        )
    return wanted


def detection_to_dict(box, class_name: str) -> dict:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    conf = float(box.conf[0])
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return {
        "class": class_name,
        "confidence": round(conf, 4),
        "bbox_xyxy": [round(v, 2) for v in (x1, y1, x2, y2)],
        "center_xy": [round(cx, 2), round(cy, 2)],
    }


def draw_detection(frame: np.ndarray, det: dict) -> None:
    x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
    label = f"{det['class']} {det['confidence']:.2f}"
    color = CLASS_COLORS.get(det["class"], (255, 255, 255))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = max(y1 - 4, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 2, ty - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def run_detection(
    video_path: Path,
    output_video: Path,
    output_json: Path | None,
    model,
    conf: float,
    class_filter: set[str],
    device: str | None,
    max_frames: int | None,
    imgsz: int,
) -> dict:
    model_names = {int(k): v for k, v in model.names.items()}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = get_video_frame_count(video_path)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    ensure_dir(output_video.parent)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {output_video}")

    frames_out: list[dict] = []
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc=f"Detect {video_path.name}")
    while frame_idx < total_frames:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            conf=conf,
            imgsz=imgsz,
            verbose=False,
            device=device,
        )
        result = results[0]
        detections: list[dict] = []

        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = model_names[cls_id].lower()
                if class_name not in class_filter:
                    continue
                det = detection_to_dict(box, class_name)
                detections.append(det)
                draw_detection(frame, det)

        # Draw ball on top if both ball and players exist
        balls = [d for d in detections if d["class"] == "ball"]
        if balls:
            best_ball = max(balls, key=lambda d: d["confidence"])
            cx, cy = [int(v) for v in best_ball["center_xy"]]
            cv2.circle(frame, (cx, cy), 6, CLASS_COLORS["ball"], -1)

        cv2.putText(
            frame,
            f"frame {frame_idx}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frames_out.append(
            {
                "frame": frame_idx,
                "time_ms": round(1000.0 * frame_idx / fps, 2),
                "detections": detections,
            }
        )

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    summary = {
        "video": str(video_path.resolve()),
        "output_video": str(output_video.resolve()),
        "model": str(getattr(model, "ckpt_path", None) or "yolo"),
        "fps": fps,
        "width": width,
        "height": height,
        "num_frames": frame_idx,
        "conf_threshold": conf,
        "classes": sorted(class_filter),
        "frames": frames_out,
    }

    if output_json is not None:
        ensure_dir(output_json.parent)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote detections JSON: {output_json}")

    ball_frames = sum(1 for fr in frames_out if any(d["class"] == "ball" for d in fr["detections"]))
    player_frames = sum(
        1 for fr in frames_out if any(d["class"] == "player" for d in fr["detections"])
    )
    print(f"Wrote labeled video: {output_video}")
    print(
        f"Frames with ball: {ball_frames}/{frame_idx}, "
        f"frames with player: {player_frames}/{frame_idx}"
    )
    return summary


def main() -> None:
    args = parse_args()
    video_path = resolve_video_path(args.video)
    clip_id = clip_id_from_video_path(video_path)

    if args.output:
        output_video = Path(args.output)
    else:
        output_video = Path("outputs/detections") / f"{clip_id}_labeled.mp4"

    output_json = Path(args.json) if args.json else output_video.with_suffix(".json")
    model_path = resolve_model_path(args.model)

    model = load_yolo(model_path, args.device)
    class_filter = parse_class_filter(args.classes, model.names)

    run_detection(
        video_path=video_path,
        output_video=output_video,
        output_json=output_json,
        model=model,
        conf=args.conf,
        class_filter=class_filter,
        device=args.device,
        max_frames=args.max_frames,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
