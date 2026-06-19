"""Fine-tune EasyChamp YOLO on local detection_data/."""

from __future__ import annotations

import argparse
from pathlib import Path

from detect_ball_players import resolve_model_path


DEFAULT_DATA_YAML = Path("detection_data/data.yaml")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune soccer YOLO detector")
    p.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_YAML),
        help="Path to data.yaml",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Starting weights (.pt). Default: EasyChamp player_detection_best.pt",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0", help="GPU id or cpu")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--name", type=str, default="soccer_finetune")
    p.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs without improvement)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume last run in --project/--name",
    )
    p.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Freeze first N layers (0 = full fine-tune)",
    )
    return p.parse_args()


def count_pairs(data_root: Path, split: str) -> tuple[int, int]:
    img_dir = data_root / "images" / split
    lbl_dir = data_root / "labels" / split
    if not img_dir.is_dir():
        return 0, 0

    images = {p.stem for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    labels = {p.stem for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else set()
    return len(images), len(images & labels)


def validate_dataset(data_yaml: Path) -> Path:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    data_root = data_yaml.parent.resolve()
    train_n, train_labeled = count_pairs(data_root, "train")
    val_n, val_labeled = count_pairs(data_root, "val")

    if train_n == 0:
        raise RuntimeError(
            "No training images in detection_data/images/train/.\n"
            "Run: python export_detection_frames.py --data-root data\n"
            "Then add labels under detection_data/labels/train/"
        )
    if train_labeled == 0:
        raise RuntimeError(
            f"Found {train_n} train images but no matching labels in labels/train/.\n"
            "See detection_data/LABEL_FORMAT.txt and detection_data/examples/"
        )
    if val_n == 0:
        print("Warning: no val images — consider adding some to images/val/ for monitoring.")

    print(f"Dataset: {train_labeled}/{train_n} train images labeled, {val_labeled}/{val_n} val")
    return data_root


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics: pip install ultralytics") from exc

    data_yaml = Path(args.data).resolve()
    validate_dataset(data_yaml)
    model_path = resolve_model_path(args.model)

    print(f"Starting weights: {model_path}")
    print(f"Data config: {data_yaml}")

    model = YOLO(str(model_path))
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        freeze=args.freeze,
        exist_ok=True,
        pretrained=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nTraining done. Best weights: {best}")
    print(f"Run inference:\n  python detect_ball_players.py --video data/clip_239/224p.mp4 --model {best}")
    return results


if __name__ == "__main__":
    main()
