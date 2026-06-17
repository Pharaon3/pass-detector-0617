"""Training script for soccer pass event detection."""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset.dataset import SoccerPassDataset, collate_fn
from models.model import PassDetectionModel, compute_loss
from utils import AverageMeter, ensure_dir, get_device, load_config, print_device_info, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pass detection model")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=None, help="Override config batch size")
    p.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exit with error if CUDA is not available (recommended)",
    )
    return p.parse_args()


def resolve_runtime_config(cfg: dict, device: torch.device) -> tuple[dict, dict, dict]:
    """Apply CPU memory-safe overrides when CUDA is unavailable."""
    runtime_cfg = copy.deepcopy(cfg)
    train_cfg = runtime_cfg["training"]
    model_cfg = runtime_cfg["model"]

    if device.type != "cuda":
        train_cfg["amp"] = False
        train_cfg["batch_size"] = train_cfg.get("cpu_batch_size", 1)
        train_cfg["grad_accum_steps"] = train_cfg.get("cpu_grad_accum_steps", 8)
        train_cfg["num_workers"] = train_cfg.get("cpu_num_workers", 0)

        cpu_backbone = train_cfg.get("cpu_backbone")
        if cpu_backbone:
            model_cfg["backbone"] = cpu_backbone

        cpu_layers = train_cfg.get("cpu_transformer_layers")
        if cpu_layers:
            model_cfg["n_layers"] = cpu_layers

        cpu_chunk = train_cfg.get("cpu_backbone_chunk_size")
        if cpu_chunk:
            model_cfg["backbone_chunk_size"] = cpu_chunk

        print(
            "CPU mode: "
            f"batch_size={train_cfg['batch_size']}, "
            f"grad_accum={train_cfg['grad_accum_steps']}, "
            f"backbone={model_cfg['backbone']}, "
            f"transformer_layers={model_cfg['n_layers']}, "
            f"backbone_chunk_size={model_cfg.get('backbone_chunk_size', 16)}"
        )
    else:
        print(
            "GPU mode: "
            f"batch_size={train_cfg['batch_size']}, "
            f"grad_accum={train_cfg['grad_accum_steps']}, "
            f"backbone={model_cfg['backbone']}, "
            f"amp={train_cfg['amp']}, "
            f"backbone_chunk_size={model_cfg.get('backbone_chunk_size', 32)}"
        )

    return train_cfg, model_cfg, runtime_cfg


def autocast_context(enabled: bool, device: torch.device):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    from torch.cuda.amp import autocast

    return autocast()


def make_grad_scaler(enabled: bool, device: torch.device):
    if not enabled or device.type != "cuda":
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    from torch.cuda.amp import GradScaler

    return GradScaler()


def evaluate_frame_auc(model: PassDetectionModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            labels = batch["labels"].cpu().numpy()
            out = model(video)
            probs = out["frame_probs"].squeeze(-1).cpu().numpy()
            all_probs.append(probs.ravel())
            all_labels.append(labels.ravel())

    if not all_probs:
        return 0.0

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_bin = (y_true >= 0.5).astype(np.int32)
    if len(np.unique(y_bin)) < 2:
        return 0.0
    return float(roc_auc_score(y_bin, y_prob))


def train_one_epoch(
    model: PassDetectionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    peak_loss_weight: float,
    grad_accum_steps: int,
    use_amp: bool,
    log_interval: int,
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    bce_meter = AverageMeter()
    peak_meter = AverageMeter()

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="Train", leave=False)
    use_cuda = device.type == "cuda"

    for step, batch in enumerate(pbar, start=1):
        video = batch["video"].to(device, non_blocking=use_cuda)
        labels = batch["labels"].to(device, non_blocking=use_cuda)
        peak_times = batch["peak_times"]

        with autocast_context(use_amp, device):
            out = model(video)
            losses = compute_loss(
                out["frame_logits"],
                labels,
                peak_times,
                peak_loss_weight=peak_loss_weight,
            )
            loss = losses["loss"] / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = video.size(0)
        loss_meter.update(losses["loss"].item(), bs)
        bce_meter.update(losses["bce"].item(), bs)
        peak_meter.update(losses["peak_mse"].item(), bs)

        if step % log_interval == 0:
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", bce=f"{bce_meter.avg:.4f}")

    return {"loss": loss_meter.avg, "bce": bce_meter.avg, "peak_mse": peak_meter.avg}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    device = get_device(require_cuda=args.require_gpu)
    print(f"Using device: {device}")
    print_device_info(device)

    train_cfg, model_cfg, runtime_cfg = resolve_runtime_config(cfg, device)
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    data_cfg = cfg["data"]
    sw_cfg = cfg["sliding_window"]
    prep_cfg = cfg["preprocessing"]
    label_cfg = cfg["labeling"]

    dataset = SoccerPassDataset(
        data_root=data_cfg["data_root"],
        video_filename=data_cfg["video_filename"],
        label_filename=data_cfg["label_filename"],
        clip_prefix=data_cfg["clip_prefix"],
        window_frames=sw_cfg["window_frames"],
        stride_frames=sw_cfg["stride_frames"],
        num_frames=data_cfg["num_frames"],
        fps=data_cfg["fps"],
        frame_size=prep_cfg["frame_size"],
        mean=prep_cfg["mean"],
        std=prep_cfg["std"],
        pass_radius_sec=label_cfg["pass_radius_sec"],
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "No training samples found. Expected data/clip_XXX/224p.mp4 + label.json pairs."
        )

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=use_cuda,
    )

    model = PassDetectionModel(
        backbone_name=model_cfg["backbone"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg["dropout"],
        pretrained=True,
        backbone_chunk_size=model_cfg.get("backbone_chunk_size", 16),
    ).to(device)

    print(f"Backbone output dim: {model.backbone.out_dim}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["num_epochs"]
    )
    scaler = make_grad_scaler(train_cfg["amp"], device)

    ckpt_dir = ensure_dir(train_cfg["checkpoint_dir"])
    best_auc = 0.0

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            peak_loss_weight=train_cfg["peak_loss_weight"],
            grad_accum_steps=train_cfg["grad_accum_steps"],
            use_amp=train_cfg["amp"],
            log_interval=train_cfg["log_interval"],
        )
        scheduler.step()

        val_auc = evaluate_frame_auc(model, val_loader, device)
        print(
            f"Epoch {epoch}/{train_cfg['num_epochs']} | "
            f"loss={metrics['loss']:.4f} bce={metrics['bce']:.4f} "
            f"peak_mse={metrics['peak_mse']:.4f} val_auc={val_auc:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_auc": val_auc,
            "config": runtime_cfg,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_auc >= best_auc:
            best_auc = val_auc
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  Saved best checkpoint (AUC={best_auc:.4f})")

    print(f"Training complete. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
