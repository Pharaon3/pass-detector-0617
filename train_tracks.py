"""Train pass event detector from player track features."""

from __future__ import annotations

import argparse
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.track_dataset import TrackPassDataset, track_collate_fn
from models.track_model import TrackPassModel, compute_track_loss
from utils import AverageMeter, ensure_dir, get_device, list_clips, load_config, print_device_info, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train track-based pass detector")
    p.add_argument("--config", type=str, default="config_tracks.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-clip-ratio", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--require-gpu", action="store_true")
    return p.parse_args()


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


def split_clip_ids(data_root: Path, val_ratio: float, seed: int, **list_kwargs) -> tuple[list[str], list[str]]:
    clips = list_clips(data_root, **list_kwargs)
    clip_ids = sorted(c.clip_id for c in clips)
    rng = random.Random(seed)
    rng.shuffle(clip_ids)
    val_n = max(1, int(round(len(clip_ids) * val_ratio)))
    val_ids = clip_ids[:val_n]
    train_ids = clip_ids[val_n:]
    if not train_ids:
        train_ids, val_ids = clip_ids[val_n:], clip_ids[:val_n]
    return train_ids, val_ids


@torch.no_grad()
def evaluate_auc(model: TrackPassModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    probs_all, labels_all = [], []
    for batch in loader:
        tracks = batch["tracks"].to(device)
        labels = batch["labels"].cpu().numpy()
        out = model(tracks)
        probs = out["frame_probs"].squeeze(-1).cpu().numpy()
        probs_all.append(probs.ravel())
        labels_all.append(labels.ravel())
    if not probs_all:
        return 0.0
    y_prob = np.concatenate(probs_all)
    y_true = np.concatenate(labels_all)
    y_bin = (y_true >= 0.5).astype(np.int32)
    if len(np.unique(y_bin)) < 2:
        return 0.0
    return float(roc_auc_score(y_bin, y_prob))


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    peak_loss_weight,
    grad_accum_steps,
    use_amp,
    log_interval,
) -> dict[str, float]:
    model.train()
    loss_m = AverageMeter()
    bce_m = AverageMeter()
    peak_m = AverageMeter()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="Train", leave=False)
    for step, batch in enumerate(pbar, start=1):
        tracks = batch["tracks"].to(device, non_blocking=device.type == "cuda")
        labels = batch["labels"].to(device, non_blocking=device.type == "cuda")
        peak_times = batch["peak_times"]

        with autocast_context(use_amp, device):
            out = model(tracks)
            losses = compute_track_loss(
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

        bs = tracks.size(0)
        loss_m.update(losses["loss"].item(), bs)
        bce_m.update(losses["bce"].item(), bs)
        peak_m.update(losses["peak_mse"].item(), bs)
        if step % log_interval == 0:
            pbar.set_postfix(loss=f"{loss_m.avg:.4f}")

    return {"loss": loss_m.avg, "bce": bce_m.avg, "peak_mse": peak_m.avg}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    device = get_device(require_cuda=args.require_gpu)
    print(f"Using device: {device}")
    print_device_info(device)

    data_cfg = cfg["data"]
    track_cfg = cfg["tracks"]
    sw_cfg = cfg["sliding_window"]
    label_cfg = cfg["labeling"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    val_ratio = args.val_clip_ratio if args.val_clip_ratio is not None else train_cfg.get("val_clip_ratio", 0.2)

    list_kwargs = dict(
        video_filename=data_cfg["video_filename"],
        label_filename=data_cfg["label_filename"],
        clip_prefix=data_cfg["clip_prefix"],
    )
    train_ids, val_ids = split_clip_ids(Path(data_cfg["data_root"]), val_ratio, args.seed, **list_kwargs)
    print(f"Clip split: {len(train_ids)} train, {len(val_ids)} val")
    print(f"Val clips: {val_ids}")

    ds_kwargs = dict(
        data_root=data_cfg["data_root"],
        window_cache_dir=track_cfg["window_cache_dir"],
        window_frames=sw_cfg["window_frames"],
        stride_frames=sw_cfg["stride_frames"],
        num_frames=data_cfg.get("num_frames"),
        fps=data_cfg["fps"],
        pass_radius_sec=label_cfg["pass_radius_sec"],
        max_players=track_cfg["max_players"],
        **list_kwargs,
    )

    train_ds = TrackPassDataset(clip_ids=train_ids, **ds_kwargs)
    val_ds = TrackPassDataset(clip_ids=val_ids, **ds_kwargs)
    if len(train_ds) == 0:
        raise RuntimeError(
            "No training windows. Run: python extract_tracks.py --windows"
        )

    wf = sw_cfg["window_frames"]
    print(
        f"Track training: {len(train_ds)} train windows, {len(val_ds)} val windows "
        f"({wf} frames = {wf / data_cfg['fps']:.0f}s video per sample)"
    )

    use_cuda = device.type == "cuda"
    use_amp = train_cfg.get("amp", True) and use_cuda
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        collate_fn=track_collate_fn,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
        collate_fn=track_collate_fn,
        pin_memory=use_cuda,
    )

    model = TrackPassModel(
        max_players=track_cfg["max_players"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["num_epochs"]
    )
    scaler = make_grad_scaler(use_amp, device)
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
            grad_accum_steps=train_cfg.get("grad_accum_steps", 1),
            use_amp=use_amp,
            log_interval=train_cfg.get("log_interval", 10),
        )
        scheduler.step()
        val_auc = evaluate_auc(model, val_loader, device)
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
            "config": cfg,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_auc >= best_auc:
            best_auc = val_auc
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  Saved best checkpoint (AUC={best_auc:.4f})")

    print(f"Training complete. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
