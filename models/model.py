"""Full pass detection model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import RegNetYBackbone
from models.frame_head import FrameWiseHead
from models.gsm import GateShiftModule
from models.transformer import TemporalTransformer


class PassDetectionModel(nn.Module):
    """
    End-to-end temporal pass event detector.

    Pipeline:
      frames -> RegNetY -> GSM -> Transformer -> frame-wise sigmoid head
    """

    def __init__(
        self,
        backbone_name: str = "regnety_008",
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        pretrained: bool = True,
        backbone_chunk_size: int = 16,
    ) -> None:
        super().__init__()
        self.backbone = RegNetYBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            chunk_size=backbone_chunk_size,
        )
        self.gsm = GateShiftModule(n_div=8)
        self.temporal = TemporalTransformer(
            d_in=self.backbone.out_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.frame_head = FrameWiseHead(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, 3, H, W)

        Returns:
            frame_probs: (B, T, 1)
            features:    (B, T, d_model)
        """
        feats = self.backbone(x)
        feats = self.gsm(feats)
        temporal_feats = self.temporal(feats)
        frame_probs = self.frame_head(temporal_feats)
        return {
            "frame_probs": frame_probs,
            "features": temporal_feats,
        }


def compute_loss(
    frame_probs: torch.Tensor,
    labels: torch.Tensor,
    peak_times_list: list[torch.Tensor],
    peak_loss_weight: float = 0.2,
) -> dict[str, torch.Tensor]:
    """
    Combined loss:
      BCE(frame_probs, frame_labels) + weight * MSE(peak_time_pred, gt_peak_time)
    """
    probs = frame_probs.squeeze(-1)  # (B, T)
    bce = F.binary_cross_entropy(probs, labels, reduction="mean")

    peak_mse_vals = []
    for i, gt_peaks in enumerate(peak_times_list):
        if gt_peaks.numel() == 0:
            continue
        p = probs[i]
        t_len = p.shape[0]
        # Soft argmax peak locations (differentiable)
        weights = F.softmax(p * 10.0, dim=0)
        indices = torch.arange(t_len, device=p.device, dtype=p.dtype) / max(t_len - 1, 1)
        pred_peak = (weights * indices).sum()
        # Match closest GT peak in window
        gt = gt_peaks.to(p.device)
        closest = gt[(gt - pred_peak).abs().argmin()]
        peak_mse_vals.append((pred_peak - closest) ** 2)

    if peak_mse_vals:
        peak_mse = torch.stack(peak_mse_vals).mean()
    else:
        peak_mse = torch.tensor(0.0, device=probs.device)

    total = bce + peak_loss_weight * peak_mse
    return {"loss": total, "bce": bce, "peak_mse": peak_mse}
