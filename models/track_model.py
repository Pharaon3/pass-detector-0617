"""Pass event detector from player track features (no video backbone)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.frame_head import FrameWiseHead
from models.transformer import TemporalTransformer
from tracks.features import GLOBAL_FEAT_DIM, PLAYER_FEAT_DIM


class TrackPassModel(nn.Module):
    """
    Input:  (B, T, F) track feature vectors per frame
    Output: frame-wise pass logits
    """

    def __init__(
        self,
        max_players: int = 22,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_players = max_players
        self.player_feat_dim = PLAYER_FEAT_DIM
        self.global_feat_dim = GLOBAL_FEAT_DIM
        self.in_dim = max_players * PLAYER_FEAT_DIM + GLOBAL_FEAT_DIM

        self.player_encoder = nn.Sequential(
            nn.Linear(PLAYER_FEAT_DIM, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_FEAT_DIM, d_model // 4),
            nn.GELU(),
        )
        self.fuse = nn.Linear(d_model // 2 + d_model // 4, d_model)

        self.temporal = TemporalTransformer(
            d_in=d_model,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.frame_head = FrameWiseHead(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, F) full feature vector per frame
        """
        b, t, f = x.shape
        p = self.max_players
        pf = self.player_feat_dim

        player_flat = x[:, :, : p * pf].reshape(b, t, p, pf)
        global_feat = x[:, :, p * pf : p * pf + self.global_feat_dim]

        player_emb = self.player_encoder(player_flat)
        player_pool = player_emb.mean(dim=2)
        global_emb = self.global_encoder(global_feat)
        fused = self.fuse(torch.cat([player_pool, global_emb], dim=-1))

        temporal_feats = self.temporal(fused)
        frame_logits = self.frame_head(temporal_feats)
        return {
            "frame_logits": frame_logits,
            "frame_probs": torch.sigmoid(frame_logits),
            "features": temporal_feats,
        }


def compute_track_loss(
    frame_logits: torch.Tensor,
    labels: torch.Tensor,
    peak_times_list: list[torch.Tensor],
    peak_loss_weight: float = 0.2,
) -> dict[str, torch.Tensor]:
    logits = frame_logits.squeeze(-1)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
    probs = torch.sigmoid(logits)

    peak_mse_vals = []
    for i, gt_peaks in enumerate(peak_times_list):
        if gt_peaks.numel() == 0:
            continue
        p = probs[i]
        t_len = p.shape[0]
        weights = F.softmax(p * 10.0, dim=0)
        indices = torch.arange(t_len, device=p.device, dtype=p.dtype) / max(t_len - 1, 1)
        pred_peak = (weights * indices).sum()
        gt = gt_peaks.to(p.device)
        closest = gt[(gt - pred_peak).abs().argmin()]
        peak_mse_vals.append((pred_peak - closest) ** 2)

    if peak_mse_vals:
        peak_mse = torch.stack(peak_mse_vals).mean()
    else:
        peak_mse = torch.tensor(0.0, device=probs.device)

    total = bce + peak_loss_weight * peak_mse
    return {"loss": total, "bce": bce, "peak_mse": peak_mse}
