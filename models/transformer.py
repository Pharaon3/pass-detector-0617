"""Temporal Transformer encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """
    Transformer encoder over temporal dimension.

    Input:  (B, T, D_in)
    Output: (B, T, d_model)
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :t, :]
        x = self.encoder(x)
        return self.norm(x)
