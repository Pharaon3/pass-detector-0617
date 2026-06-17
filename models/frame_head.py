"""Frame-wise pass probability head."""

from __future__ import annotations

import torch
import torch.nn as nn


class FrameWiseHead(nn.Module):
    """
    Per-frame binary pass probability head.

    Input:  (B, T, d_model)
    Output: (B, T, 1) probabilities in [0, 1]
    """

    def __init__(self, d_model: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)  # (B, T, 1)
        return torch.sigmoid(logits)
