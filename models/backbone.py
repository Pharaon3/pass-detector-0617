"""RegNetY backbone via timm."""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class RegNetYBackbone(nn.Module):
    """
    Frame-wise feature extractor using timm RegNetY.

    Input:  (B, T, 3, H, W)
    Output: (B, T, D) where D = backbone feature dimension
    """

    def __init__(
        self,
        model_name: str = "regnety_008",
        pretrained: bool = True,
        chunk_size: int = 16,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.out_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        flat = x.view(b * t, c, h, w)
        n = flat.shape[0]
        chunk = self.chunk_size if self.chunk_size > 0 else n

        if n <= chunk:
            feats = self.backbone(flat)
            return feats.view(b, t, -1)

        parts = []
        for start in range(0, n, chunk):
            parts.append(self.backbone(flat[start : start + chunk]))
        feats = torch.cat(parts, dim=0)
        return feats.view(b, t, -1)
