"""Gate Shift Module for lightweight temporal modeling."""

from __future__ import annotations

import torch
import torch.nn as nn


class GateShiftModule(nn.Module):
    """
    GSM: splits channels into three groups and shifts them along time
    (backward / identity / forward) to capture motion cues without extra params.
    """

    def __init__(self, n_div: int = 8) -> None:
        super().__init__()
        self.n_div = n_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        b, t, c = x.shape
        fold = c // self.n_div
        if fold == 0:
            return x

        out = x.clone()
        out[:, 1:, :fold] = x[:, :-1, :fold]           # shift backward
        out[:, :-1, fold : 2 * fold] = x[:, 1:, fold : 2 * fold]  # shift forward
        # middle channels unchanged
        return out
