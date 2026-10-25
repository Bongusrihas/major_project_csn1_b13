"""Flat Linear Baseline Encoder (Paper Section IV.G Component Ablation).

Used when evaluating the ablation condition: 'w/o Graph Structure (R-GCN)'.
Replaces the relational graph convolution with a flat feed-forward projection.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from typing import Any


class FlatEncoder(nn.Module):
    """Encodes clinical features without graph relational structure."""

    def __init__(self, V: int, cfg: Any):
        super().__init__()
        self.V = V
        d_out = cfg.d_c + cfg.d_s
        self.net = nn.Sequential(
            nn.Linear(2 * V, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, d_out)
        )

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Linear mapping from concatenated values and missingness mask."""
        return self.net(torch.cat([x, m], dim=-1))
