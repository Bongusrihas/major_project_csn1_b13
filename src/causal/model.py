"""Causal Deep Learning Architecture (Unified and Backward-Compatible Bridge).

Imports from modular components:
- src.models.config: CausalCfg
- src.encoders: StructuralGraphEncoderOriginal, StructuralGraphEncoderLECG, RelationalGraphConvOriginal, FlatEncoder
- src.temporal: MRTF, TemporalCausalTransformer, CausalSelfAttention, GRUTemporal
- src.uncertainty: focal_loss, mog_negative_log_likelihood, MoGSeverityHead
- src.disentanglement: hsic_estimator, maximum_mean_discrepancy, contrastive_trajectory_loss, total_correlation_penalty
- src.models: CausalDeepModel, OriginalPaperCausalModel, LECGImprovedCausalModel
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import CausalCfg
from src.encoders.rgcn_original import RelationalGraphConvOriginal as RelationalGraphConv, StructuralGraphEncoderOriginal
from src.encoders.rgcn_lecg import StructuralGraphEncoderLECG
from src.encoders.flat_encoder import FlatEncoder
from src.temporal.mrtf import MRTF
from src.temporal.tct import CausalSelfAttention, TemporalCausalTransformer, GRUTemporal
from src.uncertainty.uq import focal_loss, mog_negative_log_likelihood, MoGSeverityHead
from src.disentanglement.losses import (
    contrastive_trajectory_loss,
    hsic_estimator,
    maximum_mean_discrepancy,
    total_correlation_penalty,
)
from src.models.causal_deep_model import CausalDeepModel
from src.models.original_paper_model import OriginalPaperCausalModel
from src.models.lecg_improved_model import LECGImprovedCausalModel

# Unified structural graph encoder that toggles LECG based on cfg.use_lecg
class StructuralGraphEncoder(nn.Module):
    """Bridge encoder supporting both Original Paper and Proposed LECG."""

    def __init__(self, V: int, A: np.ndarray | torch.Tensor, cfg: CausalCfg):
        super().__init__()
        self.V = V
        self.cfg = cfg
        d = cfg.d

        self.register_buffer("A", torch.as_tensor(A, dtype=torch.float32))

        if cfg.use_lecg:
            self.edge_logits = nn.Parameter(torch.full_like(self.A, cfg.init_edge_confidence))
        else:
            self.register_parameter("edge_logits", None)

        self.val_embed = nn.Parameter(torch.randn(V, d) * 0.2)
        self.mask_embed = nn.Parameter(torch.randn(V, d) * 0.1)
        self.node_identity = nn.Parameter(torch.randn(V + 1, d) * 0.2)

        self.rgcn_layers = nn.ModuleList([RelationalGraphConv(d, self.A.shape[0]) for _ in range(2)])
        self.pool_attn = nn.Linear(d, 1)

        d_out = cfg.d_c + cfg.d_s
        self.proj_out = nn.Linear(2 * d, d_out)
        self.skip_path = nn.Linear(2 * V, d_out)

    def get_effective_adjacency(self) -> torch.Tensor:
        if self.cfg.use_lecg and self.edge_logits is not None:
            return self.A * torch.sigmoid(self.edge_logits)
        return self.A

    def get_edge_confidences(self) -> torch.Tensor:
        if self.cfg.use_lecg and self.edge_logits is not None:
            return torch.sigmoid(self.edge_logits) * (self.A > 0).float()
        return (self.A > 0).float()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h_obs = self.node_identity[None, :self.V] + x[..., None] * self.val_embed + m[..., None] * self.mask_embed
        h_latent = self.node_identity[None, self.V:].expand(B, 1, -1)
        h = torch.cat([h_obs, h_latent], dim=1)

        A_eff = self.get_effective_adjacency()
        for layer in self.rgcn_layers:
            h = layer(h, A_eff)

        attn_w = F.softmax(self.pool_attn(h[:, :self.V]), dim=1)
        pooled = (attn_w * h[:, :self.V]).sum(dim=1)
        u_rep = h[:, self.V]
        g_repr = torch.cat([pooled, u_rep], dim=-1)

        raw_cat = torch.cat([x, m], dim=-1)
        z = self.proj_out(g_repr) + self.skip_path(raw_cat)
        return z


# Backward compatibility aliases
CausalModel = CausalDeepModel
focal_bce = focal_loss
mog_nll = mog_negative_log_likelihood
hsic = hsic_estimator
mmd = maximum_mean_discrepancy
supcon = contrastive_trajectory_loss

__all__ = [
    "CausalCfg",
    "RelationalGraphConv",
    "StructuralGraphEncoder",
    "StructuralGraphEncoderOriginal",
    "StructuralGraphEncoderLECG",
    "FlatEncoder",
    "MRTF",
    "CausalSelfAttention",
    "TemporalCausalTransformer",
    "GRUTemporal",
    "CausalDeepModel",
    "OriginalPaperCausalModel",
    "LECGImprovedCausalModel",
    "focal_loss",
    "mog_negative_log_likelihood",
    "hsic_estimator",
    "maximum_mean_discrepancy",
    "contrastive_trajectory_loss",
    "total_correlation_penalty",
]
