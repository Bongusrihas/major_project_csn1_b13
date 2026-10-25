"""Proposed Major Project Modification: Learnable Edge-Confidence Gating (LECG).

Project Title:
    "Improving Autoimmune Disease Progression Prediction Using
     Confidence-Weighted Causal Graph Learning"

Motivation & Problem in Original Paper:
    - The original R-GCN architecture processes information along static, unweighted edges.
    - However, causal DAGs constructed from observational clinical data inevitably contain
      uncertain, misspecified, or noisy edges (e.g. confounding shortcuts).
    - The paper explicitly notes missing or misspecified edges as an open limitation.

Our Architectural Solution:
    - Introduce a learnable confidence logit theta_e for every existing graph edge:
          w_e = sigmoid(theta_e) in (0, 1)
    - Initialize theta_e = 2.2, giving an initial confidence w_e ~ 0.90 (trust prior DAG).
    - Confidence-weighted relational message passing:
          m_{u -> v} = w_e * W_r * h_u
    - Effective adjacency tensor:
          A_tilde = A * sigmoid(Theta)
    - Non-existent edges (A = 0) remain strictly zero with zero gradient leakage.
    - Spurious or misspecified connections are dynamically attenuated (w_e -> 0),
      while invariant causal mechanisms are preserved and reinforced (w_e -> 1).
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.rgcn_original import RelationalGraphConvOriginal


class StructuralGraphEncoderLECG(nn.Module):
    """Proposed Confidence-Weighted Structural Graph Encoder with Learnable Edge Gating.
    
    Extends the original R-GCN with trainable scalar confidence parameters theta_e:
        w_e = sigmoid(theta_e)
        m_{u -> v} = w_e * W_r * h_u
    """

    def __init__(self, V: int, A: np.ndarray | torch.Tensor, cfg: Any):
        super().__init__()
        self.V = V
        self.cfg = cfg
        d = cfg.d

        # Base relational adjacency tensor (n_rel, V+1, V+1) where node V is latent immune state U
        self.register_buffer("A", torch.as_tensor(A, dtype=torch.float32))

        # ---------------------------------------------------------------------
        # KEY ARCHITECTURAL MODIFICATION: Learnable Edge Confidence Parameter
        # ---------------------------------------------------------------------
        # Trainable logit theta_e for every existing directed edge in the causal DAG
        # Initialized to 2.2 -> sigmoid(2.2) = 0.9002 (starts with 90% confidence)
        self.edge_logits = nn.Parameter(torch.full_like(self.A, cfg.init_edge_confidence))

        # Node value embedding & missingness mask embedding
        self.val_embed = nn.Parameter(torch.randn(V, d) * 0.2)
        self.mask_embed = nn.Parameter(torch.randn(V, d) * 0.1)
        # Base node identity embeddings (+1 for latent immune variable U)
        self.node_identity = nn.Parameter(torch.randn(V + 1, d) * 0.2)

        # 2-layer R-GCN
        self.rgcn_layers = nn.ModuleList([
            RelationalGraphConvOriginal(d, self.A.shape[0]) for _ in range(2)
        ])
        self.pool_attn = nn.Linear(d, 1)

        # Output projection to latent space [z_causal, z_spurious]
        d_out = cfg.d_c + cfg.d_s
        self.proj_out = nn.Linear(2 * d, d_out)
        self.skip_path = nn.Linear(2 * V, d_out)

    def get_effective_adjacency(self) -> torch.Tensor:
        """Returns the confidence-gated effective adjacency: A_tilde = A * sigmoid(theta_e).
        
        Zero edges (non-existent relations) remain strictly zero.
        """
        w_e = torch.sigmoid(self.edge_logits)
        return self.A * w_e

    def get_edge_confidences(self) -> torch.Tensor:
        """Returns the learned edge confidence weights w_e in [0, 1] for all valid DAG edges."""
        return torch.sigmoid(self.edge_logits) * (self.A > 0).float()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Forward pass with confidence-weighted message passing.
        
        Args:
            x: Raw feature values of shape (Batch, V)
            m: Missingness mask of shape (Batch, V)
        Returns:
            Disentangled latent state z_t = [z_causal, z_spurious] of shape (Batch, d_c + d_s)
        """
        B = x.shape[0]
        # Embed observed clinical nodes
        h_obs = self.node_identity[None, :self.V] + x[..., None] * self.val_embed + m[..., None] * self.mask_embed
        # Embed unobserved latent confounder U
        h_latent = self.node_identity[None, self.V:].expand(B, 1, -1)
        h = torch.cat([h_obs, h_latent], dim=1)  # (Batch, V+1, d)

        # Compute dynamic effective adjacency tensor gated by learned edge weights w_e
        A_eff = self.get_effective_adjacency()
        # Edge dropout discourages dependence on any single graph shortcut.
        A_eff = F.dropout(A_eff, p=getattr(self.cfg, "edge_dropout", 0.0), training=self.training)

        # R-GCN message passing with confidence-weighted edges
        for layer in self.rgcn_layers:
            h = layer(h, A_eff)

        # Readout: Attention pooling over observed nodes + latent U readout
        attn_w = F.softmax(self.pool_attn(h[:, :self.V]), dim=1)
        pooled = (attn_w * h[:, :self.V]).sum(dim=1)
        u_rep = h[:, self.V]
        g_repr = torch.cat([pooled, u_rep], dim=-1)

        # Skip connection from raw inputs
        raw_cat = torch.cat([x, m], dim=-1)
        z = self.proj_out(g_repr) + self.skip_path(raw_cat)
        return z
