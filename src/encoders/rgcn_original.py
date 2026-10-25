"""Original Paper Implementation: Structural Graph Encoder (R-GCN).

Paper Reference:
    "Causal Representation Learning for Predicting Autoimmune Disease Progression
    From Longitudinal Multimodal Clinical Data" (IEEE Access, Vol. 13, 2025).
    Section III.D, Equations 5 & 6.

Key Mechanism:
    - Relational Graph Convolutional Network (R-GCN) message passing over a fixed
      causal Directed Acyclic Graph (DAG) G_t.
    - Adjacency tensor A in {0, 1}^{R x V x V} is fixed and unweighted.
    - Relational message passing equation (Paper Eq. 6):
        h_v^{(l+1)} = sigma( sum_{r in R} sum_{u in N_r(v)} (1 / c_{v,r}) W_r^{(l)} h_u^{(l)} + W_0^{(l)} h_v^{(l)} )
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalGraphConvOriginal(nn.Module):
    """Standard Relational Graph Convolutional Network layer from original paper (Eq. 6).
    
    Computes relational message passing with fixed relation-specific weights W_r.
    """

    def __init__(self, d: int, n_rel: int):
        super().__init__()
        self.d = d
        self.n_rel = n_rel
        self.w_self = nn.Linear(d, d)
        self.w_rel = nn.Parameter(torch.randn(n_rel, d, d) * (1.0 / math.sqrt(d)))
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Forward message passing.
        
        Args:
            h: Node representations of shape (Batch, Nodes, d)
            A: Static relational adjacency matrix (n_rel, Nodes, Nodes)
        Returns:
            Updated node representations of shape (Batch, Nodes, d)
        """
        # msg: (Batch, n_rel, Nodes, d)
        msg = torch.einsum("rij,bjd->brid", A, h)
        # transform: (Batch, Nodes, d)
        out_msg = torch.einsum("brid,rde->bie", msg, self.w_rel)
        h_new = self.w_self(h) + out_msg
        return self.norm(h + F.relu(h_new))


class StructuralGraphEncoderOriginal(nn.Module):
    """Original Paper Structural Graph Encoder (Paper Section III.D, Eq. 5).
    
    Encodes clinical variables at time t using the unweighted causal DAG G_t:
        z_t = GNNEncode(G_t) = sum_{v in V_t} phi_v(x_v^t, {x_u^t | (u -> v) in E_t})
    
    Adjacency tensor A is fixed and unweighted (no learned confidence gating).
    """

    def __init__(self, V: int, A: np.ndarray | torch.Tensor, cfg: Any):
        super().__init__()
        self.V = V
        self.cfg = cfg
        d = cfg.d

        # Fixed relational adjacency tensor (n_rel, V+1, V+1) where node V is latent immune state U
        self.register_buffer("A", torch.as_tensor(A, dtype=torch.float32))

        # Value and observation mask embeddings
        self.val_embed = nn.Parameter(torch.randn(V, d) * 0.2)
        self.mask_embed = nn.Parameter(torch.randn(V, d) * 0.1)
        # Node identity embeddings (+1 for latent immune variable U)
        self.node_identity = nn.Parameter(torch.randn(V + 1, d) * 0.2)

        # 2-layer R-GCN
        self.rgcn_layers = nn.ModuleList([
            RelationalGraphConvOriginal(d, self.A.shape[0]) for _ in range(2)
        ])
        self.pool_attn = nn.Linear(d, 1)

        # Output projections to [z_causal, z_spurious]
        d_out = cfg.d_c + cfg.d_s
        self.proj_out = nn.Linear(2 * d, d_out)
        self.skip_path = nn.Linear(2 * V, d_out)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Encodes clinical feature values x and observation masks m at visit t.
        
        Args:
            x: Raw feature values of shape (Batch, V)
            m: Missingness mask of shape (Batch, V) (1 = observed, 0 = imputed)
        Returns:
            Disentangled latent state z_t = [z_causal, z_spurious] of shape (Batch, d_c + d_s)
        """
        B = x.shape[0]
        # Embed observed clinical nodes
        h_obs = self.node_identity[None, :self.V] + x[..., None] * self.val_embed + m[..., None] * self.mask_embed
        # Embed unobserved latent confounder U
        h_latent = self.node_identity[None, self.V:].expand(B, 1, -1)
        h = torch.cat([h_obs, h_latent], dim=1)  # (Batch, V+1, d)

        # R-GCN message passing with static adjacency A
        for layer in self.rgcn_layers:
            h = layer(h, self.A)

        # Readout: Attention pooling + latent U readout
        attn_w = F.softmax(self.pool_attn(h[:, :self.V]), dim=1)
        pooled = (attn_w * h[:, :self.V]).sum(dim=1)
        u_rep = h[:, self.V]
        g_repr = torch.cat([pooled, u_rep], dim=-1)

        # Skip connection directly from raw observed inputs
        raw_cat = torch.cat([x, m], dim=-1)
        z = self.proj_out(g_repr) + self.skip_path(raw_cat)
        return z
