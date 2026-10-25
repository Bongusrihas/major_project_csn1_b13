"""Temporal Causal Transformer (TCT).

Paper Reference:
    "Causal Representation Learning for Predicting Autoimmune Disease Progression
    From Longitudinal Multimodal Clinical Data" (IEEE Access, Vol. 13, 2025).
    Section III.F, Equations 10-12.

Key Components:
    1. Causal Self-Attention with DAG and lower-triangular causal masking (Eq. 10):
           alpha_{ij} = [ M_{ij} * exp(q_i^T k_j / sqrt(d_k)) ] / sum [ exp(q_i^T k_j' / sqrt(d_k)) ]
    2. Time-aware intervention gating (Eq. 11):
           g_t = sigma(W_{Delta} Delta t_t + W_a a_t + b)
    3. Gated context aggregation (Eq. 12):
           c_t = sum_{j=1}^t alpha_{tj} * v_j * g_j
"""
from __future__ import annotations

from typing import Any
import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal temporal masking (Eq. 10)."""

    def __init__(self, d: int, n_heads: int, dropout: float = 0.15):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.dk = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, d)
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute causal masked attention across clinical visits."""
        B, T, d = x.shape
        x_norm = self.norm1(x)
        q, k, v = self.qkv(x_norm).view(B, T, 3, self.n_heads, self.dk).permute(2, 0, 3, 1, 4)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)

        # Causal mask: strictly prevents attention to future visits (j > i)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        valid_mask = causal_mask[None, None, :, :] & pad_mask[:, None, None, :]
        scores = scores.masked_fill(~valid_mask, -1e9)

        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        attn_out = (attn_weights @ v).transpose(1, 2).reshape(B, T, d)
        x = x + self.dropout(self.proj(attn_out))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, attn_weights.mean(dim=1)


class TemporalCausalTransformer(nn.Module):
    """Temporal Causal Transformer (TCT) with intervention gating (Paper Eq. 11-12)."""

    def __init__(self, d_in: int, d_h: int, n_treat: int, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(d_in, d_h)
        self.pos_emb = nn.Embedding(128, d_h)
        self.time_gap_proj = nn.Linear(1, d_h)
        self.layers = nn.ModuleList([
            CausalSelfAttention(d_h, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)
        ])
        # Time-aware intervention gating (Eq. 11)
        self.gate_dt = nn.Linear(1, d_h)
        self.gate_a = nn.Linear(n_treat, d_h)
        self.norm = nn.LayerNorm(d_h)

    def forward(
        self,
        s: torch.Tensor,
        gap: torch.Tensor,
        a: torch.Tensor,
        pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes temporal sequence with intervention gating."""
        B, T, _ = s.shape
        pos = torch.arange(T, device=s.device)[None, :]
        h = self.input_proj(s) + self.pos_emb(pos) + self.time_gap_proj(gap[..., None])

        attn_map = None
        for layer in self.layers:
            h_new, attn_map = layer(h, pad_mask)
            if self.cfg.use_gate:
                # Gating vector g_t (Eq. 11)
                g_t = torch.sigmoid(self.gate_dt(gap[..., None]) + self.gate_a(a))
                h = g_t * h_new + (1.0 - g_t) * h
            else:
                h = h_new

        return self.norm(h), attn_map


class GRUTemporal(nn.Module):
    """Ablation: GRU temporal model without causal attention mechanism."""

    def __init__(self, d_in: int, d_h: int):
        super().__init__()
        self.gru = nn.GRU(d_in, d_h, batch_first=True)
        self.norm = nn.LayerNorm(d_h)

    def forward(self, s: torch.Tensor, *args) -> tuple[torch.Tensor, None]:
        h, _ = self.gru(s)
        return self.norm(h), None
