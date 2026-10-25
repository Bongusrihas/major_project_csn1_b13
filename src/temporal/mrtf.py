"""Multi-Resolution Temporal Fusion (MRTF).

Paper Reference:
    "Causal Representation Learning for Predicting Autoimmune Disease Progression
    From Longitudinal Multimodal Clinical Data" (IEEE Access, Vol. 13, 2025).
    Section III.G, Equations 13-15.

Formulation:
    1. Multi-scale sliding window pooling over latent causal states (Eq. 13):
           z_t^{(s)} = Pooling_s({z_tau}_{tau=t-w_s}^t)
    2. Adaptive temporal fusion gate (Eq. 14):
           gamma_t^{(s)} = sigma(W_s z_t^{(s)} + b_s)
    3. Multi-resolution fused representation (Eq. 15):
           z_t^{fused} = sum_{s=1}^S gamma_t^{(s)} * z_t^{(s)}
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MRTF(nn.Module):
    """Multi-Resolution Temporal Fusion module (Paper Section III.G).
    
    Captures multi-scale disease dynamics across fine (acute changes), medium,
    and coarse (long-term baseline trajectory) temporal resolutions.
    """

    def __init__(self, d_in: int, d_out: int, scales: tuple = (1, 3, 6)):
        super().__init__()
        self.scales = scales
        self.d_out = d_out
        self.projections = nn.ModuleList([nn.Linear(d_in, d_out) for _ in scales])
        self.gate_linear = nn.Linear(d_in * len(scales), d_out * len(scales))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Fuses multi-scale sliding windows along the temporal dimension.
        
        Args:
            z: Latent states of shape (Batch, Time, d_in)
        Returns:
            Multi-resolution fused latent states of shape (Batch, Time, d_out)
        """
        B, T, d_in = z.shape
        # Fast multi-scale sliding window average pooling via prefix cumsum
        cumsum = F.pad(torch.cumsum(z, dim=1), (0, 0, 1, 0))  # (B, T+1, d_in)

        scale_embeddings = []
        for s in self.scales:
            idx = torch.arange(T, device=z.device)
            window_start = (idx + 1 - s).clamp(min=0)
            window_len = (idx + 1 - window_start).float()[None, :, None]
            # Average pooling over sliding window w_s
            pooled_s = (cumsum[:, idx + 1] - cumsum[:, window_start]) / window_len
            scale_embeddings.append(pooled_s)

        # Concatenate scale embeddings for gating
        concat_scales = torch.cat(scale_embeddings, dim=-1)  # (B, T, d_in * S)
        gates = torch.sigmoid(self.gate_linear(concat_scales)).view(B, T, len(self.scales), self.d_out)

        # Projected scale representations
        proj_scales = torch.stack([proj(s_emb) for proj, s_emb in zip(self.projections, scale_embeddings)], dim=2)
        # Gated fusion (Eq. 15)
        z_fused = (gates * proj_scales).sum(dim=2)
        return z_fused
