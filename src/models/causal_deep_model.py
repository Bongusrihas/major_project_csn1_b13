"""Base Causal Deep Architecture integrating all structural and temporal components.

Paper Reference:
    "Causal Representation Learning for Predicting Autoimmune Disease Progression
    From Longitudinal Multimodal Clinical Data" (IEEE Access, Vol. 13, 2025).
    Algorithm 1 & Section III.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.encoders.flat_encoder import FlatEncoder
from src.encoders.rgcn_lecg import StructuralGraphEncoderLECG
from src.encoders.rgcn_original import StructuralGraphEncoderOriginal
from src.models.config import CausalCfg
from src.temporal.mrtf import MRTF
from src.temporal.tct import GRUTemporal, TemporalCausalTransformer


class CausalDeepModel(nn.Module):
    """Full Longitudinal Causal Deep Learning Model.
    
    Integrates:
    - Structural Graph Encoder (Original static R-GCN or Proposed LECG R-GCN)
    - Multi-Resolution Temporal Fusion (MRTF)
    - Temporal Causal Transformer (TCT)
    - Counterfactual Simulation Engine (CSE)
    - Causal Disentanglement & Invariance (CDI)
    - Dual Uncertainty Quantification (UQ via MC Dropout and MoG)
    """

    def __init__(
        self,
        V: int,
        treat_idx: list[int],
        A: np.ndarray | torch.Tensor,
        cfg: CausalCfg,
        n_nuisance: int = 1
    ):
        super().__init__()
        self.cfg = cfg
        self.V = V
        self.treat_idx = list(treat_idx)
        n_treat = len(self.treat_idx)

        # 1. Structural Graph Encoder (Original Paper or Proposed LECG)
        if not cfg.use_graph:
            self.encoder = FlatEncoder(V, cfg)
        elif cfg.use_lecg:
            self.encoder = StructuralGraphEncoderLECG(V, A, cfg)
        else:
            self.encoder = StructuralGraphEncoderOriginal(V, A, cfg)

        # 2. Multi-Resolution Temporal Fusion (MRTF)
        d_c = cfg.d_c if cfg.use_cdi else cfg.d_c + cfg.d_s
        self.mrtf = MRTF(d_c, cfg.d_h, cfg.scales)

        # 3. Temporal Model (TCT or GRU ablation)
        self.temporal = (
            TemporalCausalTransformer(cfg.d_h, cfg.d_h, n_treat, cfg)
            if cfg.use_tct else GRUTemporal(cfg.d_h, cfg.d_h)
        )

        # 4. Counterfactual Simulation Engine (CSE, Section III.E)
        self.abduction = nn.Linear(cfg.d_h, 2 * cfg.d_eps) if cfg.use_cse else None
        self.action_path = nn.Linear(n_treat, cfg.d_h) if cfg.use_cse else None

        # 5. Prediction Heads
        d_head = cfg.d_h + (cfg.d_eps if cfg.use_cse else 0)
        self.trunk = nn.Sequential(
            nn.Linear(d_head, cfg.d_h),
            nn.GELU(),
            nn.Dropout(cfg.dropout)
        )

        # Binary Flare Event Head (Task 1, Paper Eq. 1)
        self.flare_head = nn.Linear(cfg.d_h, 1)

        # Mixture of Gaussians Severity Head (Task 2, Paper Eq. 23)
        self.k_mix = cfg.k_mix if cfg.use_mog else 1
        self.sev_head = nn.Linear(cfg.d_h, 3 * self.k_mix)

        # Generative Reconstruction Head for ELBO (Paper Eq. 27)
        self.recon_head = nn.Linear(cfg.d_h, V)

        # CDI Spurious Head: ensures spurious latent components capture nuisance noise
        self.nuisance_head = nn.Linear(cfg.d_s, n_nuisance)

    def forward(
        self,
        X: torch.Tensor,
        M: torch.Tensor,
        gap: torch.Tensor,
        pad_mask: torch.Tensor,
        intervention_action: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        sample_noise: bool = True,
        sample: bool | None = None,
        abducted_eps: torch.Tensor | None = None
    ) -> dict:
        """Forward pass through complete causal framework."""
        if sample is not None:
            sample_noise = sample
        B, T, V = X.shape

        # Encode static/spatial graph relationships at each time step
        z = self.encoder(X.reshape(B * T, V), M.reshape(B * T, V)).view(B, T, -1)

        # Causal disentanglement: z = [z_causal, z_spurious]
        if self.cfg.use_cdi:
            zc = z[..., :self.cfg.d_c]
            zs = z[..., self.cfg.d_c:]
        else:
            zc = z
            zs = z[..., :0]

        # Multi-resolution temporal fusion across sliding windows
        s = self.mrtf(zc)

        # Treatment interventions
        a = X[:, :, self.treat_idx]
        if intervention_action is not None:
            a = intervention_action

        # Temporal Causal Transformer with causal masking
        h, attn_weights = self.temporal(s, gap, a, pad_mask)

        out = {"zc": zc, "zs": zs, "h": h, "att": attn_weights}

        # CSE Abduction and surgical intervention propagation
        if self.cfg.use_cse:
            mu_eps, lv_eps = self.abduction(h).chunk(2, dim=-1)
            if noise is None:
                noise = torch.randn_like(mu_eps)
            eps = mu_eps + 0.3 * torch.exp(0.5 * lv_eps) * noise if sample_noise else mu_eps
            if abducted_eps is not None:
                eps = abducted_eps
            out.update(mu_eps=mu_eps, lv_eps=lv_eps, noise=noise, eps=eps)
            h_combined = torch.cat([h + self.action_path(a), eps], dim=-1)
        else:
            h_combined = h

        # Shared feature trunk
        feat = self.trunk(h_combined)

        # Task 1: Binary flare logits
        out["flare"] = self.flare_head(feat).squeeze(-1)

        # Task 2: Mixture of Gaussians parameters for disease severity
        p_mix = self.sev_head(feat).view(B, T, self.k_mix, 3)
        out["pi"] = torch.softmax(p_mix[..., 0], dim=-1)
        out["mu_s"] = p_mix[..., 1] + 1.5
        out["ls"] = p_mix[..., 2].clamp(-3.0, 2.0)

        # Generative observation reconstruction
        out["recon"] = self.recon_head(feat)
        return out

    @staticmethod
    def sev_mean(out: dict) -> torch.Tensor:
        """Predictive mean of severity score under Mixture of Gaussians."""
        return (out["pi"] * out["mu_s"]).sum(dim=-1)

    @staticmethod
    def sev_var(out: dict) -> torch.Tensor:
        """Total predictive variance Var_MoG(y_t) (Paper Eq. 24)."""
        mean = (out["pi"] * out["mu_s"]).sum(dim=-1, keepdim=True)
        comp_var = torch.exp(2.0 * out["ls"])
        mixture_spread = (out["mu_s"] - mean).pow(2)
        return (out["pi"] * (comp_var + mixture_spread)).sum(dim=-1)

    def get_edge_confidences(self) -> torch.Tensor | None:
        """Returns learned edge confidence weights w_e in [0, 1]."""
        if hasattr(self.encoder, "get_edge_confidences"):
            return self.encoder.get_edge_confidences()
        if hasattr(self.encoder, "A"):
            return (self.encoder.A > 0).float()
        return None
