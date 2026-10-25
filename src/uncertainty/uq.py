"""Uncertainty Quantification (UQ) and Clinical Prediction Losses.

Paper Reference:
    "Causal Representation Learning for Predicting Autoimmune Disease Progression
    From Longitudinal Multimodal Clinical Data" (IEEE Access, Vol. 13, 2025).
    Section III.I & III.J.1, Equations 20-25.

Dual Uncertainty Decomposition:
    1. Epistemic Uncertainty (Model Uncertainty):
       Quantified via Monte Carlo Dropout sampling over T stochastic forward passes.
       Credible intervals (90% CI) and epistemic standard deviation:
           sigma_{epistemic}^2 = (1 / M) sum_{m=1}^M (hat{y}^{(m)} - bar{y})^2
    2. Aleatoric Uncertainty (Clinical Data Heterogeneity):
       Modeled via Mixture of Gaussians (MoG) severity head with parameters (pi, mu, sigma):
           p(Y_{sev} | x) = sum_{k=1}^K pi_k * N(mu_k, sigma_k^2)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> torch.Tensor:
    """Focal Loss for imbalanced binary flare-up prediction (Paper Eq. 20).
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    weight = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return -weight * (1.0 - p_t).pow(gamma) * torch.log(p_t.clamp_min(1e-6))


def mog_negative_log_likelihood(out: dict, targets: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood for Mixture of Gaussians severity output (Paper Eq. 23).
    
    L_{MoG} = -log( sum_{k=1}^K pi_k * N(y; mu_k, sigma_k^2) )
    """
    sigma = torch.exp(out["ls"])
    diff = (targets[..., None] - out["mu_s"]) / sigma
    log_gaussian = -0.5 * diff.pow(2) - out["ls"] - 0.9189385
    log_probs = torch.log(out["pi"].clamp_min(1e-6)) + log_gaussian
    return -torch.logsumexp(log_probs, dim=-1)


class MoGSeverityHead(nn.Module):
    """Mixture of Gaussians head for ordinal disease severity with aleatoric UQ (Eq. 22)."""

    def __init__(self, d_h: int, k_mix: int = 3):
        super().__init__()
        self.k_mix = k_mix
        self.pi_linear = nn.Linear(d_h, k_mix)
        self.mu_linear = nn.Linear(d_h, k_mix)
        self.log_sigma_linear = nn.Linear(d_h, k_mix)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi = F.softmax(self.pi_linear(h), dim=-1)
        mu = F.softplus(self.mu_linear(h))  # Severity is non-negative
        ls = self.log_sigma_linear(h).clamp(min=-3.0, max=2.0)
        return pi, mu, ls
