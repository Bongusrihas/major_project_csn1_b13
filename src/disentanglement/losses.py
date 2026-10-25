"""Causal Disentanglement & Invariance (CDI) Loss Functions.

Paper Reference:
    "Causal Representation Learning for Predicting Autoimmune Disease Progression
    From Longitudinal Multimodal Clinical Data" (IEEE Access, Vol. 13, 2025).
    Section III.D, Equations 17-19, 28, 31.

Loss Functions:
    1. Hilbert-Schmidt Independence Criterion (HSIC, Eq. 17):
       Penalizes statistical dependence between invariant causal representation z_c
       and clinical nuisance covariates (e.g. site, batch, demographics).
    2. Maximum Mean Discrepancy (MMD, Eq. 18):
       Forces representation invariance across heterogeneous patient subgroups.
    3. Supervised Contrastive Trajectory Loss (Eq. 31):
       Pulls latent trajectories of patients with concordant flare outcomes closer.
    4. Total Correlation Penalty (Eq. 28):
       Encourages factorized, non-redundant latent dimension representations.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def hsic_estimator(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Hilbert-Schmidt Independence Criterion (HSIC) penalty (Paper Eq. 17).
    
    Minimizes statistical dependence between z_c and nuisance covariates.
    """
    n = x.shape[0]
    if n < 4:
        return torch.tensor(0.0, device=x.device)

    def _rbf_kernel(z):
        dists = torch.cdist(z, z).pow(2)
        median_val = dists.detach().median().clamp_min(1e-3)
        return torch.exp(-dists / (2.0 * median_val + 1e-6))

    K = _rbf_kernel(x)
    L = _rbf_kernel(y)
    H = torch.eye(n, device=x.device) - (1.0 / n)
    return torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)


def maximum_mean_discrepancy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Maximum Mean Discrepancy (MMD) loss across subgroups (Paper Eq. 18)."""
    if len(x) < 4 or len(y) < 4:
        return torch.tensor(0.0, device=x.device)

    all_pts = torch.cat([x, y], dim=0)
    bandwidth = torch.cdist(all_pts, all_pts).pow(2).detach().median().clamp_min(1e-3)

    def _k(a, b):
        return torch.exp(-torch.cdist(a, b).pow(2) / (2.0 * bandwidth + 1e-6))

    return _k(x, x).mean() + _k(y, y).mean() - 2.0 * _k(x, y).mean()


def contrastive_trajectory_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.3) -> torch.Tensor:
    """Supervised contrastive loss on latent trajectories (Paper Eq. 31)."""
    if len(z) < 4:
        return torch.tensor(0.0, device=z.device)

    z_norm = F.normalize(z, dim=-1)
    similarity = (z_norm @ z_norm.T) / temperature
    n = len(z)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    positives = (y[:, None] == y[None, :]) & ~eye

    log_prob = similarity - torch.logsumexp(similarity.masked_fill(eye, -1e9), dim=1, keepdim=True)
    num_pos = positives.sum(dim=1)
    valid_mask = num_pos > 0

    if not valid_mask.any():
        return torch.tensor(0.0, device=z.device)

    return -((log_prob * positives).sum(dim=1)[valid_mask] / num_pos[valid_mask]).mean()


def total_correlation_penalty(q_z: torch.Tensor) -> torch.Tensor:
    """Total Correlation penalty (Paper Eq. 28) encouraging factorized latent codes."""
    if q_z.shape[0] < 4:
        return torch.tensor(0.0, device=q_z.device)
    centered = q_z - q_z.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (q_z.shape[0] - 1)
    diag = torch.diag(cov)
    corr = cov / (torch.sqrt(diag[:, None] * diag[None, :]) + 1e-6)
    off_diag = corr.pow(2) * (~torch.eye(corr.shape[0], device=q_z.device, dtype=torch.bool)).float()
    return off_diag.mean()
