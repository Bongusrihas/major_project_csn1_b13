"""Feature Attribution via Integrated Gradients.

Paper Reference:
    "Causal Representation Learning for Predicting Autoimmune Disease Progression
    From Longitudinal Multimodal Clinical Data" (IEEE Access, Vol. 13, 2025).
    Section III.J.2, Equation 26.

Formulation:
    IG_i(x) = (x_i - x_i') * int_0^1 [ partial f(x' + alpha (x - x')) / partial x_i ] d alpha
    Approximated by Riemann sum over m interpolation steps.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn as nn


def compute_integrated_gradients(
    model: nn.Module,
    dataset: Any,
    patient_idx: int,
    visit_idx: int,
    steps: int = 30,
    device: torch.device = torch.device("cpu")
) -> np.ndarray:
    """Computes Integrated Gradients feature attribution at visit t.
    
    Args:
        model: Trained causal deep model
        dataset: SequenceDataset containing patient features
        patient_idx: Patient index
        visit_idx: Time-step t
        steps: Number of Riemann interpolation steps
        device: Hardware device
        
    Returns:
        Feature attribution array of shape (V,)
    """
    model.eval()
    pi = patient_idx
    t = visit_idx

    X_orig = torch.as_tensor(dataset.X[pi:pi+1], dtype=torch.float32, device=device)
    M = torch.as_tensor(dataset.M[pi:pi+1], dtype=torch.float32, device=device)
    gap = torch.as_tensor(dataset.gap[pi:pi+1], dtype=torch.float32, device=device)
    pad = torch.as_tensor(dataset.pad[pi:pi+1], dtype=torch.bool, device=device)

    # Baseline input: zeros (mean-imputed baseline)
    X_base = torch.zeros_like(X_orig)
    alphas = torch.linspace(0.0, 1.0, steps, device=device)

    # Accumulate gradients
    total_grads = torch.zeros_like(X_orig)

    for alpha in alphas:
        X_step = (X_base + alpha * (X_orig - X_base)).clone().detach().requires_grad_(True)
        out = model(X_step, M, gap, pad)
        # Compute gradient with respect to flare risk logit at visit t
        logit = out["flare"][0, t]
        logit.backward()
        if X_step.grad is not None:
            total_grads += X_step.grad

    # Riemann sum approximation (Eq. 26)
    avg_grads = total_grads / steps
    ig = ((X_orig - X_base) * avg_grads).detach().cpu().numpy()[0, t]
    return ig
