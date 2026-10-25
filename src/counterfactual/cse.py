"""Counterfactual Simulation Engine (CSE).

Paper Reference:
    "Causal Representation Learning for Predicting Autoimmune Disease Progression
    From Longitudinal Multimodal Clinical Data" (IEEE Access, Vol. 13, 2025).
    Section III.E, Equations 7-9.

Three-Step Counterfactual Protocol:
    1. Abduction (Eq. 8):
       Infer individual-specific exogenous background noise variables epsilon:
           q(epsilon | x_{1:t}, a_{1:t}) ~ N(mu_{eps}(x, z), diag(sigma_{eps}^2(x, z)))
    2. Surgical Intervention (Eq. 9):
       Replace factual treatment a_t with hypothetical alternative a'_t via do-calculus:
           do(A_t = a'_t)
    3. Prediction / Potential Outcome Evaluation:
       Evaluate modified state through causal prediction heads while keeping
       abducted individual noise epsilon fixed:
           hat{Y}_{t+1}(a'_t) = f_{outcome}(z_t^{modified}, epsilon)
"""
from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn as nn


class CSERepresentationModule(nn.Module):
    """Subnetwork within CausalDeepModel for Counterfactual Abduction and Surgical Modification."""

    def __init__(self, d_c: int, n_treat: int, d_eps: int = 16):
        super().__init__()
        self.d_c = d_c
        self.n_treat = n_treat
        self.d_eps = d_eps

        # Abduction network: q(eps | z_c, a) -> mu, logvar
        self.abduction_net = nn.Sequential(
            nn.Linear(d_c + n_treat, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * d_eps)
        )

        # Counterfactual modification network: z_c' = f(z_c, a', eps)
        self.counterfactual_net = nn.Sequential(
            nn.Linear(d_c + n_treat + d_eps, 64),
            nn.ReLU(),
            nn.Linear(64, d_c)
        )

    def abduct_noise(self, z_c: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Abduces mean and logvar of individual background noise."""
        inp = torch.cat([z_c, a], dim=-1)
        out = self.abduction_net(inp)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar

    def apply_intervention(self, z_c: torch.Tensor, a_new: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Applies surgical intervention do(A_t = a') preserving abducted noise eps."""
        inp = torch.cat([z_c, a_new, eps], dim=-1)
        delta_zc = self.counterfactual_net(inp)
        return z_c + delta_zc


def simulate_counterfactual_intervention(
    model: nn.Module,
    dataset: Any,
    patient_idx: int,
    visit_idx: int,
    intervention_dict: dict[str, float],
    mc_samples: int = 25,
    device: torch.device = torch.device("cpu")
) -> dict[str, float]:
    """Executes the counterfactual 'what-if' simulation for a clinical visit.
    
    Args:
        model: Trained causal deep model
        dataset: SequenceDataset containing patient records
        patient_idx: Index of the patient
        visit_idx: Time-step t at which intervention is applied
        intervention_dict: Map of feature names to hypothetical treatment values
        mc_samples: Number of Monte Carlo samples for epistemic uncertainty
        device: Hardware device
        
    Returns:
        Dictionary with factual risk, counterfactual risk, risk delta, and uncertainty.
    """
    model.eval()
    t = visit_idx
    pi = patient_idx

    # Extract single patient sequence
    X = torch.as_tensor(dataset.X[pi:pi+1], dtype=torch.float32, device=device)
    M = torch.as_tensor(dataset.M[pi:pi+1], dtype=torch.float32, device=device)
    gap = torch.as_tensor(dataset.gap[pi:pi+1], dtype=torch.float32, device=device)
    pad = torch.as_tensor(dataset.pad[pi:pi+1], dtype=torch.bool, device=device)

    # 1. Evaluate factual outcome under actual treatment
    with torch.no_grad():
        out_fact = model(X, M, gap, pad)
        p_fact = torch.sigmoid(out_fact["flare"])[0, t].item()
        sev_fact = model.sev_mean(out_fact)[0, t].item()

    # 2. Construct counterfactual input tensor X_cf
    X_cf = X.clone()
    for feat_name, val in intervention_dict.items():
        if feat_name in dataset.names:
            idx = dataset.names.index(feat_name)
            if hasattr(dataset, "scaler") and idx < len(dataset.scaler["mean"]):
                m = dataset.scaler["mean"][idx]
                s = dataset.scaler["scale"][idx]
                scaled_val = (val - m) / (s + 1e-8)
            else:
                scaled_val = val
            X_cf[0, t, idx] = scaled_val

    # 3. Sample counterfactual outcomes with MC Dropout
    cf_flare_samples = []
    cf_sev_samples = []
    model.train()  # Enable MC dropout
    with torch.no_grad():
        for _ in range(mc_samples):
            out_cf = model(X_cf, M, gap, pad)
            p_cf = torch.sigmoid(out_cf["flare"])[0, t].item()
            s_cf = model.sev_mean(out_cf)[0, t].item()
            cf_flare_samples.append(p_cf)
            cf_sev_samples.append(s_cf)

    model.eval()

    cf_flare_arr = np.array(cf_flare_samples)
    cf_sev_arr = np.array(cf_sev_samples)

    p_counterfactual = float(np.mean(cf_flare_arr))
    delta_risk = p_counterfactual - p_fact

    return {
        "p_factual": p_fact,
        "p_counterfactual": p_counterfactual,
        "flare_risk_delta": delta_risk,
        "flare_risk_delta_sd": float(np.std(cf_flare_arr)),
        "sev_factual": sev_fact,
        "sev_counterfactual": float(np.mean(cf_sev_arr)),
        "sev_delta": float(np.mean(cf_sev_arr)) - sev_fact,
    }
