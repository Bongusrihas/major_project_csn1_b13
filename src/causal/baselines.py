"""Baseline models for comparative evaluation (Paper Section IV.B, Table 2, Table 3, Table 4).

Implements:
1. LSTM baseline: Standard recurrent deep learning model.
2. Vanilla Transformer: Standard multi-head self-attention without DAG constraints or causal modules.
3. Variational RNN (VRNN): Recurrent neural network with latent stochastic transitions.
4. Clinical Risk Baselines: SLEDAI and DAS28 scoring algorithms.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, mean_absolute_error, root_mean_squared_error


# -----------------------------------------------------------------------------
# 1. LSTM Baseline (Paper Table 2 & Table 3)
# -----------------------------------------------------------------------------
class LSTMBaseline(nn.Module):
    """Multi-layer LSTM model for longitudinal sequence modeling."""

    def __init__(self, d_in: int, d_h: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.d_h = d_h
        self.input_proj = nn.Linear(d_in, d_h)
        self.lstm = nn.LSTM(
            input_size=d_h,
            hidden_size=d_h,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.flare_head = nn.Linear(d_h, 1)
        self.sev_head = nn.Linear(d_h, 1)

    def forward(self, x: torch.Tensor, pad: torch.Tensor | None = None):
        """x: (B, T, D), pad: (B, T) boolean mask."""
        h0 = self.input_proj(x)
        out, _ = self.lstm(h0)
        out = self.dropout(out)
        flare_logit = self.flare_head(out).squeeze(-1)
        sev_pred = self.sev_head(out).squeeze(-1)
        return {"flare": flare_logit, "sev": sev_pred, "h": out}


# -----------------------------------------------------------------------------
# 2. Vanilla Transformer Baseline (Paper Table 2 & Table 3)
# -----------------------------------------------------------------------------
class VanillaTransformerBaseline(nn.Module):
    """Standard Transformer encoder without causal DAG masking or intervention gating."""

    def __init__(self, d_in: int, d_h: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.d_h = d_h
        self.input_proj = nn.Linear(d_in, d_h)
        self.pos_emb = nn.Embedding(128, d_h)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_h,
            nhead=n_heads,
            dim_feedforward=2 * d_h,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.flare_head = nn.Linear(d_h, 1)
        self.sev_head = nn.Linear(d_h, 1)

    def forward(self, x: torch.Tensor, pad: torch.Tensor | None = None):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device)[None, :]
        h = self.input_proj(x) + self.pos_emb(pos)

        # Standard non-causal or simple padding mask
        src_key_padding_mask = ~pad if pad is not None else None
        out = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        flare_logit = self.flare_head(out).squeeze(-1)
        sev_pred = self.sev_head(out).squeeze(-1)
        return {"flare": flare_logit, "sev": sev_pred, "h": out}


# -----------------------------------------------------------------------------
# 3. Variational RNN (VRNN) Baseline (Paper Table 2 & Table 3)
# -----------------------------------------------------------------------------
class VariationalRNNBaseline(nn.Module):
    """Variational Recurrent Neural Network (VRNN) with latent stochastic variables."""

    def __init__(self, d_in: int, d_h: int = 48, d_z: int = 16, dropout: float = 0.2):
        super().__init__()
        self.d_h, self.d_z = d_h, d_z
        # Feature extractors
        self.phi_x = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU())
        self.phi_z = nn.Sequential(nn.Linear(d_z, d_h), nn.ReLU())

        # Prior p(z_t | h_{t-1})
        self.prior = nn.Sequential(nn.Linear(d_h, d_h), nn.ReLU(), nn.Linear(d_h, 2 * d_z))
        # Posterior q(z_t | x_t, h_{t-1})
        self.enc = nn.Sequential(nn.Linear(2 * d_h, d_h), nn.ReLU(), nn.Linear(d_h, 2 * d_z))

        # Recurrent cell
        self.rnn = nn.GRUCell(2 * d_h, d_h)

        # Output prediction heads
        self.flare_head = nn.Linear(d_h + d_z, 1)
        self.sev_head = nn.Linear(d_h + d_z, 1)

    def forward(self, x: torch.Tensor, pad: torch.Tensor | None = None):
        B, T, D = x.shape
        device = x.device
        h_t = torch.zeros(B, self.d_h, device=device)

        flare_logits = []
        sev_preds = []
        kl_losses = []

        for t in range(T):
            x_t = x[:, t]
            phi_x_t = self.phi_x(x_t)

            # Prior
            prior_params = self.prior(h_t)
            mu_p, lv_p = prior_params.chunk(2, dim=-1)

            # Posterior
            enc_input = torch.cat([phi_x_t, h_t], dim=-1)
            enc_params = self.enc(enc_input)
            mu_q, lv_q = enc_params.chunk(2, dim=-1)

            # Reparameterization
            std_q = torch.exp(0.5 * lv_q)
            eps = torch.randn_like(std_q)
            z_t = mu_q + eps * std_q

            # KL divergence
            kl = 0.5 * (lv_p - lv_q + (torch.exp(lv_q) + (mu_q - mu_p).pow(2)) / torch.exp(lv_p) - 1.0).sum(-1)
            kl_losses.append(kl)

            phi_z_t = self.phi_z(z_t)
            h_t = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=-1), h_t)

            # Prediction
            pred_feat = torch.cat([h_t, z_t], dim=-1)
            flare_logits.append(self.flare_head(pred_feat).squeeze(-1))
            sev_preds.append(self.sev_head(pred_feat).squeeze(-1))

        fl_out = torch.stack(flare_logits, dim=1)
        sv_out = torch.stack(sev_preds, dim=1)
        kl_out = torch.stack(kl_losses, dim=1)
        return {"flare": fl_out, "sev": sv_out, "kl": kl_out}


# -----------------------------------------------------------------------------
# 4. Clinical Baseline Scores (Paper Table 4)
# -----------------------------------------------------------------------------
def evaluate_clinical_baselines(raw_df: pd.DataFrame) -> dict:
    """Evaluate SLEDAI and DAS28 clinical baseline risk scores.

    SLEDAI: Systemic Lupus Erythematosus Disease Activity Index.
    DAS28: Disease Activity Score in 28 joints.
    """
    results = {}

    # Valid rows with next visit outcomes
    valid = raw_df["next_flare_event"].notna() & raw_df["next_severity_score"].notna()
    eval_df = raw_df[valid].copy()

    y_flare = eval_df["next_flare_event"].to_numpy().astype(int)
    y_sev = eval_df["next_severity_score"].to_numpy().astype(float)

    # 1. SLEDAI-based baseline
    # Flare rule: current severity / activity high -> predicted probability proportional to sigmoid(sledai - threshold)
    # Severity rule: persistence of current severity
    sledai_proxy = eval_df["severity_score"].to_numpy().astype(float)
    # Probability estimate calibrated to prevalence
    p_sledai = 1.0 / (1.0 + np.exp(-1.2 * (sledai_proxy - 2.0)))
    pred_sev_sledai = sledai_proxy

    auroc_sledai = float(roc_auc_score(y_flare, p_sledai))
    auprc_sledai = float(average_precision_score(y_flare, p_sledai))
    f1_sledai = float(f1_score(y_flare, (p_sledai > 0.35).astype(int), zero_division=0))
    mae_sledai = float(mean_absolute_error(y_sev, pred_sev_sledai))
    rmse_sledai = float(root_mean_squared_error(y_sev, pred_sev_sledai))

    results["SLEDAI"] = {
        "auroc": round(auroc_sledai, 3),
        "auprc": round(auprc_sledai, 3),
        "f1": round(f1_sledai, 3),
        "mae": round(mae_sledai, 2),
        "rmse": round(rmse_sledai, 2)
    }

    # 2. DAS28-based baseline (weighted ESR/CRP + swollen/tender joints)
    das28_proxy = (
        0.56 * np.sqrt(eval_df["joint_tenderness"].clip(lower=0))
        + 0.28 * np.sqrt(eval_df["joint_swelling"].clip(lower=0))
        + 0.70 * np.log(np.clip(eval_df["esr"], 1.0, 100.0))
        + 0.014 * eval_df["fatigue_score"] * 10.0
    )
    p_das28 = 1.0 / (1.0 + np.exp(-1.1 * (das28_proxy - das28_proxy.mean()) / (das28_proxy.std() + 1e-6)))
    pred_sev_das28 = np.clip(np.round(das28_proxy / 2.0), 0, 4)

    auroc_das28 = float(roc_auc_score(y_flare, p_das28))
    auprc_das28 = float(average_precision_score(y_flare, p_das28))
    f1_das28 = float(f1_score(y_flare, (p_das28 > 0.35).astype(int), zero_division=0))
    mae_das28 = float(mean_absolute_error(y_sev, pred_sev_das28))
    rmse_das28 = float(root_mean_squared_error(y_sev, pred_sev_das28))

    results["DAS28"] = {
        "auroc": round(auroc_das28, 3),
        "auprc": round(auprc_das28, 3),
        "f1": round(f1_das28, 3),
        "mae": round(mae_das28, 2),
        "rmse": round(rmse_das28, 2)
    }

    return results

