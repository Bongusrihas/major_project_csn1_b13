"""Proposed Major Project Model Architecture: LECG-Enhanced Causal Deep Model.

Project Title:
    "Improving Autoimmune Disease Progression Prediction Using
     Confidence-Weighted Causal Graph Learning"

Architecture Innovation:
    1. Replaces static R-GCN with Learnable Edge-Confidence Gating (LECG).
    2. Learns a scalar confidence logit theta_e for every directed edge in the causal DAG:
           w_e = sigmoid(theta_e) in (0, 1)
    3. Gated Message Passing:
           m_{u -> v} = w_e * W_r * h_u
    4. Automatically attenuates spurious, noisy, or misspecified clinical relationships (w_e -> 0)
       while reinforcing invariant causal pathways (w_e -> 1).
"""
from __future__ import annotations

import copy
import numpy as np
import torch

from src.models.causal_deep_model import CausalDeepModel
from src.models.config import CausalCfg


class LECGImprovedCausalModel(CausalDeepModel):
    """Our proposed enhanced architecture with Learnable Edge-Confidence Gating.
    
    Dynamically learns edge reliability weights inside the structural graph encoder (use_lecg=True).
    """

    def __init__(
        self,
        V: int,
        treat_idx: list[int],
        A: np.ndarray | torch.Tensor,
        cfg: CausalCfg | None = None,
        n_nuisance: int = 1
    ):
        model_cfg = copy.deepcopy(cfg) if cfg is not None else CausalCfg()
        model_cfg.use_lecg = True  # Enforce proposed confidence-weighted gating
        super().__init__(V=V, treat_idx=treat_idx, A=A, cfg=model_cfg, n_nuisance=n_nuisance)
