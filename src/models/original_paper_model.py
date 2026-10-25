"""Original Paper Causal Model Architecture.

Paper: "Causal Representation Learning for Predicting Autoimmune Disease Progression
        From Longitudinal Multimodal Clinical Data" (IEEE Access, Vol. 13, 2025).

Architecture Summary:
    1. Structural Graph Encoder: Unweighted R-GCN over static prior+stat DAG.
    2. Multi-Resolution Temporal Fusion (MRTF).
    3. Temporal Causal Transformer (TCT).
    4. Counterfactual Simulation Engine (CSE).
    5. Causal Disentanglement & Invariance (CDI).
    6. Uncertainty Quantification (Epistemic UQ via MC Dropout, Aleatoric via MoG).
"""
from __future__ import annotations

import copy
import numpy as np
import torch

from src.models.causal_deep_model import CausalDeepModel
from src.models.config import CausalCfg


class OriginalPaperCausalModel(CausalDeepModel):
    """The exact model architecture published in the IEEE Access 2025 paper.
    
    Uses static, unweighted relational message passing in the R-GCN (use_lecg=False).
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
        model_cfg.use_lecg = False  # Enforce original static paper baseline
        super().__init__(V=V, treat_idx=treat_idx, A=A, cfg=model_cfg, n_nuisance=n_nuisance)
