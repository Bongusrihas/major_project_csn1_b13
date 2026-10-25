"""Configuration dataclass for Causal Representation Learning and LECG Extension.

Defines hyperparameters for:
1. Structural Graph Encoder (R-GCN) & LECG modification.
2. Multi-Resolution Temporal Fusion (MRTF).
3. Temporal Causal Transformer (TCT).
4. Counterfactual Simulation Engine (CSE).
5. Causal Disentanglement and Invariance (CDI).
6. Dual Epistemic & Aleatoric Uncertainty Quantification (UQ).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CausalCfg:
    """Hyperparameters for the Causal Deep Framework and LECG Extension."""
    # Latent dimensions
    d: int = 32                 # Node embedding dimension inside R-GCN
    d_c: int = 32               # Causal invariant latent dimension (z_causal)
    d_s: int = 16               # Spurious latent dimension (z_spurious)
    d_h: int = 64               # Temporal transformer hidden dimension
    n_heads: int = 4            # Multi-head attention heads in TCT
    n_layers: int = 2           # Transformer encoder layers in TCT
    scales: tuple = (1, 3, 6)   # Temporal pooling scales for MRTF (fine to coarse)
    dropout: float = 0.15       # Dropout probability (used in MC Dropout epistemic UQ)
    k_mix: int = 3              # Number of Gaussian mixture components (aleatoric UQ)
    d_eps: int = 16             # Latent noise abduction dimension in CSE

    # -------------------------------------------------------------------------
    # Major Project Architectural Modification:
    # Learnable Edge-Confidence Gating (LECG)
    # -------------------------------------------------------------------------
    use_lecg: bool = True       # If True, enables confidence-weighted graph messages
    init_edge_confidence: float = 2.2 # Logit ~2.2 -> sigmoid(2.2) ~ 0.90 initial confidence

    # Optional regularized LECG candidate; disabled for legacy checkpoints.
    edge_dropout: float = 0.0
    edge_sparsity: float = 0.0

    # Ablation toggles (Paper Section IV.G, Fig 13)
    use_graph: bool = True      # Structural graph encoder enabled
    use_cdi: bool = True        # Causal disentanglement & invariance penalties (HSIC + MMD)
    use_cse: bool = True        # Counterfactual simulation engine (abduction + consistency)
    use_gate: bool = True       # Time-aware intervention gating in TCT
    use_tct: bool = True        # Temporal causal transformer (vs GRU / flat baseline)
    use_mog: bool = True        # Mixture of Gaussians head for aleatoric UQ
