"""Glue: build the causal graph from TRAIN patients and wrap everything for the scripts / app."""
from __future__ import annotations

import numpy as np

from src.causal import graph as G


def build_graph(sd, train_idx, lam=0.6, keep=0.8):
    """Returns (W fused adjacency, A relation tensor, info dict). Uses only the training patients."""
    Xz = np.where(sd.M > 0, sd.X, np.nan)[train_idx]
    P = G.prior_graph(sd.names, sd.roles)
    cov = None
    S, R = G.stat_graph(Xz, sd.n_vis[train_idx], sd.names, sd.roles, covariates=cov)
    W = G.fuse(P, S, lam=lam, keep=keep)
    A = G.relation_tensor(W, sd.names, sd.roles)
    info = dict(P=P, S=S, R=R, W=W, acyclic=G.is_acyclic(W), n_edges=int((W > 0).sum()))
    return W, A, info
