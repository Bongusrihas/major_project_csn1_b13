"""Causal Graph Construction (Paper Section III.C, Eqs. 3-4).

Implements the hybrid data-driven and knowledge-informed causal graph pipeline:
1. Prior Knowledge Encoding (G_prior): Directed edges initialized from rheumatology literature
   and autoimmune ontologies (UMLS, MeSH).
2. Conditional Independence Testing (G_stat): Prunes or confirms edges using Markov blanket
   conditioning and kernel-based independence tests (PC-algorithm / HSIC / partial correlation) (Eq. 3).
3. BuildDAG Synthesis: Combines prior graph G_prior and data-driven graph G_stat through a
   weighted edge agreement function:
       G = lambda * G_prior + (1 - lambda) * G_stat,   lambda in [0, 1]   (Eq. 4).
4. Verifies DAG acyclicity and constructs R-GCN relation tensors (Eq. 6).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

ORDER = {
    "treatment": 0,
    "latent": 1,
    "lab": 2,
    "immuno": 2,
    "vital": 2,
    "symptom": 3,
    "outcome": 4
}
OBS_ROLES = ("lab", "immuno", "vital")

# Relation types for the Relational Graph Convolutional Network (R-GCN)
# Each forward relation has an inverse relation (messages flow bi-directionally)
REL_DEFS = [
    ("treatment", "latent"),
    ("treatment", "obs"),
    ("treatment", "symout"),
    ("latent", "obs"),
    ("latent", "symout"),
    ("obs", "symout"),
    ("symptom", "outcome")
]
N_REL = 2 * len(REL_DEFS)


def _group_role(role: str) -> str:
    if role in OBS_ROLES:
        return "obs"
    if role in ("symptom", "outcome"):
        return "symout"
    return role


# -----------------------------------------------------------------------------
# 1. Prior Knowledge Encoding (Paper Section III.C.1)
# -----------------------------------------------------------------------------
PRIOR_PHARMACOLOGICAL_RULES = {
    "corticosteroid": ["crp", "esr", "il6", "wbc", "platelet_count", "sbp", "dbp", "temperature", "rash_flag", "fatigue_score"],
    "steroid": ["crp", "esr", "il6", "wbc", "platelet_count", "sbp", "fatigue"],
    "dmard_mtx": ["crp", "esr", "joint_swelling", "joint_tenderness", "morning_stiffness", "ast", "alt"],
    "dmard_leflunomide": ["crp", "esr", "joint_swelling", "joint_tenderness", "alt"],
    "dmard": ["crp", "esr", "joint_swelling", "joint_tenderness"],
    "biologic_anti_tnf": ["tnfa", "crp", "esr", "joint_swelling", "joint_tenderness", "morning_stiffness"],
    "biologic_rituximab": ["dsdna", "igg", "igm", "cd4cd8_ratio", "crp", "esr"],
    "biologic": ["tnfa", "il6", "joint_swelling"],
    "hydroxychloroquine": ["dsdna", "ana", "c3", "c4", "rash_flag"],
    "nsaid": ["joint_swelling", "joint_tenderness", "morning_stiffness", "fatigue_score", "temperature"],
    "azathioprine": ["wbc", "platelet_count", "dsdna", "creatinine"],
}

BIOMARKER_OUTCOME_RULES = {
    "crp": ["severity_score", "flare_event", "joint_swelling"],
    "esr": ["severity_score", "flare_event", "joint_swelling"],
    "dsdna": ["severity_score", "flare_event", "creatinine", "rash_flag"],
    "c3": ["severity_score", "flare_event"],
    "c4": ["severity_score", "flare_event"],
    "il6": ["severity_score", "flare_event", "fatigue_score", "temperature"],
    "tnfa": ["severity_score", "flare_event", "joint_swelling"],
    "joint_swelling": ["severity_score", "flare_event"],
    "joint_tenderness": ["severity_score", "flare_event"],
    "morning_stiffness": ["severity_score", "flare_event"],
    "rash_flag": ["severity_score", "flare_event"],
    "fatigue_score": ["severity_score", "flare_event"],
}


def build_prior_graph(names: list[str], roles: list[str]) -> np.ndarray:
    """Construct prior knowledge graph G_prior over V+1 nodes (index V = latent immune state U)."""
    V = len(names)
    P = np.zeros((V + 1, V + 1), dtype=np.float32)
    lat_idx = V  # Latent node

    for i, (name, role) in enumerate(zip(names, roles)):
        if role == "treatment":
            # Treatments act on the latent immune state
            P[i, lat_idx] = 1.0
            # Check specific prior pharmacological rules
            for rule_key, targets in PRIOR_PHARMACOLOGICAL_RULES.items():
                if rule_key in name:
                    for j, target_name in enumerate(names):
                        if target_name in targets or any(target_name.startswith(t) for t in targets):
                            P[i, j] = 1.0

        elif role in OBS_ROLES or role in ("symptom", "outcome"):
            # Latent immune dysregulation drives biomarkers and symptoms
            P[lat_idx, i] = 1.0

        # Biomarker -> Symptom / Outcome rules
        if name in BIOMARKER_OUTCOME_RULES:
            for j, target_name in enumerate(names):
                if target_name in BIOMARKER_OUTCOME_RULES[name]:
                    P[i, j] = 1.0

        # General structural constraint: symptoms -> outcomes
        if role == "symptom":
            for j, r_j in enumerate(roles):
                if r_j == "outcome":
                    P[i, j] = 1.0

    return P


# -----------------------------------------------------------------------------
# 2. Conditional Independence Testing (Paper Section III.C.2, Eq. 3)
# -----------------------------------------------------------------------------
def _residualize(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Regress y on conditioning set Z and return residuals."""
    if Z.ndim == 1:
        Z = Z[:, None]
    Z_const = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(Z_const, y, rcond=None)
    return y - Z_const @ beta


def conditional_independence_test(x: np.ndarray, y: np.ndarray, S: np.ndarray) -> tuple[float, float]:
    """Tests X _||_ Y | S via partial correlation / residualized independence (Eq. 3)."""
    res_x = _residualize(x, S)
    res_y = _residualize(y, S)
    r = np.corrcoef(res_x, res_y)[0, 1]
    n = len(x)
    k = S.shape[1] if S.ndim > 1 else 1
    # Fisher z-transformation
    z = np.arctanh(np.clip(r, -0.999999, 0.999999)) * np.sqrt(max(n - k - 3, 1))
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(r), float(p_value)


def build_statistical_graph(
    X_norm: np.ndarray,
    n_vis: np.ndarray,
    names: list[str],
    roles: list[str],
    covariates: np.ndarray | None = None,
    alpha: float = 1e-3,
    r_min: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Data-driven conditional independence testing graph G_stat.

    Treatments act with a temporal lag (A_{t-1} -> X_t).
    Biomarkers condition on the latent immune proxy (first principal component).
    """
    N, T, V = X_norm.shape
    valid_pairs = np.where((np.arange(T)[None, :] < n_vis[:, None]) & (np.arange(T)[None, :] >= 1))
    cur = X_norm[valid_pairs[0], valid_pairs[1]]
    prev = X_norm[valid_pairs[0], valid_pairs[1] - 1]

    # Compute conditioning proxy (Markov blanket latent proxy)
    obs_idx = [i for i, r in enumerate(roles) if r in OBS_ROLES]
    B = cur[:, obs_idx]
    col_means = np.nanmean(B, axis=0)
    B_imputed = np.where(np.isnan(B), col_means, B)
    B_centered = B_imputed - B_imputed.mean(axis=0)

    # First principal component
    _, _, vt = np.linalg.svd(B_centered, full_matrices=False)
    pc1 = B_centered @ vt[0]
    pc1 = (pc1 - pc1.mean()) / (pc1.std() + 1e-8)
    S_conditioning = pc1[:, None]

    G_stat = np.zeros((V + 1, V + 1), dtype=np.float32)
    R_corr = np.zeros((V + 1, V + 1), dtype=np.float32)

    for i, role_i in enumerate(roles):
        for j, role_j in enumerate(roles):
            # Only test causally permissible directions
            if ORDER.get(role_i, 2) >= ORDER.get(role_j, 2) or role_i == "latent":
                continue

            if role_i == "treatment" and role_j in OBS_ROLES + ("symptom",):
                x_val, y_val = prev[:, i], cur[:, j]
            elif role_i in OBS_ROLES and role_j in ("symptom", "outcome"):
                x_val, y_val = cur[:, i], cur[:, j]
            elif role_i == "symptom" and role_j == "outcome":
                x_val, y_val = cur[:, i], cur[:, j]
            else:
                continue

            valid_mask = ~np.isnan(x_val) & ~np.isnan(y_val)
            if valid_mask.sum() < 150:
                continue

            r, p_val = conditional_independence_test(x_val[valid_mask], y_val[valid_mask], S_conditioning[valid_mask])
            R_corr[i, j] = r
            if p_val < alpha and abs(r) > r_min:
                G_stat[i, j] = 1.0

    return G_stat, R_corr


# -----------------------------------------------------------------------------
# 3. BuildDAG Synthesis (Paper Section III.C, Eq. 4)
# -----------------------------------------------------------------------------
def synthesize_causal_dag(
    G_prior: np.ndarray,
    G_stat: np.ndarray,
    lam: float = 0.6,
    threshold: float = 0.35,
    keep: float | None = None
) -> np.ndarray:
    """Eq. 4: G = lambda * G_prior + (1 - lambda) * G_stat, lambda in [0, 1]."""
    thr = keep if keep is not None else threshold
    G = lam * G_prior + (1.0 - lam) * G_stat
    lat_idx = G_prior.shape[0] - 1
    # Latent immune state connections are anchored by prior
    G[lat_idx, :] = G_prior[lat_idx, :]
    G[:, lat_idx] = G_prior[:, lat_idx]
    G[G < thr] = 0.0
    return G


def verify_acyclicity(G: np.ndarray) -> bool:
    """Verify that synthesized graph is a Directed Acyclic Graph (DAG)."""
    adj = (G > 0).astype(int)
    n = len(adj)
    in_degree = adj.sum(axis=0)
    queue = [i for i in range(n) if in_degree[i] == 0]
    visited = 0

    while queue:
        curr = queue.pop(0)
        visited += 1
        for neighbor in np.where(adj[curr])[0]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return visited == n


def build_relation_tensor(G: np.ndarray, names: list[str], roles: list[str]) -> np.ndarray:
    """Construct row-normalized relation message tensor A (n_rel, V+1, V+1) for R-GCN (Eq. 6)."""
    n = G.shape[0]
    all_roles = list(roles) + ["latent"]
    A = np.zeros((N_REL, n, n), dtype=np.float32)

    rel_mapping = {
        ("treatment", "latent"): 0,
        ("treatment", "obs"): 1,
        ("treatment", "symout"): 2,
        ("latent", "obs"): 3,
        ("latent", "symout"): 4,
        ("obs", "symout"): 5,
        ("symptom", "outcome"): 6,
    }

    for s in range(n):
        for d in range(n):
            weight = G[s, d]
            if weight <= 0:
                continue

            grp_s = _group_role(all_roles[s])
            grp_d = _group_role(all_roles[d])

            if all_roles[s] == "symptom" and all_roles[d] == "outcome":
                r_idx = 6
            else:
                r_idx = rel_mapping.get((grp_s, grp_d))

            if r_idx is not None:
                # Forward relation
                A[r_idx, d, s] = weight
                # Inverse relation
                A[len(REL_DEFS) + r_idx, s, d] = weight

    # Degree normalization (1 / c_{v,r})
    degrees = A.sum(axis=2, keepdims=True)
    return A / np.maximum(degrees, 1e-6)


# Backward compatibility aliases
prior_graph = build_prior_graph
stat_graph = build_statistical_graph
fuse = synthesize_causal_dag
is_acyclic = verify_acyclicity
relation_tensor = build_relation_tensor

