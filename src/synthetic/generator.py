"""Random Disease Cohort Generator and Multi-Dataset Benchmarking Engine.

Provides:
1. Procedural generation of synthetic longitudinal disease cohorts with custom:
   - Disease pathology and biomarker dynamics (e.g., SLE, RA, IBD, Psoriasis, MS).
   - Injected spurious/corrupted causal graph edges (controlled misspecification).
   - Dynamic clinical noise, irregular visit schedules, and observational confounding.
2. Multi-Dataset Stress-Testing:
   - Takes N as input (number of datasets/trials to generate).
   - Trains/evaluates both the Original Paper Model (static R-GCN) and the
     Proposed temporal residual boosting improvement.
   - Outputs side-by-side comparative metrics (AUROC, AUPRC, F1, MAE) across all N datasets.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import torch

from src.causal import dataset as D
from src.causal_graph.dag_builder import (
    build_prior_graph,
    build_relation_tensor,
    synthesize_causal_dag,
)


DISEASE_PROFILES = [
    {
        "name": "Systemic Lupus Erythematosus (SLE)",
        "organs": ["Kidney (Nephritis)", "Skin (Malar Rash)", "Hematologic"],
        "hallmarks": ["Anti-dsDNA", "C3/C4 Complement", "IL-6"],
        "treatment_efficacy": 0.85
    },
    {
        "name": "Rheumatoid Arthritis (RA)",
        "organs": ["Synovial Joints", "Bone Erosions"],
        "hallmarks": ["ESR", "CRP", "RF", "TNF-alpha"],
        "treatment_efficacy": 0.78
    },
    {
        "name": "Inflammatory Bowel Disease (IBD - Crohn's)",
        "organs": ["Intestinal Mucosa", "Mesenteric Nodes"],
        "hallmarks": ["Calprotectin", "CRP", "IL-1beta"],
        "treatment_efficacy": 0.72
    },
    {
        "name": "Systemic Sclerosis (SSc)",
        "organs": ["Vascular Endothelium", "Dermal Fibrosis", "Pulmonary"],
        "hallmarks": ["Anti-Scl70", "ANA", "Endothelin-1"],
        "treatment_efficacy": 0.65
    },
    {
        "name": "Psoriatic Arthritis (PsA)",
        "organs": ["Dactylitis", "Enthesitis", "Cutaneous Plaques"],
        "hallmarks": ["IL-17", "IL-23", "CRP"],
        "treatment_efficacy": 0.82
    }
]


def generate_single_disease_dataset(
    dataset_id: int,
    n_patients: int = 120,
    n_visits: int = 6,
    noise_ratio: float = 0.15,
    seed: int = 42
) -> tuple[D.SequenceDataset, np.ndarray, list[tuple[int, int, int]], dict[str, Any]]:
    """Generates a single synthetic longitudinal disease dataset with controlled edge noise.
    
    Returns:
        sd: Prepared SequenceDataset ready for model input.
        A_noisy: Adjacency tensor with injected noisy edges.
        noisy_coords: Coordinates of injected spurious edges.
        meta: Dataset metadata (disease name, patient count, flare rate).
    """
    if n_patients < 12 or n_visits < 2 or not 0 <= noise_ratio <= 0.5:
        raise ValueError("Use at least 12 patients, 2 visits, and noise_ratio in [0, 0.5].")
    rng = np.random.default_rng(seed)
    profile = DISEASE_PROFILES[dataset_id % len(DISEASE_PROFILES)]

    # Standard feature vocabulary
    names = [
        "crp", "esr", "wbc", "platelets", "creatinine",
        "biomarker_alpha", "biomarker_beta", "autoantibody_titre", "complement_activity",
        "heart_rate", "sbp", "temperature",
        "symptom_fatigue", "symptom_pain", "organ_manifestation",
        "steroid_dose", "immunosuppressive_agent", "targeted_biologic",
    ]
    roles = [
        "lab", "lab", "lab", "lab", "lab",
        "immuno", "immuno", "immuno", "immuno",
        "vital", "vital", "vital",
        "symptom", "symptom", "symptom",
        "treatment", "treatment", "treatment"
    ]
    treat_idx = [names.index(n) for n, r in zip(names, roles) if r == "treatment"]
    V = len(names)

    # 1. Synthesize patient longitudinal trajectories
    raw = np.zeros((n_patients, n_visits, V), dtype=np.float32)
    gap = np.zeros((n_patients, n_visits), dtype=np.float32)
    y_flare = np.zeros((n_patients, n_visits), dtype=np.float32)
    y_sev = np.zeros((n_patients, n_visits), dtype=np.float32)
    valid = np.ones((n_patients, n_visits), dtype=bool)
    n_vis = np.full(n_patients, n_visits, dtype=int)

    for i in range(n_patients):
        latent_u = rng.normal(0.0, 0.5)
        for t in range(n_visits):
            # Time gap between visits (30 to 90 days)
            gap[i, t] = rng.uniform(28.0, 95.0) if t > 0 else 0.0

            # Treatments (doctors intensify treatments if latent disease U is high)
            steroid = rng.choice([0.0, 5.0, 10.0, 20.0], p=[0.4, 0.3, 0.2, 0.1] if latent_u < 0.2 else [0.1, 0.2, 0.4, 0.3])
            immuno = float(rng.random() < (0.3 + 0.3 * (latent_u > 0)))
            biologic = float(rng.random() < (0.2 + 0.4 * (latent_u > 0.5)))

            raw[i, t, names.index("steroid_dose")] = steroid
            raw[i, t, names.index("immunosuppressive_agent")] = immuno
            raw[i, t, names.index("targeted_biologic")] = biologic

            # Treatment suppression on latent state
            treatment_effect = (0.04 * steroid + 0.5 * immuno + 0.9 * biologic) * profile["treatment_efficacy"]
            latent_u = 0.7 * latent_u - treatment_effect + rng.normal(0.2, 0.4)

            # Biomarkers driven by latent U
            raw[i, t, names.index("crp")] = np.clip(np.exp(latent_u + rng.normal(1.0, 0.4)), 0.5, 120.0)
            raw[i, t, names.index("esr")] = np.clip(np.exp(latent_u + rng.normal(2.0, 0.3)), 5.0, 100.0)
            raw[i, t, names.index("wbc")] = np.clip(5.0 + 2.0 * latent_u + rng.normal(0, 1.0), 2.0, 18.0)
            raw[i, t, names.index("platelets")] = np.clip(250.0 + 40.0 * latent_u + rng.normal(0, 30.0), 100.0, 500.0)
            raw[i, t, names.index("creatinine")] = np.clip(0.8 + 0.3 * max(0.0, latent_u) + rng.normal(0, 0.1), 0.4, 3.5)

            raw[i, t, names.index("biomarker_alpha")] = np.clip(latent_u + rng.normal(2.0, 0.5), 0.0, 10.0)
            raw[i, t, names.index("biomarker_beta")] = np.clip(latent_u * 1.5 + rng.normal(1.0, 0.5), 0.0, 10.0)
            raw[i, t, names.index("autoantibody_titre")] = np.clip(np.exp(latent_u + 1.2), 0.0, 50.0)
            raw[i, t, names.index("complement_activity")] = np.clip(1.2 - 0.4 * latent_u + rng.normal(0, 0.1), 0.2, 2.0)

            raw[i, t, names.index("heart_rate")] = 72.0 + 10.0 * max(0.0, latent_u) + rng.normal(0, 5.0)
            raw[i, t, names.index("sbp")] = 120.0 + 8.0 * (steroid / 10.0) + rng.normal(0, 6.0)
            raw[i, t, names.index("temperature")] = 36.6 + 0.5 * max(0.0, latent_u) + rng.normal(0, 0.2)

            raw[i, t, names.index("symptom_fatigue")] = np.clip(latent_u * 2.0 + rng.normal(3.0, 1.0), 0.0, 10.0)
            raw[i, t, names.index("symptom_pain")] = np.clip(latent_u * 2.5 + rng.normal(2.5, 1.2), 0.0, 10.0)
            raw[i, t, names.index("organ_manifestation")] = float(latent_u > 0.4)

            # Outcomes for NEXT visit (t+1)
            next_u = latent_u + rng.normal(0, 0.3)
            p_flare = 1.0 / (1.0 + np.exp(-1.8 * (next_u - 0.3)))
            y_flare[i, t] = float(rng.random() < p_flare)
            y_sev[i, t] = np.clip(int(round(np.clip(next_u + 1.5, 0, 4))), 0, 4)

    y_flare[:, :-1] = y_flare[:, 1:]
    y_sev[:, :-1] = y_sev[:, 1:]
    valid[:, -1] = False
    gap = np.log1p(gap / 30.0).astype(np.float32)

    # Missingness mask (simulate 5% missingness)
    M = (rng.random(raw.shape) > 0.05).astype(np.float32)
    # Preliminary scaling is replaced with training-only scaling in the benchmark.
    mean_v = np.nanmean(raw, axis=(0, 1))
    std_v = np.nanstd(raw, axis=(0, 1)) + 1e-6
    X = np.where(M > 0, (raw - mean_v) / std_v, 0).astype(np.float32)

    patient_ids = [f"SYN-{profile['name'][:3].upper()}-{i+1:03d}" for i in range(n_patients)]
    nuisance = rng.normal(0, 1, (n_patients, 2)).astype(np.float32)
    subgroup = rng.integers(0, 2, n_patients).astype(int)

    sd = D.SequenceDataset(
        name=f"Dataset_{dataset_id+1}_{profile['name']}",
        X=X,
        M=M,
        raw=raw,
        gap=gap,
        n_vis=n_vis,
        y_flare=y_flare,
        y_sev=y_sev,
        valid=valid,
        nuisance=nuisance,
        subgroup=subgroup,
        names=names,
        roles=roles,
        treat_idx=treat_idx,
        patient_ids=patient_ids
    )

    # 2. Build Causal DAG
    g_prior = build_prior_graph(names, roles)
    adj_clean = synthesize_causal_dag(g_prior, g_prior, lam=0.6)
    A_clean = build_relation_tensor(adj_clean, names, roles)

    # 3. Inject Spurious/Noisy Edges based on noise_ratio
    A_noisy = A_clean.copy()
    n_rel, V_tot, _ = A_clean.shape
    total_edges = int(np.sum(A_clean > 0))
    k_noise = int(total_edges * noise_ratio)

    noisy_coords = []
    cand_nodes = list(range(V_tot - 1))
    attempts = 0
    while len(noisy_coords) < k_noise and attempts < 1000:
        attempts += 1
        r = int(rng.integers(0, n_rel))
        u = int(rng.choice(cand_nodes))
        v = int(rng.choice(cand_nodes))
        if u != v and A_noisy[r, v, u] == 0:
            A_noisy[r, v, u] = 1.0
            noisy_coords.append((r, v, u))

    meta = {
        "dataset_id": dataset_id + 1,
        "disease_name": profile["name"],
        "n_patients": n_patients,
        "n_visits": n_visits,
        "flare_rate": float(np.mean(y_flare)),
        "clean_edges": total_edges,
        "injected_noise_edges": len(noisy_coords),
        "noise_ratio": noise_ratio
    }

    return sd, A_noisy, noisy_coords, meta


def run_multi_dataset_benchmark(
    n_datasets=3, n_patients=120, n_visits=6, noise_ratio=.2,
    base_seed=42, device=torch.device("cpu"), epochs=30, progress_callback=None,
):
    """Compare the paper implementation and our single residual improvement."""
    import tempfile
    from pathlib import Path
    from src.project.pair import train_pair
    if n_datasets < 1 or epochs < 1:
        raise ValueError("n_datasets and epochs must be positive")
    records, protocol, predictions = [], [], []
    for index in range(n_datasets):
        seed = base_seed + 17 * index
        sd, adjacency, _, meta = generate_single_disease_dataset(index, n_patients, n_visits, noise_ratio, seed)
        order = np.random.default_rng(seed).permutation(sd.N)
        sd.splits = dict(zip(("train", "val", "test"), np.split(order, [int(.6 * sd.N), int(.8 * sd.N)])))
        sd.scaler = D.fit_scaler(np.where(sd.M > 0, sd.raw, np.nan), sd.n_vis, sd.names, sd.roles, set(), sd.splits["train"])
        normalized, _ = D.apply_scaler(sd.raw, sd.names, sd.scaler)
        sd.X = np.where(sd.M > 0, normalized, 0).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix="two-method-test-") as tmp:
            result = train_pair(sd, tmp, epochs=epochs, seed=seed, adjacency=adjacency,
                                device=device, progress=lambda message: None)
            frame = pd.read_csv(Path(tmp) / "predictions.csv")
        row = {"dataset_index": index + 1, "disease_name": meta["disease_name"],
               "orig_epochs": len(result["history"]), "methods": "paper,improved"}
        for source, prefix in (("paper", "orig"), ("improved", "improved")):
            row.update({f"{prefix}_{key}": value for key, value in result["metrics"][source].items()})
        row["delta_auroc"] = result["delta_auroc"]
        records.append(row)
        protocol.append({f"{key}_patients": value.tolist() for key, value in sd.splits.items()})
        predictions.append(frame.to_dict(orient="list"))
        if progress_callback:
            progress_callback((index + 1) / n_datasets, f"Dataset {index + 1}: evaluated both methods")
    table = pd.DataFrame(records)
    return {"table": table, "summary": {"n_datasets": n_datasets,
        "mean_orig_auroc": float(table.orig_auroc.mean()),
        "mean_improved_auroc": float(table.improved_auroc.mean())},
        "protocol": protocol, "predictions": predictions}


if __name__ == "__main__":
    from src.project.pipeline import main
    main()
