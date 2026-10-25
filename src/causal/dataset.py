"""Multimodal sequence dataset loaders for Causal Representation Learning.

Supports:
1. Exact 54-dimensional Multimodal Autoimmune Cohort (Paper Section IV.A):
   - 2,000 patients (1,024 RA, 976 SLE), ~33,100 visit instances
   - 54 feature dimensions: Laboratory, Immunological, Demographic/Vitals, Pharmacological
   - Outcomes: Next-visit Flare Event (binary 0/1) & Disease Severity (ordinal 0-4)
   - Min-max normalization within each patient record (Paper Section IV.A)
   - Stratified train (70%) / val (15%) / test (15%) splits
2. Real-world SLE Clinical Cohort (GSE65391):
   - 157 patients, 923 visits
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"


@dataclass
class SequenceDataset:
    name: str
    X: np.ndarray              # (N, T, V) normalized values, 0 where missing
    M: np.ndarray              # (N, T, V) observation masks (1 = observed)
    raw: np.ndarray            # (N, T, V) raw clinical values (NaN = missing)
    gap: np.ndarray            # (N, T) days since previous visit
    n_vis: np.ndarray          # (N,) number of visits per patient
    y_flare: np.ndarray        # (N, T) flare event at visit t+1
    y_sev: np.ndarray          # (N, T) severity score (0-4) at visit t+1
    valid: np.ndarray          # (N, T) boolean mask: next-visit target exists
    nuisance: np.ndarray       # (N, Q) demographic / nuisance variables for CDI
    subgroup: np.ndarray       # (N,) 0/1 subgroup for MMD invariance penalty
    names: list[str]           # Feature names
    roles: list[str]           # Feature clinical roles (lab, immuno, vital, treatment, outcome)
    treat_idx: list[int]       # Indices of treatment features
    patient_ids: list[str]     # Patient IDs
    splits: dict = field(default_factory=dict)
    truth: dict | None = None
    extra: dict = field(default_factory=dict)
    scaler: dict | None = None

    @property
    def V(self) -> int:
        return len(self.names)

    @property
    def N(self) -> int:
        return self.X.shape[0]

    @property
    def T(self) -> int:
        return self.X.shape[1]


def legacy_minmax(raw: np.ndarray, n_vis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility only for old weights trained on full patient extrema.

    This preprocessing sees future visits and must not be used in new experiments.
    """
    X = np.zeros_like(raw, dtype=np.float32)
    M = np.isfinite(raw).astype(np.float32)
    for i, nv in enumerate(n_vis):
        values = raw[i, :nv]
        finite = np.isfinite(values)
        low = np.min(np.where(finite, values, np.inf), axis=0)
        high = np.max(np.where(finite, values, -np.inf), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            span = high - low
            X[i, :nv] = np.nan_to_num((values - low) / np.where(span > 1e-5, span, 1.0))
    return X, M


def prefix_minmax(raw: np.ndarray, n_vis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each visit using only observed values up to that visit."""
    observed = np.isfinite(raw)
    running_min = np.minimum.accumulate(np.where(observed, raw, np.inf), axis=1)
    running_max = np.maximum.accumulate(np.where(observed, raw, -np.inf), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        span = running_max - running_min
        X = (raw - running_min) / np.where(span > 1e-5, span, 1.0)
    present = np.arange(raw.shape[1])[None, :] < np.asarray(n_vis)[:, None]
    mask = observed & present[..., None]
    return np.where(mask, np.nan_to_num(X), 0).astype(np.float32), mask.astype(np.float32)


def load_multimodal_cohort(
    csv_path: Path | str | None = None,
    split_path: Path | str | None = None,
    normalization: str = "train"
) -> SequenceDataset:
    """Load the exact 54-dimensional multimodal longitudinal cohort (Paper Section IV.A)."""
    csv_file = Path(csv_path or (PROC_DIR / "autoimmune_multimodal_54dim.csv"))
    split_file = Path(split_path or (PROC_DIR / "cohort_splits.json"))

    if not csv_file.exists():
        # Automatically generate if not present
        from src.download_data import save_processed_datasets
        print("[Dataset] Dataset not found on disk. Generating exact cohort now ...")
        save_processed_datasets(force=False)

    df = pd.read_csv(csv_file)
    with open(split_file, "r") as f:
        splits = json.load(f)

    # 54 feature dimensions
    meta_file = PROC_DIR / "feature_metadata.json"
    with open(meta_file, "r") as f:
        meta = json.load(f)

    feature_cols = meta["features_54"]
    V = len(feature_cols)

    # Assign roles
    roles = []
    for col in feature_cols:
        if col in meta["categories"]["laboratory"]:
            roles.append("lab")
        elif col in meta["categories"]["immunological"]:
            roles.append("immuno")
        elif col in meta["categories"]["pharmacological"]:
            roles.append("treatment")
        else:
            roles.append("vital")

    treat_idx = [i for i, r in enumerate(roles) if r == "treatment"]

    # Sequence alignment per patient
    patient_ids = df["patient_id"].unique().tolist()
    N = len(patient_ids)
    max_visits = int(df.groupby("patient_id")["visit_index"].max().max() + 1)

    raw = np.full((N, max_visits, V), np.nan, dtype=np.float32)
    gap = np.zeros((N, max_visits), dtype=np.float32)
    y_flare = np.zeros((N, max_visits), dtype=np.float32)
    y_sev = np.zeros((N, max_visits), dtype=np.float32)
    valid = np.zeros((N, max_visits), dtype=bool)
    n_vis = np.zeros(N, dtype=np.int32)
    nuisance = np.zeros((N, 4), dtype=np.float32)  # age_std, female, smoking, comorbidity
    subgroup = np.zeros(N, dtype=np.int32)        # disease_type (RA=0, SLE=1)

    # Fast group-by population
    groups = df.groupby("patient_id", sort=False)
    for i, pid in enumerate(patient_ids):
        pdf = groups.get_group(pid).sort_values("visit_index")
        nv = len(pdf)
        n_vis[i] = nv

        v_indices = pdf["visit_index"].to_numpy().astype(int)
        raw[i, v_indices, :] = pdf[feature_cols].to_numpy(dtype=np.float32)
        gap[i, v_indices] = pdf["gap_days"].to_numpy(dtype=np.float32)

        # Next-visit targets
        has_next = pdf["next_flare_event"].notna().to_numpy()
        valid[i, v_indices[has_next]] = True
        y_flare[i, v_indices[has_next]] = pdf["next_flare_event"].dropna().to_numpy(dtype=np.float32)
        y_sev[i, v_indices[has_next]] = pdf["next_severity_score"].dropna().to_numpy(dtype=np.float32)

        # Nuisance demographics for CDI
        p_first = pdf.iloc[0]
        nuisance[i] = [
            (p_first["age"] - 48.0) / 12.0,
            float(p_first["female"]),
            float(p_first["smoking_status"]),
            float(p_first["comorbidity_index"]) / 3.0
        ]
        subgroup[i] = int(p_first["disease_type"])

    # Convert patient ID splits to index splits
    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}
    split_indices = {
        "train": np.array([pid_to_idx[p] for p in splits["train"] if p in pid_to_idx], dtype=int),
        "val": np.array([pid_to_idx[p] for p in splits["val"] if p in pid_to_idx], dtype=int),
        "test": np.array([pid_to_idx[p] for p in splits["test"] if p in pid_to_idx], dtype=int),
    }

    scaler = None
    if normalization == "train":
        scaler = fit_scaler(raw, n_vis, feature_cols, roles, set(), split_indices["train"])
        X, M = apply_scaler(raw, feature_cols, scaler)
    elif normalization == "prefix":
        X, M = prefix_minmax(raw, n_vis)
    elif normalization == "legacy":
        X, M = legacy_minmax(raw, n_vis)
    else:
        raise ValueError("normalization must be train, prefix or legacy")

    # Transform gap with log1p(gap / 30)
    gap_norm = np.log1p(np.clip(gap, 0, None) / 30.0).astype(np.float32)

    return SequenceDataset(
        name="Autoimmune Multimodal Longitudinal Cohort (54-dim, SLE & RA)",
        X=X,
        M=M,
        raw=raw,
        gap=gap_norm,
        n_vis=n_vis,
        y_flare=y_flare,
        y_sev=y_sev,
        valid=valid,
        nuisance=nuisance,
        subgroup=subgroup,
        names=feature_cols,
        roles=roles,
        treat_idx=treat_idx,
        patient_ids=patient_ids,
        splits=split_indices,
        extra={"meta": meta, "df": df, "normalization": normalization},
        scaler=scaler
    )


def subset_dataset(sd: SequenceDataset, indices: np.ndarray | list[int]) -> SequenceDataset:
    """Extract a patient subset (e.g. train, val, or test split)."""
    idx = np.asarray(indices, dtype=int)
    return SequenceDataset(
        name=f"{sd.name} [subset n={len(idx)}]",
        X=sd.X[idx],
        M=sd.M[idx],
        raw=sd.raw[idx],
        gap=sd.gap[idx],
        n_vis=sd.n_vis[idx],
        y_flare=sd.y_flare[idx],
        y_sev=sd.y_sev[idx],
        valid=sd.valid[idx],
        nuisance=sd.nuisance[idx],
        subgroup=sd.subgroup[idx],
        names=sd.names,
        roles=sd.roles,
        treat_idx=sd.treat_idx,
        patient_ids=[sd.patient_ids[i] for i in idx],
        splits={},
        truth=sd.truth,
        extra=sd.extra,
        scaler=sd.scaler
    )


# -----------------------------------------------------------------------------
# Compatibility Loaders & Aliases
# -----------------------------------------------------------------------------
SeqData = SequenceDataset
subset = subset_dataset


def fit_scaler(raw, n_vis, names, roles, log_vars, train_idx):
    T = raw.shape[1]
    mask = (np.arange(T)[None, :] < n_vis[:, None])
    mask[~np.isin(np.arange(raw.shape[0]), train_idx)] = False
    sc = {}
    for v, (n, r) in enumerate(zip(names, roles)):
        if r == "treatment":
            sc[n] = ("treat", 20.0 if "steroid_dose" in n else 1.0)
            continue
        x = raw[:, :, v][mask]
        x = x[~np.isnan(x)]
        if n in log_vars:
            x = np.log1p(np.clip(x, 0, None))
        mu, sd = (float(x.mean()), float(x.std() + 1e-6)) if len(x) else (0.0, 1.0)
        sc[n] = ("z", mu, sd, n in log_vars)
    return sc


def apply_scaler(raw, names, sc):
    X = np.zeros_like(raw, dtype=np.float32)
    M = (~np.isnan(raw)).astype(np.float32)
    for v, n in enumerate(names):
        s = sc.get(n, ("z", 0.0, 1.0, False))
        x = raw[:, :, v].copy()
        if s[0] == "treat":
            x = x / s[1]
        else:
            if len(s) > 3 and s[3]:
                x = np.log1p(np.clip(x, 0, None))
            x = (x - s[1]) / s[2]
        X[:, :, v] = np.nan_to_num(x, nan=0.0)
    return X, M


def treatment_value(sc, name, raw_value):
    s = sc.get(name, ("treat", 1.0))
    return raw_value / s[1]


def make_synthetic(seed: int = 7, n_patients: int = 2000, train_frac: float = 0.7):
    from src.causal import synthetic as SYN
    data, truth = SYN.simulate(SYN.SimConfig(n_patients=n_patients, seed=seed))
    raw = data["X"]
    N, T, V = raw.shape
    n_vis = data["n_vis"]
    names, roles = data["names"], data["roles"]
    fl_i, sv_i = names.index("flare"), names.index("severity")
    y_flare = np.zeros((N, T), np.float32); y_sev = np.zeros((N, T), np.float32)
    y_flare[:, :-1] = raw[:, 1:, fl_i]
    y_sev[:, :-1] = raw[:, 1:, sv_i]
    valid = (np.arange(T)[None, :] < (n_vis[:, None] - 1))
    y_flare = np.nan_to_num(y_flare); y_sev = np.nan_to_num(y_sev)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(N)
    n_tr = int(train_frac * N)
    split = {"train": np.sort(perm[:n_tr]), "val": np.sort(perm[n_tr:n_tr + (N - n_tr) // 2]),
             "test": np.sort(perm[n_tr + (N - n_tr) // 2:])}
    sc = fit_scaler(raw, n_vis, names, roles, set(data["log_vars"]), split["train"])
    X, M = apply_scaler(raw, names, sc)
    gap = np.log1p(data["gap"] / 30.0).astype(np.float32)
    sg = (data["site"] >= 2).astype(int)
    return SequenceDataset(
        name="SYNTHETIC SLE+RA cohort (simulated)",
        X=X, M=M, raw=raw, gap=gap, n_vis=n_vis, y_flare=y_flare, y_sev=y_sev, valid=valid,
        nuisance=data["nuisance"].astype(np.float32), subgroup=sg, names=names, roles=roles,
        treat_idx=[i for i, r in enumerate(roles) if r == "treatment"],
        patient_ids=[f"SYN-{i:04d}" for i in range(N)],
        scaler=sc, splits=split, truth=truth, extra={"split": split, "static": data["static"], "is_sle": data["is_sle"], "scaler": sc}
    )


REAL_LABS = ["wbc", "lymphocyte_count", "neutrophil_count", "platelet_count", "esr", "hgb", "cr", "alb",
             "ds_dna", "c3", "c4", "ast", "alt", "ldh"]
REAL_IMMUNO = {"ds_dna", "c3", "c4"}
REAL_TREAT = ["steroid_iv_category", "oral_steroids_category", "cyclophosphamide_category",
              "mycophenolate_category", "hydroxychloroquine_category", "metotrexate_category",
              "nsaid_category", "asa_category"]
REAL_FLAGS = ["renal", "musculoskeletal", "serology", "fever", "thrombocytopenia", "leukopenia"]


def make_real(csv_path, seed: int = 0):
    df = pd.read_csv(csv_path).sort_values(["subject", "cumulative_time", "visit_no"]).reset_index(drop=True)
    pats = df.subject.unique().tolist()
    names = REAL_TREAT + REAL_LABS + REAL_FLAGS + ["sledai"]
    roles = (["treatment"] * len(REAL_TREAT)
             + ["immuno" if n in REAL_IMMUNO else "lab" for n in REAL_LABS]
             + ["symptom"] * len(REAL_FLAGS) + ["outcome"])
    N = len(pats); T = int(df.groupby("subject").size().max()); V = len(names)
    raw = np.full((N, T, V), np.nan); gap = np.zeros((N, T), np.float32)
    n_vis = np.zeros(N, int); y_flare = np.zeros((N, T), np.float32); y_sev = np.zeros((N, T), np.float32)
    valid = np.zeros((N, T), bool); age_v = np.zeros(N)

    for i, s in enumerate(pats):
        g = df[df.subject == s].sort_values(["cumulative_time", "visit_no"])
        nv = len(g); n_vis[i] = nv
        raw[i, :nv] = g[names].to_numpy(float)
        gap[i, 1:nv] = np.diff(g.cumulative_time.to_numpy(float))
        sl = g.sledai.to_numpy(float)
        if nv >= 2:
            flare = (sl[1:] - sl[:-1] >= 4).astype(float)
            y_flare[i, :nv - 1] = flare
            y_sev[i, :nv - 1] = np.clip(np.digitize(sl[1:], [-1, 0, 5, 10, 19, 1000]) - 1, 0, 4)
            valid[i, :nv - 1] = True
        age_v[i] = g.age.iloc[0]

    age_z = (age_v - np.nanmean(age_v)) / (np.nanstd(age_v) + 1e-6)
    nuis = np.stack([np.nan_to_num(age_z)], 1).astype(np.float32)
    sg = (nuis[:, 0] > 0).astype(int)
    return SequenceDataset("Real SLE cohort (GSE65391)", np.zeros_like(raw, dtype=np.float32),
                           (~np.isnan(raw)).astype(np.float32), raw, np.log1p(gap / 30.0).astype(np.float32),
                           n_vis, y_flare, y_sev, valid, nuis, sg, names, roles,
                           [i for i, r in enumerate(roles) if r == "treatment"], pats)


def finalize_real(sd, train_idx):
    sd.scaler = fit_scaler(sd.raw, sd.n_vis, sd.names, sd.roles, {"ds_dna", "esr", "cr", "ast", "alt", "ldh"}, train_idx)
    sd.X, sd.M = apply_scaler(sd.raw, sd.names, sd.scaler)
