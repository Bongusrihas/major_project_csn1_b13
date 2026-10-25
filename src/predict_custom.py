"""Inference Pipeline for Personal / Custom Longitudinal Patient Data.

Loads the trained Paper Causal Framework (R-GCN + TCT + CSE + CDI + UQ) on GPU,
evaluates arbitrary patient clinical records, estimates next-visit flare risk with
epistemic & aleatoric uncertainty, and simulates counterfactual treatment decisions.

Usage:
    # 1. Run inference on custom patient CSV:
    python -m src.predict_custom --input data/template_patient.csv --output reports/custom_predictions.csv

    # 2. Run for a specific patient ID:
    python -m src.predict_custom --input data/template_patient.csv --patient_id PATIENT-DEMO-01

    # 3. Simulate specific counterfactual interventions:
    python -m src.predict_custom --input data/template_patient.csv --simulate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.causal import dataset as D
from src.causal import train as T
from src.causal.dataset import SequenceDataset
from src.causal.model import CausalDeepModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_PATH = ROOT / "models" / "causal_validated_model.pt"
if not DEFAULT_MODEL_PATH.exists():
    DEFAULT_MODEL_PATH = ROOT / "models" / "causal_paper_model.pt"
METADATA_PATH = ROOT / "data" / "processed" / "feature_metadata.json"


def load_trained_causal_model(model_path: Path = DEFAULT_MODEL_PATH, device: torch.device = DEVICE) -> tuple[CausalDeepModel, dict[str, Any]]:
    """Loads checkpoint and initializes CausalDeepModel on the target device."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. "
            f"Please run `python -m src.causal.run_experiments` to train the model first."
        )

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    names = ckpt["names"]
    treat_idx = ckpt["treat_idx"]
    A_relation = ckpt.get("A_relation", ckpt.get("A"))

    # We assume 4 nuisance dimensions (age, female, smoking, comorbidity)
    model = CausalDeepModel(
        V=len(names),
        treat_idx=treat_idx,
        A=A_relation,
        cfg=cfg,
        n_nuisance=ckpt.get("n_nuisance", 4)
    ).to(device)

    model.load_state_dict(ckpt.get("state_dict", ckpt.get("state")))
    if ckpt.get("platt") is not None:
        model.platt_calibrator = ckpt["platt"]
    model.eval()
    return model, ckpt


def prepare_custom_dataset(
    df: pd.DataFrame,
    expected_features: list[str],
    treat_idx: list[int],
    normalization: str = "prefix",
    scaler: dict | None = None
) -> SequenceDataset:
    """Transforms a custom pandas DataFrame into a normalized SequenceDataset.

    Handles missing columns by imputing reasonable cohort defaults and masking with M=0.
    """
    df = df.copy()

    # Required identifiers
    if "patient_id" not in df.columns:
        if "subject" in df.columns:
            df["patient_id"] = df["subject"]
        else:
            df["patient_id"] = "CUSTOM_PATIENT"

    if "visit_index" not in df.columns:
        if "visit_no" in df.columns:
            df["visit_index"] = df["visit_no"].astype(int)
        else:
            df["visit_index"] = df.groupby("patient_id").cumcount()

    if "gap_days" not in df.columns:
        if "cumulative_days" in df.columns:
            df["gap_days"] = df.groupby("patient_id")["cumulative_days"].diff().fillna(0.0)
        else:
            df["gap_days"] = 60.0  # Typical 2-month interval

    patient_ids = list(df["patient_id"].unique())
    N = len(patient_ids)
    max_visits = df.groupby("patient_id")["visit_index"].count().max()
    V = len(expected_features)

    raw = np.full((N, max_visits, V), np.nan, dtype=np.float32)
    gap = np.zeros((N, max_visits), dtype=np.float32)
    n_vis = np.zeros(N, dtype=int)
    valid = np.zeros((N, max_visits), dtype=bool)
    y_flare = np.zeros((N, max_visits), dtype=np.float32)
    y_sev = np.zeros((N, max_visits), dtype=np.float32)
    nuisance = np.zeros((N, 4), dtype=np.float32)
    subgroup = np.zeros(N, dtype=int)

    # Fill feature columns
    for f in expected_features:
        if f not in df.columns:
            # Impute typical neutral baseline
            df[f] = np.nan

    for i, pid in enumerate(patient_ids):
        pdf = df[df["patient_id"] == pid].sort_values("visit_index")
        nv = len(pdf)
        n_vis[i] = nv

        v_indices = np.arange(nv)
        raw[i, v_indices, :] = pdf[expected_features].to_numpy(dtype=np.float32)
        gap[i, v_indices] = pdf["gap_days"].to_numpy(dtype=np.float32)

        # Valid prediction step for any visit where next visit or future outcome is evaluated
        valid[i, :nv] = True

        # Ground truth outcomes if provided
        if "flare_event" in pdf.columns:
            y_flare[i, :nv] = pdf["flare_event"].to_numpy(dtype=np.float32)
        if "severity_score" in pdf.columns:
            y_sev[i, :nv] = pdf["severity_score"].to_numpy(dtype=np.float32)

        # Demographics / Nuisance
        p_first = pdf.iloc[0]
        age_val = float(p_first.get("age", 45.0))
        fem_val = float(p_first.get("female", 1.0))
        smk_val = float(p_first.get("smoking_status", 0.0))
        com_val = float(p_first.get("comorbidity_index", 0.0))
        nuisance[i] = [
            (age_val - 48.0) / 12.0,
            fem_val,
            smk_val,
            com_val / 3.0
        ]
        subgroup[i] = int(p_first.get("disease_type", 0))

    if normalization == "train":
        if scaler is None:
            raise ValueError("The training-fitted scaler is required for this checkpoint")
        X, M = D.apply_scaler(raw, expected_features, scaler)
    elif normalization not in ("prefix", "legacy"):
        raise ValueError("normalization must be prefix or legacy")
    else:
        X, M = (D.prefix_minmax if normalization == "prefix" else D.legacy_minmax)(raw, n_vis)

    gap_norm = np.log1p(np.clip(gap, 0, None) / 30.0).astype(np.float32)

    roles = [
        "treatment" if i in treat_idx else ("nuisance" if f in ["age", "female"] else "symptom")
        for i, f in enumerate(expected_features)
    ]

    return SequenceDataset(
        name="Custom Personal Patient Cohort",
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
        names=expected_features,
        roles=roles,
        treat_idx=treat_idx,
        patient_ids=patient_ids,
        extra={"normalization": normalization},
        scaler=scaler,
        splits={"test": np.arange(N)}
    )


def predict_patient_cohort(
    model: CausalDeepModel,
    sd: SequenceDataset,
    mc_samples: int = 30,
    device: torch.device = DEVICE,
    run_counterfactuals: bool = True
) -> pd.DataFrame:
    """Computes comprehensive factual predictions, UQ intervals, and counterfactuals."""
    print(f"[Inference] Generating predictions on {device} ({sd.N} patients, {sd.n_vis.sum()} total visits) ...")

    # Predict outcomes with MC Dropout
    preds = T.predict_outcomes(model, sd, mc_samples=mc_samples, device=device)

    records = []
    treat_names = [sd.names[i] for i in sd.treat_idx]

    for i, pid in enumerate(sd.patient_ids):
        nv = sd.n_vis[i]
        for t in range(nv):
            p_flare = float(preds["p"][i, t])
            p_lo = float(preds["p_lo"][i, t])
            p_hi = float(preds["p_hi"][i, t])
            sev = float(preds["sev"][i, t])
            sev_sd = float(preds["sev_sd"][i, t])

            # Clinical risk tier
            if p_flare < 0.20:
                tier = "LOW"
            elif p_flare < 0.50:
                tier = "MODERATE"
            elif p_flare < 0.75:
                tier = "HIGH"
            else:
                tier = "CRITICAL"

            rec = {
                "patient_id": pid,
                "visit_index": t,
                "predicted_flare_prob": round(p_flare, 4),
                "flare_prob_90ci_lower": round(p_lo, 4),
                "flare_prob_90ci_upper": round(p_hi, 4),
                "predicted_severity": round(sev, 2),
                "severity_sd": round(sev_sd, 3),
                "risk_tier": tier,
            }

            # Add key current clinical labs and treatments
            for key_feat in ["crp", "esr", "corticosteroid_dose", "dmard_mtx", "biologic_anti_tnf"]:
                if key_feat in sd.names:
                    idx = sd.names.index(key_feat)
                    rec[key_feat] = round(float(sd.raw[i, t, idx]), 2)

            # Optional counterfactual simulations
            if run_counterfactuals and len(treat_names) > 0:
                # 1. Stop steroids
                if "corticosteroid_dose" in treat_names:
                    cf_stop = T.simulate_counterfactual_intervention(
                        model, sd, i, t,
                        intervention_dict={"corticosteroid_dose": 0.0},
                        mc_samples=15, device=device
                    )
                    rec["cf_risk_stop_steroids"] = round(cf_stop["p_counterfactual"], 4)
                    rec["effect_stop_steroids"] = round(cf_stop["flare_risk_delta"], 4)

                # 2. Add Biologic Anti-TNF
                if "biologic_anti_tnf" in treat_names:
                    cf_bio = T.simulate_counterfactual_intervention(
                        model, sd, i, t,
                        intervention_dict={"biologic_anti_tnf": 1.0},
                        mc_samples=15, device=device
                    )
                    rec["cf_risk_add_biologic"] = round(cf_bio["p_counterfactual"], 4)
                    rec["effect_add_biologic"] = round(cf_bio["flare_risk_delta"], 4)

            records.append(rec)

    return pd.DataFrame(records)


def print_cli_summary(df_preds: pd.DataFrame) -> None:
    """Prints a clear, formatted clinical summary report to stdout."""
    print("\n" + "=" * 90)
    print("PATIENT LONGITUDINAL HEALTH PREDICTION - CLINICAL INFERENCE REPORT")
    print("=" * 90)

    for pid, group in df_preds.groupby("patient_id"):
        print(f"\n[Patient ID: {pid}]  (Total Visits Tracked: {len(group)})")
        print("-" * 90)
        print(f"{'Visit':<7} | {'Flare Risk':<12} | {'90% CI':<14} | {'Severity':<10} | {'Tier':<10} | {'CF: Stop Steroids':<18} | {'CF: Add Biologic':<16}")
        print("-" * 90)

        for _, row in group.iterrows():
            v = int(row["visit_index"])
            p = row["predicted_flare_prob"]
            ci = f"[{row['flare_prob_90ci_lower']:.1%}, {row['flare_prob_90ci_upper']:.1%}]"
            sev = f"{row['predicted_severity']:.1f} ± {row['severity_sd']:.2f}"
            tier = row["risk_tier"]

            cf_steroid = f"{row.get('cf_risk_stop_steroids', 0.0):.1%} ({row.get('effect_stop_steroids', 0.0)*100:+.1f}%)" if "cf_risk_stop_steroids" in row else "N/A"
            cf_bio = f"{row.get('cf_risk_add_biologic', 0.0):.1%} ({row.get('effect_add_biologic', 0.0)*100:+.1f}%)" if "cf_risk_add_biologic" in row else "N/A"

            # Color styling indicator
            tag = "🔴" if tier in ["HIGH", "CRITICAL"] else ("🟡" if tier == "MODERATE" else "🟢")
            print(f"V-{v:<5} | {p:<10.1%} | {ci:<14} | {sev:<10} | {tag} {tier:<7} | {cf_steroid:<18} | {cf_bio:<16}")

    print("\n" + "=" * 90)
    print("Legend: [🟢 LOW: <20%]  [🟡 MODERATE: 20-50%]  [🔴 HIGH/CRITICAL: >50%]")
    print("CF = Counterfactual Potential Outcome risk if clinician modifies treatment regimen at that visit.")
    print("=" * 90 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run causal inference on personal patient data.")
    parser.add_argument("--input", type=str, default="data/template_patient.csv", help="Path to input patient CSV")
    parser.add_argument("--patient_id", type=str, default=None, help="Filter to a specific patient ID")
    parser.add_argument("--output", type=str, default="reports/custom_predictions.csv", help="Output CSV path")
    parser.add_argument("--json_output", type=str, default=None, help="Optional output JSON path")
    parser.add_argument("--no_cf", action="store_true", help="Skip counterfactual simulations for faster inference")
    parser.add_argument("--mc_samples", type=int, default=30, help="Monte Carlo Dropout samples for uncertainty")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to trained model")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)

    print(f"[Setup] Hardware: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"[Setup] Loading model from: {args.model_path}")
    model, ckpt = load_trained_causal_model(Path(args.model_path), device=DEVICE)

    print(f"[Setup] Loading input data from: {input_path}")
    df_raw = pd.read_csv(input_path)

    if args.patient_id is not None:
        df_raw = df_raw[df_raw["patient_id"] == args.patient_id]
        if len(df_raw) == 0:
            print(f"Error: Patient ID '{args.patient_id}' not found in {input_path}")
            sys.exit(1)

    # Prepare dataset
    sd = prepare_custom_dataset(df_raw, ckpt["names"], ckpt["treat_idx"], normalization=ckpt.get("normalization", "legacy"), scaler=ckpt.get("scaler"))

    # Generate predictions
    df_preds = predict_patient_cohort(
        model, sd,
        mc_samples=args.mc_samples,
        device=DEVICE,
        run_counterfactuals=not args.no_cf
    )

    # Print summary
    print_cli_summary(df_preds)

    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_preds.to_csv(out_path, index=False)
    print(f"[Output] Full tabular predictions saved to: {out_path}")

    if args.json_output:
        json_p = Path(args.json_output)
        json_p.parent.mkdir(parents=True, exist_ok=True)
        with open(json_p, "w") as f:
            json.dump(df_preds.to_dict(orient="records"), f, indent=2)
        print(f"[Output] JSON records saved to: {json_p}")


if __name__ == "__main__":
    main()
