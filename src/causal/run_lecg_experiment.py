"""Major Project Benchmark: Confidence-Weighted Causal Graph Learning (LECG).

Project Title:
'Improving Autoimmune Disease Progression Prediction Using Confidence-Weighted Causal Graph Learning'

This script executes:
1. Multi-seed training of Original Paper R-GCN (use_lecg=False) vs Proposed LECG R-GCN (use_lecg=True)
   across 3 independent seeds [42, 123, 456].
2. Noise Robustness Experiment:
   Artificially injects K=15 noisy/spurious edges into the causal DAG and compares
   performance drop of Original Paper R-GCN vs Proposed LECG R-GCN.
3. Edge Confidence Analysis:
   Quantifies learned edge confidence w_e = sigmoid(theta_e) on true biological edges vs noisy edges.
4. Generates publication figures:
   - fig14_lecg_edge_confidence_distribution.png
   - fig15_lecg_ablation_performance.png
5. Exports:
   - reports/causal/lecg_comparison_table.csv
   - reports/causal/lecg_comparison_results.json
   - models/causal_lecg_model.pt

Usage:
    python -m src.causal.run_lecg_experiment [--fast] [--epochs 16]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "causal"
MODEL_DIR = ROOT / "models"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

from src.causal import dataset as D
from src.causal import graph as G
from src.causal import train as T
from src.causal.model import CausalCfg, CausalDeepModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inject_noisy_edges(
    A_clean: np.ndarray,
    k_noise: int = 15,
    seed: int = 999
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Injects k_noise random/spurious edges into adjacency tensor A.

    Returns:
        A_noisy: Adjacency tensor with added noisy edges.
        noisy_coords: List of (rel, src, dst) tuples identifying the injected edges.
    """
    rng = np.random.default_rng(seed)
    A_noisy = A_clean.copy()
    n_rel, V_tot, _ = A_clean.shape

    noisy_coords = []
    # Avoid latent U node (index V_tot - 1)
    cand_nodes = list(range(V_tot - 1))

    attempts = 0
    while len(noisy_coords) < k_noise and attempts < 1000:
        attempts += 1
        r = int(rng.integers(0, n_rel))
        u = int(rng.choice(cand_nodes))
        v = int(rng.choice(cand_nodes))
        if u != v and A_noisy[r, v, u] == 0:
            A_noisy[r, v, u] = 1.0  # Directed edge u -> v
            noisy_coords.append((r, v, u))

    return A_noisy, noisy_coords


def train_and_eval_condition(
    condition_name: str,
    use_lecg: bool,
    A_tensor: np.ndarray,
    sd_train: D.SequenceDataset,
    sd_val: D.SequenceDataset,
    sd_test: D.SequenceDataset,
    seed: int,
    epochs: int = 16,
    batch_size: int = 64,
    device: torch.device = DEVICE
) -> tuple[dict[str, float], CausalDeepModel]:
    """Trains and evaluates one experimental condition."""
    cfg = CausalCfg(
        d=32,
        d_c=32,
        d_s=16,
        d_h=64,
        n_heads=4,
        n_layers=2,
        scales=(1, 3, 6),
        dropout=0.15,
        k_mix=3,
        use_lecg=use_lecg,
        init_edge_confidence=2.2
    )

    model = T.train_causal_model(
        sd_train=sd_train,
        sd_val=sd_val,
        A_relation=A_tensor,
        cfg=cfg,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-4,
        seed=seed,
        device=device,
        verbose=False
    )

    test_preds = T.predict_outcomes(model, sd_test, mc_samples=25, device=device)
    v_test = sd_test.valid
    metrics = T.calculate_metrics(
        y_true_flare=sd_test.y_flare[v_test],
        y_pred_flare=test_preds["p"][v_test],
        y_true_sev=sd_test.y_sev[v_test],
        y_pred_sev=test_preds["sev"][v_test]
    )

    return metrics, model


def plot_edge_confidence_distribution(
    model_lecg_noisy: CausalDeepModel,
    noisy_coords: list[tuple[int, int, int]],
    A_clean: np.ndarray,
    save_path: Path
) -> dict[str, float]:
    """Generates distribution plot of learned confidence on True vs Noisy edges."""
    confs = model_lecg_noisy.get_edge_confidences()
    if confs is None:
        return {"true_conf_mean": 0.0, "noisy_conf_mean": 0.0}

    confs_np = confs.detach().cpu().numpy()

    # Mask for true edges (in A_clean)
    true_mask = (A_clean > 0)
    true_confs = confs_np[true_mask]

    # Values on injected noisy edges
    noisy_confs = np.array([confs_np[r, v, u] for (r, v, u) in noisy_coords])

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.hist(true_confs, bins=15, range=(0, 1), alpha=0.7, color="#2ca02c", label=f"Prior Graph Edges (N={len(true_confs)})", edgecolor="black")
    ax.hist(noisy_confs, bins=15, range=(0, 1), alpha=0.75, color="#d62728", label=f"Injected Noisy Edges (N={len(noisy_confs)})", edgecolor="black")

    mean_true = float(np.mean(true_confs))
    mean_noisy = float(np.mean(noisy_confs))

    ax.axvline(mean_true, color="#1b5e20", linestyle="--", linewidth=2, label=f"Mean True Confidence: {mean_true:.2f}")
    ax.axvline(mean_noisy, color="#b71c1c", linestyle="--", linewidth=2, label=f"Mean Noisy Confidence: {mean_noisy:.2f}")

    ax.set_title("Figure 14: Learned Edge-Confidence Gating (LECG) Distribution\nPrior and Injected Edge Weights", fontsize=11, fontweight="bold")
    ax.set_xlabel("Learned Edge Confidence $w_e = \\sigma(\\theta_e) \\in [0, 1]$", fontsize=10)
    ax.set_ylabel("Edge Count", fontsize=10)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    return {"true_conf_mean": mean_true, "noisy_conf_mean": mean_noisy}


def plot_performance_comparison(
    results_summary: dict[str, Any],
    save_path: Path
) -> None:
    """Generates comparison bar chart between Original Paper R-GCN and LECG."""
    labels = ["Clean Causal DAG", "Noisy Corrupted DAG (+15 edges)"]
    x = np.arange(len(labels))
    width = 0.35

    orig_auroc = [
        results_summary["clean"]["original"]["auroc_mean"],
        results_summary["noisy"]["original"]["auroc_mean"]
    ]
    orig_auroc_std = [
        results_summary["clean"]["original"]["auroc_std"],
        results_summary["noisy"]["original"]["auroc_std"]
    ]

    lecg_auroc = [
        results_summary["clean"]["lecg"]["auroc_mean"],
        results_summary["noisy"]["lecg"]["auroc_mean"]
    ]
    lecg_auroc_std = [
        results_summary["clean"]["lecg"]["auroc_std"],
        results_summary["noisy"]["lecg"]["auroc_std"]
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # 1. AUROC comparison
    rects1 = ax1.bar(x - width/2, orig_auroc, width, yerr=orig_auroc_std, capsize=4, label="Original Paper (R-GCN)", color="#1f77b4", alpha=0.85, edgecolor="black")
    rects2 = ax1.bar(x + width/2, lecg_auroc, width, yerr=lecg_auroc_std, capsize=4, label="Proposed (LECG R-GCN)", color="#2ca02c", alpha=0.85, edgecolor="black")

    ax1.set_ylabel("Test AUROC", fontsize=10)
    ax1.set_title("Flare Prediction AUROC (Mean ± Std, 3 Seeds)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    max_a = max(max(orig_auroc), max(lecg_auroc))
    min_a = min(min(orig_auroc), min(lecg_auroc))
    ax1.set_ylim(max(0.0, min_a - 0.12), min(1.0, max_a + 0.12))
    ax1.legend(loc="lower left", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.5)

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # 2. F1-Score comparison
    orig_f1 = [
        results_summary["clean"]["original"]["f1_mean"],
        results_summary["noisy"]["original"]["f1_mean"]
    ]
    orig_f1_std = [
        results_summary["clean"]["original"]["f1_std"],
        results_summary["noisy"]["original"]["f1_std"]
    ]
    lecg_f1 = [
        results_summary["clean"]["lecg"]["f1_mean"],
        results_summary["noisy"]["lecg"]["f1_mean"]
    ]
    lecg_f1_std = [
        results_summary["clean"]["lecg"]["f1_std"],
        results_summary["noisy"]["lecg"]["f1_std"]
    ]

    rects3 = ax2.bar(x - width/2, orig_f1, width, yerr=orig_f1_std, capsize=4, label="Original Paper (R-GCN)", color="#1f77b4", alpha=0.85, edgecolor="black")
    rects4 = ax2.bar(x + width/2, lecg_f1, width, yerr=lecg_f1_std, capsize=4, label="Proposed (LECG R-GCN)", color="#2ca02c", alpha=0.85, edgecolor="black")

    ax2.set_ylabel("Test F1-Score", fontsize=10)
    ax2.set_title("Flare Prediction F1-Score (Mean ± Std, 3 Seeds)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    max_f = max(max(orig_f1), max(lecg_f1))
    min_f = min(min(orig_f1), min(lecg_f1))
    ax2.set_ylim(max(0.0, min_f - 0.12), min(1.0, max_f + 0.12))
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.5)

    for rect in rects3 + rects4:
        h = rect.get_height()
        ax2.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Figure 15: Comparative Evaluation — Learnable Edge-Confidence Gating vs Original Paper R-GCN", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def run_lecg_benchmark(fast: bool = False, epochs_override: int | None = None) -> None:
    """Master benchmark executing 3-seed evaluation across Clean and Noisy DAGs."""
    print("=" * 85)
    print("MAJOR PROJECT EXPERIMENT: CONFIDENCE-WEIGHTED CAUSAL GRAPH LEARNING (LECG)")
    print(f"Hardware: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    epochs = epochs_override if epochs_override is not None else (4 if fast else 14)
    seeds = [42, 123] if fast else [42, 123, 456]
    print(f"Configuration: Seeds={seeds}, Epochs per run={epochs}, Fast Mode={fast}")
    print("=" * 85)

    # 1. Load Multimodal Dataset
    print("\n[Step 1/5] Loading 54-dimensional Multimodal Dataset ...")
    sd = D.load_multimodal_cohort()
    sd_train = D.subset_dataset(sd, sd.splits["train"])
    sd_val = D.subset_dataset(sd, sd.splits["val"])
    sd_test = D.subset_dataset(sd, sd.splits["test"])
    print(f"Train: {sd_train.N} patients | Val: {sd_val.N} | Test: {sd_test.N}")

    # 2. Synthesize Clean Causal DAG
    print("\n[Step 2/5] Synthesizing Clean Causal DAG (Eq. 4) ...")
    G_prior = G.build_prior_graph(sd.names, sd.roles)
    X_train_norm = np.where(sd_train.M > 0, sd_train.X, np.nan)
    G_stat, _ = G.build_statistical_graph(X_train_norm, sd_train.n_vis, sd.names, sd.roles)
    G_fused = G.synthesize_causal_dag(G_prior, G_stat, lam=0.6, threshold=0.35)
    A_clean = G.build_relation_tensor(G_fused, sd.names, sd.roles)
    n_clean_edges = int((A_clean > 0).sum())
    print(f"Clean Causal DAG: {n_clean_edges} relation edges verified.")

    # 3. Create Noisy Corrupted DAG
    print("\n[Step 3/5] Injecting K=15 Spurious/Incorrect Edges for Robustness Evaluation ...")
    A_noisy, noisy_coords = inject_noisy_edges(A_clean, k_noise=15, seed=999)
    print(f"Corrupted DAG created: {int((A_noisy > 0).sum())} total edges (15 noisy edges added).")

    # 4. Multi-seed training across the 4 experimental conditions
    conditions = {
        "clean_original": {"use_lecg": False, "A": A_clean, "name": "Clean DAG - Original Paper R-GCN"},
        "clean_lecg":     {"use_lecg": True,  "A": A_clean, "name": "Clean DAG - Proposed LECG R-GCN"},
        "noisy_original": {"use_lecg": False, "A": A_noisy, "name": "Noisy DAG - Original Paper R-GCN"},
        "noisy_lecg":     {"use_lecg": True,  "A": A_noisy, "name": "Noisy DAG - Proposed LECG R-GCN"},
    }

    raw_results: dict[str, list[dict[str, float]]] = {k: [] for k in conditions}
    best_lecg_model: CausalDeepModel | None = None
    best_lecg_val_brier = float("inf")
    model_noisy_lecg: CausalDeepModel | None = None

    print("\n[Step 4/5] Executing Multi-Seed Training Benchmark ...")
    t_start = time.time()

    for cond_key, cond_info in conditions.items():
        print(f"\n>> Evaluating Condition: {cond_info['name']}")
        for s in seeds:
            print(f"   -> Seed {s} ...", end="", flush=True)
            t_run0 = time.time()
            metrics, model = train_and_eval_condition(
                condition_name=cond_info["name"],
                use_lecg=cond_info["use_lecg"],
                A_tensor=cond_info["A"],
                sd_train=sd_train,
                sd_val=sd_val,
                sd_test=sd_test,
                seed=s,
                epochs=epochs,
                device=DEVICE
            )
            raw_results[cond_key].append(metrics)
            dur = time.time() - t_run0
            print(f" AUROC={metrics['auroc']:.4f} | F1={metrics['f1']:.4f} | AUPRC={metrics['auprc']:.4f} ({dur:.1f}s)")

            # Keep track of best models
            if cond_key == "clean_lecg":
                validation = T.predict_outcomes(model, sd_val, mc_samples=0, device=DEVICE)
                score = float(np.mean((validation["p"][sd_val.valid] - sd_val.y_flare[sd_val.valid]) ** 2))
                if score < best_lecg_val_brier:
                    best_lecg_val_brier = score
                    best_lecg_model = model
            if cond_key == "noisy_lecg" and model_noisy_lecg is None:
                model_noisy_lecg = model

    total_bench_time = time.time() - t_start
    print(f"\nAll benchmark runs completed in {total_bench_time:.1f}s.")

    # Aggregate metrics across seeds
    def summarize(runs: list[dict[str, float]]) -> dict[str, float]:
        res = {}
        for metric in ["auroc", "auprc", "f1", "mae", "rmse", "ece"]:
            vals = [r[metric] for r in runs]
            res[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
            res[f"{metric}_std"] = round(float(np.std(vals)), 4)
        return res

    summary_clean_orig = summarize(raw_results["clean_original"])
    summary_clean_lecg = summarize(raw_results["clean_lecg"])
    summary_noisy_orig = summarize(raw_results["noisy_original"])
    summary_noisy_lecg = summarize(raw_results["noisy_lecg"])

    results_structured = {
        "clean": {
            "original": summary_clean_orig,
            "lecg": summary_clean_lecg,
            "gain_auroc": round(summary_clean_lecg["auroc_mean"] - summary_clean_orig["auroc_mean"], 4),
            "gain_f1": round(summary_clean_lecg["f1_mean"] - summary_clean_orig["f1_mean"], 4),
        },
        "noisy": {
            "original": summary_noisy_orig,
            "lecg": summary_noisy_lecg,
            "auroc_retention": round(summary_noisy_lecg["auroc_mean"] - summary_noisy_orig["auroc_mean"], 4),
            "f1_retention": round(summary_noisy_lecg["f1_mean"] - summary_noisy_orig["f1_mean"], 4),
        },
        "seeds": seeds,
        "n_noisy_edges": len(noisy_coords),
        "total_time_seconds": round(total_bench_time, 1)
    }

    # 5. Visualizations & Edge-Confidence Distribution Analysis
    print("\n[Step 5/5] Generating Publication Figures & Confidence Analysis ...")
    fig14_path = REPORT_DIR / "fig14_lecg_edge_confidence_distribution.png"
    if model_noisy_lecg is not None:
        conf_stats = plot_edge_confidence_distribution(
            model_lecg_noisy=model_noisy_lecg,
            noisy_coords=noisy_coords,
            A_clean=A_clean,
            save_path=fig14_path
        )
        results_structured["edge_confidence_stats"] = conf_stats
        print(f"  -> Figure 14 saved: {fig14_path}")
        print(f"     Learned True Edge Confidence:  {conf_stats['true_conf_mean']:.3f}")
        print(f"     Learned Noisy Edge Confidence: {conf_stats['noisy_conf_mean']:.3f} (learned predictive weights)")

    fig15_path = REPORT_DIR / "fig15_lecg_ablation_performance.png"
    plot_performance_comparison(results_structured, fig15_path)
    print(f"  -> Figure 15 saved: {fig15_path}")

    # Save checkpoint of best LECG model
    if best_lecg_model is not None:
        lecg_ckpt_path = MODEL_DIR / "causal_lecg_model.pt"
        torch.save({
            "state_dict": best_lecg_model.state_dict(),
            "cfg": best_lecg_model.cfg,
            "normalization": sd.extra.get("normalization", "train"),
        "scaler": sd.scaler,
            "platt": getattr(best_lecg_model, "platt_calibrator", None),
            "A_relation": A_clean,
            "names": sd.names,
            "roles": sd.roles,
            "treat_idx": sd.treat_idx,
            "metrics": summary_clean_lecg
        }, lecg_ckpt_path)
        print(f"  -> Best LECG Model checkpoint saved: {lecg_ckpt_path}")

    # Save JSON results
    json_path = REPORT_DIR / "lecg_comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(results_structured, f, indent=2)
    print(f"  -> Detailed results saved: {json_path}")

    # Build and print presentation table
    table_rows = [
        {
            "Graph Condition": "Clean Causal DAG",
            "Model Architecture": "Original Paper R-GCN",
            "AUROC": f"{summary_clean_orig['auroc_mean']:.3f} ± {summary_clean_orig['auroc_std']:.3f}",
            "AUPRC": f"{summary_clean_orig['auprc_mean']:.3f} ± {summary_clean_orig['auprc_std']:.3f}",
            "F1-Score": f"{summary_clean_orig['f1_mean']:.3f} ± {summary_clean_orig['f1_std']:.3f}",
            "Severity MAE": f"{summary_clean_orig['mae_mean']:.3f} ± {summary_clean_orig['mae_std']:.3f}",
        },
        {
            "Graph Condition": "Clean Causal DAG",
            "Model Architecture": "Proposed LECG R-GCN",
            "AUROC": f"{summary_clean_lecg['auroc_mean']:.3f} ± {summary_clean_lecg['auroc_std']:.3f}",
            "AUPRC": f"{summary_clean_lecg['auprc_mean']:.3f} ± {summary_clean_lecg['auprc_std']:.3f}",
            "F1-Score": f"{summary_clean_lecg['f1_mean']:.3f} ± {summary_clean_lecg['f1_std']:.3f}",
            "Severity MAE": f"{summary_clean_lecg['mae_mean']:.3f} ± {summary_clean_lecg['mae_std']:.3f}",
        },
        {
            "Graph Condition": "Corrupted DAG (+15 Noisy Edges)",
            "Model Architecture": "Original Paper R-GCN",
            "AUROC": f"{summary_noisy_orig['auroc_mean']:.3f} ± {summary_noisy_orig['auroc_std']:.3f}",
            "AUPRC": f"{summary_noisy_orig['auprc_mean']:.3f} ± {summary_noisy_orig['auprc_std']:.3f}",
            "F1-Score": f"{summary_noisy_orig['f1_mean']:.3f} ± {summary_noisy_orig['f1_std']:.3f}",
            "Severity MAE": f"{summary_noisy_orig['mae_mean']:.3f} ± {summary_noisy_orig['mae_std']:.3f}",
        },
        {
            "Graph Condition": "Corrupted DAG (+15 Noisy Edges)",
            "Model Architecture": "Proposed LECG R-GCN",
            "AUROC": f"{summary_noisy_lecg['auroc_mean']:.3f} ± {summary_noisy_lecg['auroc_std']:.3f}",
            "AUPRC": f"{summary_noisy_lecg['auprc_mean']:.3f} ± {summary_noisy_lecg['auprc_std']:.3f}",
            "F1-Score": f"{summary_noisy_lecg['f1_mean']:.3f} ± {summary_noisy_lecg['f1_std']:.3f}",
            "Severity MAE": f"{summary_noisy_lecg['mae_mean']:.3f} ± {summary_noisy_lecg['mae_std']:.3f}",
        },
    ]

    df_table = pd.DataFrame(table_rows)
    csv_table_path = REPORT_DIR / "lecg_comparison_table.csv"
    df_table.to_csv(csv_table_path, index=False)
    print(f"  -> Presentation table saved: {csv_table_path}")

    print("\n" + "=" * 95)
    print("MAJOR PROJECT BENCHMARK SUMMARY TABLE: ORIGINAL PAPER vs LECG")
    print("=" * 95)
    print(df_table.to_string(index=False))
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LECG major project comparative experiments.")
    parser.add_argument("--fast", action="store_true", help="Quick run with fewer seeds and epochs")
    parser.add_argument("--epochs", type=int, default=None, help="Explicit epoch budget")
    args = parser.parse_args()

    run_lecg_benchmark(fast=args.fast, epochs_override=args.epochs)
