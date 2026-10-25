"""Publication-quality figure generation reproducing Paper Figures 6-13.

Generates:
- Figure 6: Latent space t-SNE visualization (Causal vs Spurious components).
- Figure 7 & 12: Uncertainty spatiotemporal heatmap across time steps.
- Figure 8: Counterfactual simulation under DMARD withdrawal (with vs without intervention).
- Figure 9: AUROC curves by patient subgroups (age and gender strata).
- Figure 10: Attention heatmap highlighting salient time steps and clinical features.
- Figure 11: t-SNE projection of latent representations colored by flare outcome.
- Figure 13: Ablation study performance drop bar chart (AUROC and F1 drop).
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc

OUT_DIR = Path("reports/causal")
FIG_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 200,
})


def plot_fig6_tsne_disentanglement(zc: np.ndarray, zs: np.ndarray, labels: np.ndarray, save_path: Path | None = None):
    """Figure 6: Latent space visualization using t-SNE.
    Left: Causal features form structured clusters by disease state.
    Right: Spurious components lack structure, showing effective disentanglement.
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=800)
    # Subsample if large
    n = min(len(zc), 1000)
    idx = np.random.default_rng(42).choice(len(zc), n, replace=False)
    
    emb_c = tsne.fit_transform(zc[idx])
    emb_s = tsne.fit_transform(zs[idx])
    y = labels[idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Causal Latent Space
    sc1 = ax1.scatter(emb_c[y == 0, 0], emb_c[y == 0, 1], c="#2b5c8f", alpha=0.6, s=15, label="Stable / Remission")
    sc2 = ax1.scatter(emb_c[y == 1, 0], emb_c[y == 1, 1], c="#d95f02", alpha=0.7, s=18, label="Flare State")
    ax1.set_title("Causal Features (Structured Clusters)", fontweight="bold")
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")
    ax1.legend(frameon=True, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Spurious Latent Space
    ax2.scatter(emb_s[y == 0, 0], emb_s[y == 0, 1], c="#7570b3", alpha=0.5, s=15, label="Subgroup A")
    ax2.scatter(emb_s[y == 1, 0], emb_s[y == 1, 1], c="#1b9e77", alpha=0.5, s=15, label="Subgroup B")
    ax2.set_title("Spurious Features (Overlapping)", fontweight="bold")
    ax2.set_xlabel("t-SNE Dimension 1")
    ax2.set_ylabel("t-SNE Dimension 2")
    ax2.legend(frameon=True, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("FIGURE 6. Latent Space Disentanglement (t-SNE Visualization)", y=0.98, fontweight="bold")
    fig.tight_layout()
    target = save_path or (OUT_DIR / "fig6_tsne_disentanglement.png")
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Saved {target}")


def plot_fig7_12_uncertainty_heatmap(uncertainty_matrix: np.ndarray, time_steps: int = 25, save_path: Path | None = None):
    """Figure 7 & 12: Spatiotemporal uncertainty heatmap across clinical timeline.
    Darker/red regions indicate higher predictive uncertainty during flare transitions.
    """
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(uncertainty_matrix, cmap="coolwarm", aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Predictive Uncertainty", rotation=270, labelpad=12)

    ax.set_title("FIGURE 7 & 12. Predictive Uncertainty Heatmap Across Time Steps", fontweight="bold")
    ax.set_xlabel("Time Steps (Visits)")
    ax.set_ylabel("Feature Group / Patient State")
    ax.set_xticks(range(0, time_steps, 2))
    ax.set_xticklabels([f"T{i+1}" for i in range(0, time_steps, 2)])
    fig.tight_layout()
    target = save_path or (OUT_DIR / "fig7_uncertainty_heatmap.png")
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Saved {target}")


def plot_fig8_counterfactual_dmard_withdrawal(
    times: np.ndarray,
    p_with_dmard: np.ndarray,
    p_without_dmard: np.ndarray,
    intervention_step: int = 15,
    save_path: Path | None = None
):
    """Figure 8: Counterfactual simulation under DMARD withdrawal:
    predicted flare probability over time with (blue) and without (red) intervention.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))

    ax.plot(times, p_with_dmard, color="#1f77b4", lw=2.2, label="With DMARD (Factual / Maintenance)")
    ax.plot(times, p_without_dmard, color="#d62728", lw=2.2, linestyle="--", label="Without DMARD (Counterfactual / Withdrawal)")

    # Mark intervention point
    ax.axvline(x=times[intervention_step], color="black", linestyle=":", lw=1.5, alpha=0.8)
    ax.text(
        times[intervention_step] + 0.8,
        0.82,
        "DMARD Withdrawal\nIntervention do(a')",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
    )

    ax.set_title("FIGURE 8. Counterfactual Simulation Under DMARD Withdrawal", fontweight="bold")
    ax.set_xlabel("Time Steps (Days / Visits)")
    ax.set_ylabel("Predicted Flare Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, loc="upper left")

    fig.tight_layout()
    target = save_path or (OUT_DIR / "fig8_cf_dmard_withdrawal.png")
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Saved {target}")


def plot_fig9_subgroup_auroc(
    subgroups_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: Path | None = None
):
    """Figure 9: AUROC curves by patient subgroups (age and gender).
    Demonstrates consistent, equitable generalization across demographic strata.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))

    styles = {
        "Female": ("#e41a1c", "-"),
        "Male": ("#377eb8", "--"),
        "Younger (<50)": ("#4daf4a", "-."),
        "Older (>=50)": ("#984ea3", ":"),
    }

    for name, (y_true, y_pred) in subgroups_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        score = auc(fpr, tpr)
        color, ls = styles.get(name, ("#000000", "-"))
        ax.plot(fpr, tpr, color=color, linestyle=ls, lw=2.0, label=f"{name} (AUROC = {score:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.0)
    ax.set_title("FIGURE 9. AUROC Curves by Patient Subgroups", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, loc="lower right")

    fig.tight_layout()
    target = save_path or (OUT_DIR / "fig9_subgroup_auroc.png")
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Saved {target}")


def plot_fig10_attention_heatmap(
    feature_names: list[str],
    attention_matrix: np.ndarray,
    save_path: Path | None = None
):
    """Figure 10: Attention heatmap highlighting salient time steps and features preceding flare events."""
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(attention_matrix, cmap="YlOrRd", aspect="auto", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Attention Weight", rotation=270, labelpad=12)

    ax.set_title("FIGURE 10. Temporal Attention Heatmap Preceding Flare Event", fontweight="bold")
    ax.set_xlabel("Time Step (Visit Preceding Flare)")
    ax.set_ylabel("Clinical Features / Biomarkers")
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xticks(range(attention_matrix.shape[1]))
    ax.set_xticklabels([f"T{i+1}" for i in range(attention_matrix.shape[1])])

    fig.tight_layout()
    target = save_path or (OUT_DIR / "fig10_attention_heatmap.png")
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Saved {target}")


def plot_fig11_tsne_flare_outcome(
    embeddings: np.ndarray,
    flare_labels: np.ndarray,
    save_path: Path | None = None
):
    """Figure 11: t-SNE projection of latent representations colored by flare outcome."""
    tsne = TSNE(n_components=2, random_state=42, perplexity=35, max_iter=800)
    n = min(len(embeddings), 1200)
    idx = np.random.default_rng(42).choice(len(embeddings), n, replace=False)
    emb = tsne.fit_transform(embeddings[idx])
    y = flare_labels[idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(emb[y == 0, 0], emb[y == 0, 1], c="#1f77b4", alpha=0.6, s=18, label="No Flare")
    ax.scatter(emb[y == 1, 0], emb[y == 1, 1], c="#d62728", alpha=0.7, s=22, label="Flare Event")

    ax.set_title("FIGURE 11. t-SNE Projection of Latent Space Colored by Flare Outcome", fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    target = save_path or (OUT_DIR / "fig11_tsne_flare_outcome.png")
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Saved {target}")


def plot_fig13_ablation_performance_drop(
    ablation_drops: dict[str, tuple[float, float]],
    save_path: Path | None = None
):
    """Figure 13: Performance drop in AUROC and F1 after removing components.
    Components:
    - No Structural Encoder
    - No Counterfactual Engine
    - No Disentanglement
    - No Temporal Attention
    - No Uncertainty Module
    """
    labels = list(ablation_drops.keys())
    auroc_drops = [ablation_drops[k][0] for k in labels]
    f1_drops = [ablation_drops[k][1] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rects1 = ax.bar(x - width/2, auroc_drops, width, label="AUROC Drop", color="#3182bd")
    rects2 = ax.bar(x + width/2, f1_drops, width, label="F1-score Drop", color="#fd8d3c")

    ax.set_title("FIGURE 13. Performance Drop in AUROC and F1 After Removing Components", fontweight="bold")
    ax.set_ylabel("Performance Drop")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Label heights
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    target = save_path or (OUT_DIR / "fig13_ablation_drop.png")
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Saved {target}")

