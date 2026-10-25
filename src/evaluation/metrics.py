"""Evaluation metrics for flare prediction and disease severity estimation.

Paper Reference:
    "Causal Representation Learning for Predicting Autoimmune Disease Progression
    From Longitudinal Multimodal Clinical Data" (IEEE Access, Vol. 13, 2025).
    Section IV, Tables 2, 3, and 4.

Key Metrics:
    - Binary Classification (Task 1): AUROC, AUPRC, F1-score, Expected Calibration Error (ECE), Brier Score.
    - Ordinal / Continuous Severity (Task 2): Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, roc_auc_score, root_mean_squared_error


def calculate_metrics(
    y_true_flare: np.ndarray,
    y_pred_flare: np.ndarray,
    y_true_sev: np.ndarray,
    y_pred_sev: np.ndarray,
    threshold: float = 0.5
) -> dict[str, float]:
    """Computes clinical discrimination, calibration, and regression error metrics."""
    auroc = float(roc_auc_score(y_true_flare, y_pred_flare)) if np.unique(y_true_flare).size == 2 else float("nan")
    auprc = float(average_precision_score(y_true_flare, y_pred_flare)) if np.any(y_true_flare) else float("nan")
    best_f1 = f1_score(y_true_flare, y_pred_flare >= threshold, zero_division=0)

    mae = float(mean_absolute_error(y_true_sev, y_pred_sev))
    rmse = float(root_mean_squared_error(y_true_sev, y_pred_sev))

    # Expected Calibration Error (ECE) over 10 confidence bins
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for b_lo, b_hi in zip(bins[:-1], bins[1:]):
        in_bin = (y_pred_flare >= b_lo) & ((y_pred_flare <= b_hi) if b_hi == 1 else (y_pred_flare < b_hi))
        if in_bin.sum() > 0:
            bin_acc = y_true_flare[in_bin].mean()
            bin_conf = y_pred_flare[in_bin].mean()
            ece += np.abs(bin_acc - bin_conf) * (in_bin.sum() / len(y_pred_flare))

    brier = float(np.mean((y_pred_flare - y_true_flare) ** 2))

    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "f1": round(best_f1, 4),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "ece": round(ece, 4),
        "brier": round(brier, 4),
    }
