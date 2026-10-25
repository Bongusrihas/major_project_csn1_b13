"""End-to-End Training and Evaluation Loop (Paper Algorithm 1, Section III.K & III.L).

Implements:
1. End-to-End training with Adam optimizer (lr=1e-4, batch_size=64) (Paper Section III.K.6).
2. Full composite loss function (Eq. 32):
   L = L_ELBO + lambda_TC * L_TC + lambda_CF * L_CF + lambda_FL * L_FL + lambda_CT * L_CT
3. Counterfactual Simulation Engine (CSE) forward simulation and consistency enforcement.
4. Uncertainty Quantification (UQ) via Monte Carlo dropout and Mixture of Gaussians (Eqs. 20-24).
5. Interpretability: Integrated Gradients (Eq. 26) and attention weight heatmaps.
6. Full GPU acceleration (NVIDIA CUDA) with automated device allocation and memory efficiency.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)

from src.causal.dataset import SequenceDataset, apply_scaler
from src.causal.model import (
    CausalCfg,
    CausalDeepModel,
    contrastive_trajectory_loss,
    focal_loss,
    hsic_estimator,
    maximum_mean_discrepancy,
    mog_negative_log_likelihood,
    total_correlation_penalty,
)

# Detect and configure device (GPU specifically)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device() -> torch.device:
    return DEVICE


def to_torch_batch(sd: SequenceDataset, indices: np.ndarray | None = None, device: torch.device = DEVICE) -> dict[str, torch.Tensor]:
    """Convert dataset rows to PyTorch tensors placed on the specified device."""
    idx = np.arange(sd.N) if indices is None else np.asarray(indices)
    T = sd.T

    # Boolean mask of valid visits for each patient
    pad = torch.as_tensor(np.arange(T)[None, :] < sd.n_vis[idx][:, None], dtype=torch.bool, device=device)

    return {
        "X": torch.as_tensor(sd.X[idx], dtype=torch.float32, device=device),
        "M": torch.as_tensor(sd.M[idx], dtype=torch.float32, device=device),
        "gap": torch.as_tensor(sd.gap[idx], dtype=torch.float32, device=device),
        "pad": pad,
        "yf": torch.as_tensor(sd.y_flare[idx], dtype=torch.float32, device=device),
        "ys": torch.as_tensor(sd.y_sev[idx], dtype=torch.float32, device=device),
        "valid": torch.as_tensor(sd.valid[idx], dtype=torch.bool, device=device),
        "nuisance": torch.as_tensor(sd.nuisance[idx], dtype=torch.float32, device=device),
        "subgroup": torch.as_tensor(sd.subgroup[idx], dtype=torch.int64, device=device),
        "n_vis": torch.as_tensor(sd.n_vis[idx], dtype=torch.int64, device=device),
    }


def crop_batch_time(batch: dict[str, torch.Tensor], row_indices: torch.Tensor) -> dict[str, torch.Tensor]:
    """Dynamically crop temporal dimension to max visit in batch to optimize GPU memory."""
    sub_batch = {k: v[row_indices] for k, v in batch.items()}
    max_t = int(sub_batch["n_vis"].max().item())
    for key in ("X", "M", "gap", "pad", "yf", "ys", "valid"):
        sub_batch[key] = sub_batch[key][:, :max_t]
    return sub_batch


def compute_composite_loss(
    model: CausalDeepModel,
    batch: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None
) -> tuple[torch.Tensor, dict[str, float]]:
    """Composite training loss function (Paper Section III.K, Eq. 32):

    L = L_ELBO + lambda_TC * L_TC + lambda_CF * L_CF + lambda_FL * L_FL + lambda_CT * L_CT
    """
    cfg = model.cfg
    w = weights or {
        "lambda_fl": 1.0,    # Focal loss for class imbalance
        "lambda_cf": 0.8,    # Counterfactual consistency
        "lambda_tc": 0.1,    # Total correlation penalty
        "lambda_ct": 0.05,   # Contrastive trajectory loss
        "lambda_cdi": 0.5,   # HSIC + MMD disentanglement
        "lambda_recon": 0.3, # Next-visit reconstruction (ELBO)
    }

    out = model(batch["X"], batch["M"], batch["gap"], batch["pad"], sample_noise=model.training)
    v = batch["valid"]
    v_count = v.sum().clamp_min(1.0)

    loss_dict = {}

    # 1. Focal Loss for flare-up prediction (Paper Eq. 30)
    l_fl = (focal_loss(out["flare"], batch["yf"]) * v).sum() / v_count
    loss_dict["loss_focal"] = float(l_fl.item())

    # 2. Disease Severity estimation (Mixture of Gaussians NLL, Paper Eq. 23)
    if cfg.use_mog:
        l_sev = (mog_negative_log_likelihood(out, batch["ys"]) * v).sum() / v_count
    else:
        l_sev = 0.5 * (((model.sev_mean(out) - batch["ys"]) ** 2) * v).sum() / v_count
    loss_dict["loss_sev"] = float(l_sev.item())

    # 3. ELBO Likelihood: Next-visit reconstruction (Paper Eq. 27)
    tgt = batch["X"][:, 1:]
    msk = batch["M"][:, 1:] * v[:, :-1, None]
    l_recon = (((out["recon"][:, :-1] - tgt) ** 2) * msk).sum() / msk.sum().clamp_min(1.0)
    loss_dict["loss_recon"] = float(l_recon.item())

    # 4. Total Correlation & KL penalties (Paper Eq. 27-28)
    l_elbo = l_recon + 0.3 * l_sev
    if cfg.use_cse and "lv_eps" in out:
        kl_eps = -0.5 * (1.0 + out["lv_eps"] - out["mu_eps"].pow(2) - out["lv_eps"].exp()).sum(dim=-1)
        l_kl = (kl_eps * v).sum() / v_count
        l_elbo = l_elbo + 0.01 * l_kl
        # Total Correlation factorizing latent codes (Eq. 28)
        l_tc = total_correlation_penalty(out["zc"][v])
        loss_dict["loss_tc"] = float(l_tc.item())
    else:
        l_tc = torch.tensor(0.0, device=batch["X"].device)

    # 5. Counterfactual Consistency Loss (Paper Eq. 29)
    # Generates prediction under identical and alternate factual noise
    if cfg.use_cse and "noise" in out:
        out_cf = model(
            batch["X"],
            batch["M"],
            batch["gap"],
            batch["pad"],
            noise=out["noise"].detach(),
            sample_noise=False
        )
        l_cf = (((torch.sigmoid(out["flare"]) - torch.sigmoid(out_cf["flare"])) ** 2) * v).sum() / v_count
        loss_dict["loss_cf"] = float(l_cf.item())
    else:
        l_cf = torch.tensor(0.0, device=batch["X"].device)

    # 6. CDI: HSIC + MMD Disentanglement (Paper Eq. 17-19)
    if cfg.use_cdi and out["zc"].shape[-1] > 0:
        valid_coords = v.nonzero()
        if len(valid_coords) > 16:
            rand_sel = valid_coords[(torch.randperm(len(valid_coords), device=batch["X"].device) if model.training else torch.arange(len(valid_coords), device=batch["X"].device))[:256]]
            zc_sample = out["zc"][rand_sel[:, 0], rand_sel[:, 1]]
            zs_sample = out["zs"][rand_sel[:, 0], rand_sel[:, 1]]
            nuis_sample = batch["nuisance"][rand_sel[:, 0]]
            sg_sample = batch["subgroup"][rand_sel[:, 0]]

            l_hsic = hsic_estimator(zc_sample, nuis_sample)
            l_mmd = maximum_mean_discrepancy(zc_sample[sg_sample == 0], zc_sample[sg_sample == 1])
            l_spur = ((model.nuisance_head(zs_sample) - nuis_sample) ** 2).mean()
            l_cdi = l_hsic + 0.2 * l_mmd + 0.1 * l_spur
            loss_dict["loss_cdi"] = float(l_cdi.item())
        else:
            l_cdi = torch.tensor(0.0, device=batch["X"].device)
    else:
        l_cdi = torch.tensor(0.0, device=batch["X"].device)

    # 7. Contrastive Trajectory Loss (Paper Eq. 31)
    if w.get("lambda_ct", 0.0) > 0:
        valid_coords = v.nonzero()
        if len(valid_coords) > 16:
            rand_sel = valid_coords[(torch.randperm(len(valid_coords), device=batch["X"].device) if model.training else torch.arange(len(valid_coords), device=batch["X"].device))[:256]]
            h_sample = out["h"][rand_sel[:, 0], rand_sel[:, 1]]
            y_sample = batch["yf"][rand_sel[:, 0], rand_sel[:, 1]]
            l_ct = contrastive_trajectory_loss(h_sample, y_sample, temperature=0.3)
            loss_dict["loss_ct"] = float(l_ct.item())
        else:
            l_ct = torch.tensor(0.0, device=batch["X"].device)
    else:
        l_ct = torch.tensor(0.0, device=batch["X"].device)

    # Full aggregated loss (Eq. 32)
    total_loss = (
        l_elbo
        + w["lambda_fl"] * l_fl
        + w["lambda_cf"] * l_cf
        + w["lambda_tc"] * l_tc
        + w["lambda_ct"] * l_ct
        + w["lambda_cdi"] * l_cdi
    )
    strength = getattr(cfg, "edge_sparsity", 0.0)
    if strength and hasattr(model.encoder, "edge_logits"):
        edges = model.encoder.A > 0
        penalty = torch.sigmoid(model.encoder.edge_logits)[edges].mean() if edges.any() else total_loss * 0
        total_loss = total_loss + strength * penalty
        loss_dict["loss_edge_sparsity"] = float(penalty.detach())
    loss_dict["total_loss"] = float(total_loss.item())

    return total_loss, loss_dict


def train_causal_model(
    sd_train: SequenceDataset,
    sd_val: SequenceDataset,
    A_relation: np.ndarray,
    cfg: CausalCfg | None = None,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    patience: int = 6,
    seed: int = 42,
    device: torch.device = DEVICE,
    verbose: bool = True,
    model: CausalDeepModel | None = None
) -> CausalDeepModel:
    """Full End-to-End training routine (Algorithm 1, Page 13 of the Paper)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = cfg or CausalCfg()
    n_nuisance = sd_train.nuisance.shape[1]

    if verbose:
        print(f"[Training] Initializing Causal Model on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
        print(f"[Training] Features V={sd_train.V}, Treatments={len(sd_train.treat_idx)}, Batches of {batch_size}, LR={learning_rate}")

    model = model if model is not None else CausalDeepModel(
        V=sd_train.V,
        treat_idx=sd_train.treat_idx,
        A=A_relation,
        cfg=cfg,
        n_nuisance=n_nuisance
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Pre-load batches onto GPU
    train_batch = to_torch_batch(sd_train, device=device)
    val_batch = to_torch_batch(sd_val, device=device)

    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0
    training_history = []

    N_tr = sd_train.N

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(N_tr, device=device)
        total_epoch_loss = 0.0
        n_batches = 0

        for b_start in range(0, N_tr, batch_size):
            b_indices = perm[b_start:b_start + batch_size]
            b = crop_batch_time(train_batch, b_indices)

            optimizer.zero_grad(set_to_none=True)
            loss, _ = compute_composite_loss(model, b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_epoch_loss / max(1, n_batches)

        # Validation
        model.eval()
        with torch.no_grad():
            # Bounded validation prevents a whole-cohort R-GCN allocation from
            # exhausting a laptop GPU's memory budget. Weight by valid targets.
            val_total, val_steps = 0.0, 0
            for start in range(0, sd_val.N, batch_size):
                vb = crop_batch_time(val_batch, torch.arange(start, min(start + batch_size, sd_val.N), device=device))
                v_loss, _ = compute_composite_loss(model, vb)
                steps = int(vb['valid'].sum().item())
                val_total += v_loss.item() * steps
                val_steps += steps
            val_loss = val_total / max(1, val_steps)

        epoch_time = time.time() - t0
        training_history.append({"epoch": ep, "train_loss": avg_train_loss, "val_loss": val_loss, "time": epoch_time})

        if verbose and (ep % 2 == 0 or ep == 1 or ep == epochs):
            gpu_mem = f", GPU Mem: {torch.cuda.memory_allocated() / 1e6:.1f}MB" if torch.cuda.is_available() else ""
            print(f"  Epoch {ep:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s{gpu_mem}", flush=True)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"[Training] Early stopping triggered at epoch {ep} (Best Val Loss: {best_val_loss:.4f})")
                break

    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    model.eval()
    model.training_history = training_history

    # Validation probability calibration via Platt scaling
    val_preds = predict_outcomes(model, sd_val, mc_samples=0, device=device)
    val_logits = val_preds["logit"][sd_val.valid]
    val_targets = sd_val.y_flare[sd_val.valid]

    if np.unique(val_targets).size == 2:
        platt_calibrator = LogisticRegression(C=1e3)
        platt_calibrator.fit(val_logits[:, None], val_targets)
        model.platt_calibrator = platt_calibrator

    return model


def train_graph_candidates(sd_train, sd_val, A_relation, cfg=None, **training_kwargs):
    """Select static / LECG / regularized LECG using validation Brier only.

    Test patients must never be passed into this function. This prevents making
    LECG the deployed default merely because it has a more complex architecture.
    """
    import copy
    base = cfg or CausalCfg()
    scores, histories = {}, {}
    selected, best = None, float("inf")
    for name, gated, dropout, sparsity in (
        ("static", False, 0.0, 0.0), ("lecg", True, 0.0, 0.0),
        ("regularized_lecg", True, 0.15, 0.01),
    ):
        candidate_cfg = copy.deepcopy(base)
        candidate_cfg.use_lecg = gated
        candidate_cfg.edge_dropout = dropout
        candidate_cfg.edge_sparsity = sparsity
        candidate = train_causal_model(sd_train, sd_val, A_relation, cfg=candidate_cfg, **training_kwargs)
        prediction = predict_outcomes(candidate, sd_val, mc_samples=0,
                                      device=next(candidate.parameters()).device)
        score = float(np.mean((prediction["p"][sd_val.valid] - sd_val.y_flare[sd_val.valid]) ** 2))
        scores[name] = score
        histories[name] = candidate.training_history
        if score < best:
            selected, best = candidate, score
            selected.selected_method = name
    selected.candidate_validation_brier = scores
    selected.candidate_training_history = histories
    return selected


# -----------------------------------------------------------------------------
# 4. Inference & Uncertainty Quantification (Paper Section III.I, Eqs. 20-24)
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_outcomes(
    model: CausalDeepModel,
    sd: SequenceDataset,
    patient_indices: np.ndarray | None = None,
    mc_samples: int = 30,
    device: torch.device = DEVICE,
    seed: int = 42,
    batch_size: int = 64
) -> dict[str, np.ndarray]:
    """Inference supporting Monte Carlo Dropout epistemic uncertainty (Eqs. 20-22)
    and Mixture of Gaussians aleatoric uncertainty (Eqs. 23-24).
    """
    indices = np.arange(sd.N) if patient_indices is None else np.asarray(patient_indices)
    if len(indices) > batch_size:
        parts = [predict_outcomes(model, sd, indices[start:start + batch_size], mc_samples,
                    device, seed + start, batch_size) for start in range(0, len(indices), batch_size)]
        return {key: None if parts[0][key] is None else np.concatenate([part[key] for part in parts], axis=0)
                for key in parts[0]}
    model.eval()
    batch = to_torch_batch(sd, patient_indices, device=device)

    if mc_samples <= 0:
        out = model(batch["X"], batch["M"], batch["gap"], batch["pad"], sample_noise=False)
        logits = out["flare"].cpu().numpy()
        platt = getattr(model, "platt_calibrator", None)
        calibrated_logits = logits if platt is None else logits * platt.coef_[0, 0] + platt.intercept_[0]
        p = 1.0 / (1.0 + np.exp(-np.clip(calibrated_logits, -80, 80)))
        sev_m = model.sev_mean(out).cpu().numpy()
        sev_v = model.sev_var(out).cpu().numpy()
        return {
            "logit": logits,
            "p": p,
            "p_sd": np.zeros_like(p),
            "p_lo": p,
            "p_hi": p,
            "sev": sev_m,
            "sev_var": sev_v,
            "sev_sd": np.sqrt(np.clip(sev_v, 0, None)),
            "att": out["att"].cpu().numpy() if out["att"] is not None else None,
            "zc": out["zc"].cpu().numpy(),
            "zs": out["zs"].cpu().numpy(),
        }

    # Monte Carlo Dropout sampling
    torch.manual_seed(seed)
    # Enable dropout modules only
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    mc_logits, mc_sevs, mc_sev_vars = [], [], []

    for _ in range(mc_samples):
        out = model(batch["X"], batch["M"], batch["gap"], batch["pad"], sample_noise=True)
        mc_logits.append(out["flare"].cpu().numpy())
        mc_sevs.append(model.sev_mean(out).cpu().numpy())
        mc_sev_vars.append(model.sev_var(out).cpu().numpy())

    model.eval()
    all_logits = np.stack(mc_logits, axis=0)  # (S, N, T)
    all_sevs = np.stack(mc_sevs, axis=0)      # (S, N, T)
    all_sev_vars = np.stack(mc_sev_vars, axis=0)

    # Platt recalibration if present
    platt = getattr(model, "platt_calibrator", None)
    if platt is not None:
        scaled_logits = all_logits * platt.coef_[0, 0] + platt.intercept_[0]
        all_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    else:
        all_probs = 1.0 / (1.0 + np.exp(-all_logits))

    # Epistemic & Aleatoric decomposition (Paper Section III.I)
    mean_p = all_probs.mean(axis=0)
    p_sd = all_probs.std(axis=0)
    p_lo = np.quantile(all_probs, 0.05, axis=0)
    p_hi = np.quantile(all_probs, 0.95, axis=0)

    mean_sev = all_sevs.mean(axis=0)
    sev_epistemic = all_sevs.var(axis=0)
    sev_aleatoric = all_sev_vars.mean(axis=0)
    total_sev_sd = np.sqrt(sev_epistemic + sev_aleatoric)

    return {
        "att": out["att"].cpu().numpy() if out["att"] is not None else None,
        "logit": all_logits.mean(axis=0),
        "p": mean_p,
        "p_sd": p_sd,
        "p_lo": p_lo,
        "p_hi": p_hi,
        "sev": mean_sev,
        "sev_epi": sev_epistemic,
        "sev_alea": sev_aleatoric,
        "sev_epistemic": sev_epistemic,
        "sev_aleatoric": sev_aleatoric,
        "sev_sd": total_sev_sd,
    }


# -----------------------------------------------------------------------------
# 5. Counterfactual Simulation (CSE, Paper Section III.E, Eqs. 7-9)
# -----------------------------------------------------------------------------
@torch.no_grad()
def simulate_counterfactual_intervention(
    model: CausalDeepModel,
    sd: SequenceDataset,
    patient_idx: int,
    visit_idx: int,
    intervention_dict: dict[str, float],
    mc_samples: int = 30,
    device: torch.device = DEVICE
) -> dict[str, Any]:
    """Simulate surgical intervention do(A_t = a') under the SCM (Eq. 9).

    Keeps factual trajectory 1..t, performs intervention at visit t,
    and observes counterfactual next-visit flare risk and severity.
    """
    model.eval()
    one_sd = SequenceDataset(
        name="single_patient",
        X=sd.X[[patient_idx]],
        M=sd.M[[patient_idx]],
        raw=sd.raw[[patient_idx]],
        gap=sd.gap[[patient_idx]],
        n_vis=sd.n_vis[[patient_idx]],
        y_flare=sd.y_flare[[patient_idx]],
        y_sev=sd.y_sev[[patient_idx]],
        valid=sd.valid[[patient_idx]],
        nuisance=sd.nuisance[[patient_idx]],
        subgroup=sd.subgroup[[patient_idx]],
        names=sd.names,
        roles=sd.roles,
        treat_idx=sd.treat_idx,
        patient_ids=[sd.patient_ids[patient_idx]],
    )
    batch = to_torch_batch(one_sd, device=device)

    if not 0 <= visit_idx < int(sd.n_vis[patient_idx]) or mc_samples < 1:
        raise ValueError("Valid visit_idx and positive mc_samples are required")
    X_cf = batch["X"].clone()
    for name, value in intervention_dict.items():
        if name not in sd.names or sd.names.index(name) not in sd.treat_idx:
            raise ValueError(f"Unknown treatment: {name}")
        j = sd.names.index(name)
        # UI and CLI supply raw units. Match preprocessing of this dataset.
        scaler = getattr(sd, "scaler", None) or sd.extra.get("scaler")
        if scaler is not None:
            template = sd.raw[patient_idx, visit_idx].copy()[None, :]
            template[0, j] = value
            scaled, _ = apply_scaler(template[None], sd.names, scaler)
            normalized = float(scaled[0, 0, j])
        elif sd.extra.get("normalization") in ("prefix", "legacy"):
            stop = int(sd.n_vis[patient_idx]) if sd.extra["normalization"] == "legacy" else visit_idx + 1
            history = sd.raw[patient_idx, :stop, j]
            observed = history[np.isfinite(history)]
            if not len(observed):
                raise ValueError(f"Cannot normalize unobserved treatment {name}")
            low, high = min(float(observed.min()), value), max(float(observed.max()), value)
            normalized = (value - low) / (high - low) if high - low > 1e-5 else 0.0
        else:
            raise ValueError("Dataset must retain its fitted scaler or normalization metadata for raw-unit interventions")
        X_cf[0, visit_idx, j] = normalized
    factual_p, scenario_p, factual_s, scenario_s = [], [], [], []
    platt = getattr(model, "platt_calibrator", None)
    def probability(out):
        logit = out["flare"][0, visit_idx].item()
        if platt is not None:
            logit = logit * platt.coef_[0, 0] + platt.intercept_[0]
        return float(1 / (1 + np.exp(-np.clip(logit, -80, 80))))
    # Preserve factual abducted latent variables for each paired scenario draw.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(42)
        for _ in range(mc_samples):
            factual = model(batch["X"], batch["M"], batch["gap"], batch["pad"], sample_noise=True)
            scenario = model(X_cf, batch["M"], batch["gap"], batch["pad"], sample_noise=True,
                             abducted_eps=factual.get("eps"))
            factual_p.append(probability(factual))
            scenario_p.append(probability(scenario))
            factual_s.append(model.sev_mean(factual)[0, visit_idx].item())
            scenario_s.append(model.sev_mean(scenario)[0, visit_idx].item())
    return {
        "p_factual": float(np.mean(factual_p)), "p_counterfactual": float(np.mean(scenario_p)),
        "flare_risk_delta": float(np.mean(np.array(scenario_p) - factual_p)),
        "flare_risk_delta_sd": float(np.std(np.array(scenario_p) - factual_p)),
        "sev_factual": float(np.mean(factual_s)), "sev_counterfactual": float(np.mean(scenario_s)),
        "sev_delta": float(np.mean(np.array(scenario_s) - factual_s)),
    }


# -----------------------------------------------------------------------------
# 6. Evaluation Metrics & Calibration (Tables 2, 3, 4)
# -----------------------------------------------------------------------------
from src.evaluation.metrics import calculate_metrics


# -----------------------------------------------------------------------------
# 7. Integrated Gradients Attribution (Paper Section III.J.2, Eq. 26)
# -----------------------------------------------------------------------------
def compute_integrated_gradients(
    model: CausalDeepModel,
    sd: SequenceDataset,
    patient_idx: int,
    visit_idx: int,
    steps: int = 40,
    device: torch.device = DEVICE
) -> np.ndarray:
    """Computes Integrated Gradients attribution for feature importance (Paper Eq. 26)."""
    model.eval()
    one_sd = SequenceDataset(
        name="single_patient",
        X=sd.X[[patient_idx]],
        M=sd.M[[patient_idx]],
        raw=sd.raw[[patient_idx]],
        gap=sd.gap[[patient_idx]],
        n_vis=sd.n_vis[[patient_idx]],
        y_flare=sd.y_flare[[patient_idx]],
        y_sev=sd.y_sev[[patient_idx]],
        valid=sd.valid[[patient_idx]],
        nuisance=sd.nuisance[[patient_idx]],
        subgroup=sd.subgroup[[patient_idx]],
        names=sd.names,
        roles=sd.roles,
        treat_idx=sd.treat_idx,
        patient_ids=[sd.patient_ids[patient_idx]],
    )
    batch = to_torch_batch(one_sd, device=device)
    x_input = batch["X"]
    baseline = torch.zeros_like(x_input)

    accumulated_grads = torch.zeros_like(x_input)

    for alpha in np.linspace(0.0, 1.0, steps):
        x_step = baseline + float(alpha) * (x_input - baseline)
        x_step.requires_grad_(True)

        out = model(x_step, batch["M"], batch["gap"], batch["pad"], sample_noise=False)
        target_logit = out["flare"][0, visit_idx]
        grad, = torch.autograd.grad(target_logit, x_step)
        accumulated_grads += grad.detach()

    # Path integral approximation
    integrated_grad = (x_input - baseline) * (accumulated_grads / steps)
    return integrated_grad[0, :visit_idx + 1].cpu().numpy()


# -----------------------------------------------------------------------------
# Compatibility Helpers & Model Serialization
# -----------------------------------------------------------------------------
fit = train_causal_model
predict = predict_outcomes
def counterfactual(model, sd, patient_idx, visit_idx, intervention_dict, mc_samples=30, device=None):
    """Compatibility API with batched paired scenarios for SCM verification."""
    device = device or next(model.parameters()).device
    patients, visits = np.broadcast_arrays(np.asarray(patient_idx), np.asarray(visit_idx))
    if patients.ndim == 0:
        result = simulate_counterfactual_intervention(model, sd, int(patients), int(visits),
                                                       intervention_dict, mc_samples, device)
        result["effect"] = result["flare_risk_delta"]
        return result
    if mc_samples < 1:
        raise ValueError("mc_samples must be positive")
    scaler = sd.scaler or sd.extra.get("scaler")
    if scaler is None:
        raise ValueError("Batched raw-unit scenarios require a fitted scaler")
    from src.causal.dataset import subset_dataset
    effects, factual_probs, scenario_probs = [], [], []
    model.eval()
    for start in range(0, patients.size, 32):
        pp, tt = patients.ravel()[start:start + 32], visits.ravel()[start:start + 32]
        part = subset_dataset(sd, pp)
        batch = to_torch_batch(part, device=device)
        raw_cf = part.raw.copy()
        for name, value in intervention_dict.items():
            j = sd.names.index(name)
            if j not in sd.treat_idx:
                raise ValueError(f"Not a treatment: {name}")
            raw_cf[np.arange(len(pp)), tt, j] = value
        changed, _ = apply_scaler(raw_cf, sd.names, scaler)
        X_cf = torch.as_tensor(changed, dtype=torch.float32, device=device)
        rows = torch.arange(len(pp), device=device)
        cols = torch.as_tensor(tt, dtype=torch.long, device=device)
        fact, scen = [], []
        with torch.no_grad():
            for _ in range(mc_samples):
                f = model(batch["X"], batch["M"], batch["gap"], batch["pad"], sample_noise=True)
                c = model(X_cf, batch["M"], batch["gap"], batch["pad"], sample_noise=True, abducted_eps=f.get("eps"))
                platt = getattr(model, "platt_calibrator", None)
                fl, cl = f["flare"][rows, cols], c["flare"][rows, cols]
                if platt is not None:
                    fl = fl * platt.coef_[0, 0] + platt.intercept_[0]
                    cl = cl * platt.coef_[0, 0] + platt.intercept_[0]
                fact.append(torch.sigmoid(fl).cpu().numpy())
                scen.append(torch.sigmoid(cl).cpu().numpy())
        fp, cp = np.mean(fact, axis=0), np.mean(scen, axis=0)
        effects.extend(cp - fp); factual_probs.extend(fp); scenario_probs.extend(cp)
    return {"effect": np.asarray(effects).reshape(patients.shape),
            "p_factual": np.asarray(factual_probs).reshape(patients.shape),
            "p_counterfactual": np.asarray(scenario_probs).reshape(patients.shape)}
integrated_gradients = compute_integrated_gradients
def metrics(sd, predictions):
    """Dataset-oriented compatibility helper for older experiment entrypoints."""
    valid = sd.valid
    result = calculate_metrics(sd.y_flare[valid], predictions["p"][valid],
                               sd.y_sev[valid], predictions["sev"][valid])
    result.update(sev_mae=result["mae"], sev_rmse=result["rmse"])
    return result


def attention_map(model: CausalDeepModel, sd: SequenceDataset, patient_idx: int = 0) -> np.ndarray | None:
    """Extract temporal attention map for a patient."""
    res = predict_outcomes(model, sd, patient_indices=np.array([patient_idx]), mc_samples=0)
    return res.get("att")


def save_model(model: CausalDeepModel, path: Path | str, A: np.ndarray, sd: SequenceDataset):
    """Serialize trained causal model and associated metadata."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state": model.state_dict(),
        "cfg": model.cfg,
        "A": A,
        "names": sd.names,
        "roles": sd.roles,
        "treat_idx": sd.treat_idx,
        "n_nuisance": sd.nuisance.shape[1],
        "platt": getattr(model, "platt_calibrator", None),
        "normalization": sd.extra.get("normalization"),
        "selected_method": getattr(model, "selected_method", None),
        "candidate_validation_brier": getattr(model, "candidate_validation_brier", None),
        "scaler": sd.scaler,
    }, p)


def load_model(path: Path | str, device: torch.device = DEVICE) -> CausalDeepModel:
    """Load serialized causal model."""
    data = torch.load(path, map_location=device, weights_only=False)
    m = CausalDeepModel(
        V=len(data["names"]),
        treat_idx=data["treat_idx"],
        A=data["A"],
        cfg=data["cfg"],
        n_nuisance=data["n_nuisance"]
    ).to(device)
    m.load_state_dict(data["state"], strict=False)
    if data.get("platt") is not None:
        m.platt_calibrator = data["platt"]
    m.eval()
    return m
