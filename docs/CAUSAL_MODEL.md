# Stage II - Causal representation model (what is built, what is simplified, what the numbers say)

Code: `src/causal/` · Page in the app: `app/pages/1_Causal_model.py` · Reports: `reports/causal/`
Run: `python -m src.causal.run_synthetic` (≈1 h on 2 CPUs; `--fast` for a 10-minute check) and `python -m src.causal.run_real`.

## 1. Why there are two datasets
* **Real:** GSE65391 SLE cohort - 157 patients, 766 usable next-visit steps. Real, but **far too small for a deep model**,
  and nobody observed what *would have happened* under a different treatment, so counterfactuals cannot be verified.
* **SYNTHETIC (simulated):** `src/causal/synthetic.py`. 2,000 simulated SLE+RA patients (~16 visits) generated from a structural
  causal model with a **hidden immune state**, known treatment effects, confounding by indication (doctors treat the sicker),
  informative missingness and site effects. Because the true mechanism is known, we can compute the **true** effect of a
  treatment change and test whether the model finds it. **No synthetic number is a statement about real patients.**

## 2. Components: implemented / simplified / not implemented
| Paper component | Status | What the code does (file) |
|---|---|---|
| Causal graph `G = λ·G_prior + (1-λ)·G_stat` (Eq. 4) | **Simplified** | Curated pharmacology prior + latent-aware *partial-correlation* test (not full PC/HSIC); treatment edges kept only where both agree; a learned latent node for hidden immune state (`graph.py`, `pipeline.py`). Graph is a verified DAG. |
| R-GCN structural encoder (Eq. 6) | **Implemented** | 14 relation types (7 + inverses), dense message passing, 2 layers, plus a graph-free skip path (`model.py: RGCN, StructuralEncoder`). Graph is *within-visit*; time is handled by the temporal part. |
| MRTF multi-resolution fusion (Eqs. 13-15) | **Simplified** | Gated fusion of causal moving averages over the last 1, 3 and 6 **visits** (not daily/weekly/monthly windows). |
| Temporal Causal Transformer + intervention gating (Eqs. 10-12) | **Implemented (time-causal)** | Self-attention that sees only past visits; gate `g = σ(W_Δ·Δt + W_a·a_t + b)`. Mask is time-causal, not a learned DAG mask. Unit-tested that the future cannot leak (`tests/test_causal.py`). |
| Counterfactual Simulation Engine (Eqs. 7-9, 29) | **Simplified** | Abduction (posterior noise ε from history, damped), action (overwrite treatment nodes at visit t + direct action pathway), prediction (same ε). Consistency loss is a weak regulariser (sampled-ε vs mean-ε agreement). |
| Disentanglement + invariance CDI (Eqs. 16-19) | **Simplified** | Latent split `[z_causal, z_spurious]`; HSIC(z_causal, nuisance) and MMD across subgroups on minibatches; z_spurious must encode the nuisance. |
| ELBO / total correlation (Eqs. 27-28) | **Simplified** | Reconstruction + β·KL (β=2); the exact total-correlation estimator is *not* implemented. |
| Uncertainty: MC dropout + mixture-of-Gaussians (Eqs. 20-24) | **Implemented** | 30-40 MC-dropout passes (epistemic) + 2-component Gaussian mixture for severity (aleatoric); Platt-calibrated flare probability. |
| Focal loss, contrastive loss (Eqs. 30-31) | **Implemented** (contrastive is a small supervised-contrastive term, weight 0.02) | `model.py`. |
| Integrated gradients, attention maps (Eq. 26) | **Implemented** | `train.py`, shown in the app. |
| Multimodal imaging / text / genomics inputs | **Not implemented** | Only tabular labs, symptom flags and treatments. |

Targets: next-visit **flare** (SLEDAI rise ≥ 4 on real data; simulator definition on synthetic) and **severity 0-4**.

## 3. Results - SYNTHETIC cohort (test = 300 held-out simulated patients, never used for training)
Reference: even an oracle that knows the true hidden state reaches AUROC **0.774**.

| Model | Flare AUROC | AUPRC | Severity MAE |
|---|---|---|---|
| **Full causal model** (3 seeds) | **0.737 ± 0.003** | 0.283 | 0.600 |
| GRU baseline (3 seeds) | 0.711 ± 0.025 | 0.259 | 0.607 |
| Transformer (no graph/gate/CSE/CDI) | 0.686 | 0.226 | 0.589 |
| XGBoost with lag features | 0.741 | - | - |
| Ablation: w/o graph encoder | 0.713 | 0.257 | 0.615 |
| Ablation: w/o CSE | 0.739 | 0.284 | 0.601 |
| Ablation: w/o CDI | 0.722 | 0.262 | 0.597 |
| Ablation: w/o intervention gate | 0.734 | 0.287 | 0.594 |
| Ablation: w/o mixture head | 0.742 | 0.284 | 0.614 |

Honest reading: the full model is on par with XGBoost and better than the GRU/Transformer baselines, but **single-seed ablation
differences of ≈0.01-0.03 AUROC are within seed noise** (GRU alone varies ±0.025), so these ablations do **not** prove that
each component helps for prediction. Calibration is good (ECE ≈ 0.01); 90% severity intervals cover ≈ 92% (target 90%).

### Counterfactual check against the simulator's TRUE effects (the central test)
| What-if | True mean effect on next-visit flare risk | Model (full) corr / sign-acc | GRU | XGBoost S-learner |
|---|---|---|---|---|
| Stop steroids (n=1500 points) | +9.9 pts | mean +5.7 pts, corr **0.66**, sign 92% | 0.47, 79% | **0.71**, 99% |
| Stop DMARD + biologic | +4.8 pts | mean +7.7 pts, corr **0.40**, sign 95% | 0.29, 92% | 0.37, 97% |

Reading: the model recovers the *direction* and gives a useful ranking of who is most affected, but it **under-estimates
the steroid effect (5.7 vs 9.9 pts) and over-estimates the DMARD/biologic effect**, and a plain XGBoost with the same inputs
does about as well. We therefore do **not** claim that the causal architecture beats simple baselines at what-if estimation.
Graph recovery of treatment→biomarker edges: prior alone P 0.70 / R 0.88; statistics alone P 0.44 / R 1.00;
fused P 1.00 / R 0.88 (7 of 8 true edges).

### How we got here (transparency)
First complete run (`synthetic_metrics_v1_before_action_path.json`): the full model was the **worst** at what-if estimates
(corr 0.05) and AUROC 0.68. Cause: noise injection plus no direct treatment pathway. We added the action pathway and damped the
noise, retrained, and reran everything. **That design change was chosen after looking at test-set counterfactual results**, so
the test numbers above are mildly optimistic; baselines were not re-tuned. A cleaner protocol would pick the change on a
separate validation simulation.

## 4. Results - REAL SLE cohort (5-fold patient-grouped CV, one seed)
| Model | Flare AUROC (pooled out-of-fold, 95% patient-bootstrap CI) | Fold mean ± SD | Severity MAE |
|---|---|---|---|
| Causal model (small config) | 0.533 (0.48-0.59) | 0.54 ± 0.06 | 0.58 |
| GRU | 0.491 (0.42-0.56) | 0.48 ± 0.08 | 0.60 |
| Causal w/o graph | 0.511 (0.46-0.55) | 0.52 ± 0.06 | 0.64 |
| Logistic regression (flat lag features) | 0.600 (0.55-0.65) | 0.61 ± 0.04 | - |
| Stage I ensemble (out-of-bag, different protocol) | 0.628 | - | - |
| Predict-the-mean severity | - | - | 0.79 |

**On real data the deep causal model does not predict flares better than chance (CI includes 0.5) and is worse than the Stage I
classical models.** With 157 patients this is expected; it is not evidence that the idea fails, only that this dataset cannot
support it. Severity prediction beats the trivial baseline (0.58 vs 0.79), mostly because severity is persistent between visits.
The what-if tab on real patients is a demonstration of the mechanism only.

## 5. What you may claim / must not claim
May claim: the paper's architecture was implemented end-to-end (with listed simplifications); it was validated on a simulator with
known ground truth; it gives calibrated risks and uncertainty; honest comparison shows parity with strong baselines.
Must not claim: better than the paper or than XGBoost; accurate on real patients; clinically valid counterfactuals; that
synthetic results describe real patients; that ablations prove component value.
