> Historical audit of the earlier LECG experiments. The current two-method TRB project and its latest measured results are described in [README.md](../README.md).

# Project audit and corrected evaluation — 25 October 2026

## Finding

The code is a useful research prototype, but the previous random-cohort comparison was invalid:
`run_multi_dataset_benchmark` constructed neural networks and evaluated them without any optimizer or training loop.
A split named `sd_train` was created but never used. Dropout remained active during the comparison.
The apparent difference therefore reflected initialization and sampling noise.

The current tests do not demonstrate improvement over the published paper. The corrected five-cohort confirmation
finds essentially equal discrimination for the local static baseline and LECG. Do not present a guaranteed gain,
causal edge recovery, or a statistically significant improvement.

## Paper and implementation boundary

Reference: Sharma & Kaur, [IEEE Access article](https://ieeexplore.ieee.org/document/11124198/),
DOI 10.1109/ACCESS.2025.3598276. The repository also contains `extracted_paper_text.txt`.
The paper discusses sensitivity to a misspecified causal graph in its limitations, supporting an edge-robustness experiment.
That motivation does not prove that scalar gates will recover the correct graph.

The paper reports a real clinical SLE/RA cohort and 54 input dimensions. The local generator produces simulated data:
its feature list has **52 inputs** (14 laboratory + 14 immunological + 16 demographic/vital + 8 treatment),
plus two outcomes. The existing “54-dim” name does not mean an exact recreation of the paper's input schema.
The paper's original patient-level data, original author checkpoints and verified reproduction are not available here.
Public GEO data are a separate cohort, and the real GSE65391 processed CSV is missing from this checkout.

The graph builder uses a partial-correlation heuristic with a latent proxy; this is not a complete PC or FCI implementation.
Temporal attention enforces past-only sequence attention; its attention weights are not formal causal explanations.
The prior graph and its relation tensor are assumptions, not known biological truth.
The synthetic disease labels identify simulation profiles; they are not validated disease-specific simulators.

## Method changes

Four methods are now evaluated on the identical simulated cohorts:

1. Local paper-inspired static R-GCN, MRTF and temporal transformer.
2. Original LECG: effective adjacency `A * sigmoid(theta)`.
3. Regularized LECG: training-only independent edge dropout at probability 0.15 and
   `0.01 * mean(sigmoid(theta_existing_edges))` added to the shared loss.
4. Small-data history alternative: logistic regression with L2 regularization for flare, ridge regression for severity;
   current values, running history means, changes since the preceding visit and observation masks.

The graph models start from identical shared parameters, see the same patient splits and train with the same optimizer,
learning rate, maximum epochs and early-stopping rule. The regularized variant changes only its stated regularization.
Dropout and abduction sampling are disabled for deterministic evaluation. Validation penalties use a fixed subset.
Gates are learned predictive weights, not calibrated probabilities of causal edge validity. Existing gates near 0.90
have small sigmoid derivatives; the direct input skip path can also bypass graph messages, making gating differences small.
The implementation currently allocates a dense adjacency-shaped logit tensor; it does not store only |E| scalars.

Edge dropout is an existing idea: [Rong et al., DropEdge, ICLR 2020](https://openreview.net/forum?id=Hkx1qkrKPr).
This regularized LECG variant is an adaptation to test, not a claim of a newly invented causal learning method.
It cannot add missing edges and cannot identify treatment effects from observational data without additional assumptions.

## Corrected evaluation

- Reproducible independent simulation seeds; shuffled patient-level 60/20/20 train/validation/test splits.
- Feature normalization fitted only on observed training values and applied unchanged to validation/test patients.
- Missing values remain masked and their normalized inputs are zero.
- Targets are shifted to the actual next simulated visit; the last visit has no scored next-visit label.
- No injected edges when noise=0. Noise adds relation-message edges; it is a graph misspecification stress test,
  not an intervention on the data-generating mechanism or a guarantee of a medically valid new DAG.
- Models really train before scoring. Initial AUROC is retained as a reference; the history initialization is a constant-rate baseline.
- Best neural checkpoint and Platt calibration use validation patients only.
- F1 thresholds for this benchmark are chosen using validation labels only. The shared metric utility otherwise uses a fixed threshold of 0.5.
- Candidate selection uses lowest validation Brier score, never test AUROC. Selection can still worsen test AUROC.
- Single-class AUROC is reported as undefined, rather than silently replaced with 0.5.
- ECE includes predictions equal to 1.0.
- Paired patient bootstrap resamples whole patients within each simulation cohort, preserving model pairing.
  Intervals are conditional on these fitted models and cohorts; they do not capture all training or population uncertainty.

These precautions follow the [scikit-learn guidance on leakage](https://scikit-learn.org/stable/common_pitfalls.html).
Validation is reused for stopping, calibration and selection; a larger study should add nested grouped validation or
an independent calibration partition. These runs are exploratory research, not external clinical validation.

## Measured confirmation results

Five cohorts, 200 patients per cohort, six visits per patient, seeds 4200 + 17*k,
maximum 30 neural training epochs. Conditions use the same underlying cohorts and patient splits.

| Method | Clean graph mean test AUROC | 20% added edges mean test AUROC |
|---|---:|---:|
| Static R-GCN | 0.66820 | 0.66800 |
| LECG | 0.66814 | 0.66750 |
| Regularized LECG | 0.66380 | 0.66506 |
| History logistic/ridge | 0.58216 | 0.58216 |
| Validation-selected method | 0.66544 | 0.66666 |

For LECG minus static under added edges, mean AUROC difference = -0.00050;
95% paired patient bootstrap interval = [-0.00190, 0.00117], 500 resamples.
The regularized candidate's corresponding interval also includes zero.
The simpler history method was worse on these confirmation cohorts.
A smaller exploratory run at seeds 42/59/76 initially favored the history method; the fresh-seed confirmation
reversed that observation. This is why model choice must not be based on a favorable test run.

![Verified synthetic comparison](../reports/validated_benchmark/verified_comparison.png)

Full metrics, initial scores, thresholds, epoch counts, runtimes, predictions and split audits:
`reports/validated_benchmark/confirmation_clean/` and `confirmation_noisy/`.
The earlier three-cohort exploratory run is in the parent directory.
No thresholds, cohort seeds or method hyperparameters were adjusted after examining the confirmation results.

## Future training and deployed model

`train_graph_candidates` trains static, LECG and regularized LECG and returns the lowest-validation-Brier model.
`run_experiments` now uses this selection, retains negative ablation differences and scores clinical proxy baselines
on the held-out patients. Those proxy baselines are not validated implementations of clinical SLEDAI or DAS28 scores.
`run_lecg_experiment` selects its saved seed using validation, rather than test AUROC.
Old reports are retained as historical outputs and marked in the UI.

`python -m src.causal.train_validated --epochs 12` saves a separate `models/causal_validated_model.pt` and
`reports/validated_benchmark/deployed_model.json`. It does not overwrite original model checkpoints.
New models use training-fitted scaling, persist the scaler and calibrator, and the explorer prefers the new checkpoint.
Custom inference uses the same scaler and feature schema; omitted input features remain missing.
Legacy checkpoints retain full-patient normalization only for compatibility and show a warning. They require retraining
before their predictions can support a valid prospective evaluation.
Prefix min-max preprocessing is also available and tested to ignore future visits; training-fitted scaling is preferred
because per-patient normalization can erase constant features such as age and treatment history.

Counterfactual scenario code now normalizes raw treatment units and preserves the factual abducted latent vector
for paired scenario predictions. A no-change scenario has zero effect in regression tests. These are model sensitivity
scenarios, not established clinical causal treatment effects; the learned model is not a fully identified clinical SCM.

## Streamlit and implementation performance

- The landing page no longer imports PyTorch just to display hardware status.
- Model explorer renders one selected analysis section, rather than executing every tab on each rerun.
- Ordinary predictions use one deterministic pass. Twenty uncertainty passes are optional.
- Integrated gradients run on demand; `autograd.grad` avoids accumulating unused parameter gradients.
- Treatment scenarios run on an explicit button; expensive 3D extraction runs only in that section and is cached.
- Model resources and prediction cache keys include checkpoint/data file versions.
- Cohort loading uses pre-grouped patients instead of repeatedly filtering the entire visit table.
- Benchmark inputs are in a form; training starts only on submission, and completed results persist in session state.
- Previews refer to the exact generated cohort, not a different small regenerated cohort or changed slider settings.
- The comparison chart includes all methods, initial scores and the full AUROC axis; CSV export retains details.
- The comparison page can load the completed verification output without retraining.

Final Streamlit AppTest measurements: cached reruns were 0.011 seconds for the landing page, 0.067 seconds for the model explorer, and 0.116 seconds for the comparison page on this machine. Cold model startup was 14.64 seconds; this includes runtime/data/model loading. These are test-runner timings, not a browser rendering or universal latency guarantee. Raw measurements are in `reports/validated_benchmark/ui_smoke.json`. No universal sub-10-ms or 100x acceleration claim is made.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m src.synthetic.generator --datasets 5 --patients 200 --epochs 30 --seed 4200 --noise 0.2 --output reports/validated_benchmark/confirmation_noisy
.\.venv\Scripts\python.exe -m src.synthetic.generator --datasets 5 --patients 200 --epochs 30 --seed 4200 --noise 0 --output reports/validated_benchmark/confirmation_clean
.\.venv\Scripts\python.exe -m src.causal.train_validated --epochs 12
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Install requirements in a compatible Python environment if `.venv` does not exist.
The fresh app checkpoint selected regularized LECG by validation Brier (0.149804 versus 0.149817 for static),
a tiny difference that does not demonstrate superiority. On 300 held-out simulated patients its AUROC was 0.5380,
AUPRC 0.1915, Brier 0.1470 and severity MAE 0.626. Its F1 at the fixed 0.5 threshold was 0: the calibrated
probabilities did not exceed that cutoff. This is weak prediction and must not be represented as a successful clinical model.
The checkpoint and app are working, but improving predictive accuracy requires better signal and independent validation.

**Verification: 24 passed, 2 skipped.** All three Streamlit pages rendered without exceptions; comparison-form submission, persisted results, uncertainty sampling and the attribution/attention/graph/results sections also passed AppTest smoke checks.

The two custom-template tests are skipped because the sample CSV is absent. New regression tests exercise custom
preprocessing, training-only scaling, future invariance, threshold leakage, real training, split disjointness,
paired interventions, validation selection and checkpoint round trips.
