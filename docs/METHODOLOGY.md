> Historical Stage I methodology. Some referenced files are absent in this checkout.
> For the current implementation and verified findings, see [PROJECT_AUDIT.md](PROJECT_AUDIT.md).

# Research methodology, objectives and novelty

*Academic research prototype. Not clinically validated. Not a substitute for professional medical judgment.*

## 1. Research question

> Given a patient's medical history up to their current visit, what clinically relevant outcome is likely
> to occur at their next visit, and how reliable is that prediction?

The project moves from "does this patient have the disease now?" (single-visit classification) to
"what will happen at the next visit?" (longitudinal prediction).

## 2. Objectives

1. Organise visit records chronologically per patient and engineer features that use only past information.
2. Predict one clearly defined next-visit outcome: **flare at the next record** (SLEDAI rise >= 4).
3. Compare a baseline (logistic regression) with Random Forest and gradient boosting (XGBoost).
4. Test whether **history** (changes between visits) improves prediction over a **static** model that sees
   only the current visit.
5. Validate correctly: patient-level splits, a forward-in-time check, calibration and confidence intervals.
6. Report prediction **reliability** (ensemble spread) and **explanations** (SHAP), and separate
   association from causation.
7. Deliver a Streamlit app that never uses or displays future records as if they were known.

## 3. Dataset check (done before any modelling)

| Requirement | GSE65391 | Verdict |
|---|---|---|
| Repeated visits per patient | 157 SLE patients, 923 records, up to 22 per patient | yes |
| Patient identifier | `subject` | yes |
| Time information | `cumulative_time` (days from first record), `days_since_last_visit`; no calendar dates | partly |
| Meaningful future outcome | SLEDAI at the next record | yes |
| Usable prediction points | 635 (patients with >= 3 records), flare rate 17.2 % | small |

Not supported by this dataset (so **not** attempted): hospital readmission, complications, abnormal-lab
prediction, glucose/blood-pressure/BMI trends. Nothing was fabricated; no synthetic data is used.

**Future-information columns found and excluded** (they describe the whole follow-up or are derived from it):
`visit_count` (patient's total visits), `days_between_diagnosis_and_last_visit` (span to the LAST visit),
`biopsy_history`, `nephritis_class`, `days_since_kidney_biopsy` (negative before the biopsy), the study's
train/test `set`, `batch`, and derived groupings (`disease_activity`, `treatment_lmm1`, ...).
A first version of the feature code also leaked through a default value that used the patient's total number
of visits; the unit tests caught it and it was fixed.

## 4. Method

**Target.** For visit *t* with a next record *t+1*: `y = 1` if `SLEDAI(t+1) - SLEDAI(t) >= 4`.
This is a proxy flare label derived from the disease-activity score, not an adjudicated clinical event.

**Features (87, longitudinal).** Current values (labs, treatments, SLEDAI items, age, time since diagnosis),
plus for 11 key variables: previous value, change since previous visit, rate of change per 30 days, mean of
the last 3 visits; plus history features (highest and mean SLEDAI so far, flares already observed,
visits since the last flare, consecutive rises, treatment changes, visits on oral steroids). The **static**
comparison model uses only the 31 current-visit features. Race and gender are not used by default.

**Leakage control.** Features of row *t* are computed only from records 1..*t*. `tests/test_no_leakage.py`
verifies (a) identical features when later visits are removed and (b) unchanged features when later visits
are overwritten with extreme values.

**Models.** Logistic regression (L2), Random Forest, XGBoost, plus two sanity baselines (average rate and
current-SLEDAI-only). Hyper-parameters are fixed and modest because of the small cohort. LSTM/GRU were
**not** used: about 110 patients with a usable next visit is too few for a sequence network to be reliable.

**Validation.**
- Patient-level hold-out (79 train / 32 test patients, no overlap, asserted in code).
- Model choice: repeated (3x) patient-grouped, outcome-stratified 5-fold CV on training patients only.
- Threshold and Platt calibration are fitted on out-of-fold training predictions.
- Forward-in-time check: train on early records (<= median record index) of training patients, test on later
  records of held-out patients. (There are no calendar dates, so "time" is patient time.)
- Research experiment: static vs longitudinal on **all** patients with 5x repeated grouped CV, and a paired
  patient-level bootstrap (1000x) for the difference.
- Metrics: AUROC, PR-AUC, precision, recall, F1, Brier score, expected calibration error, 95 % patient-bootstrap CIs.

**Uncertainty and the deployed model.** 60 models are trained on patient bootstrap resamples. A patient
already in the data is scored **only** by models that never saw them (out-of-bag); a new patient is scored by
all. The reported probability is the Platt-calibrated ensemble mean; the band is the 10th-90th percentile of
the models. This captures model uncertainty only.

**Explainability.** SHAP (linear or tree explainer, whichever matches the selected model), averaged over
up to 10 of the models that produced the prediction, summarised by type of information (recent trend, longer
history, treatment, current values, timing). SHAP describes what the model uses, **not** what would happen if
a measurement or treatment were changed.

## 5. Results (from `reports/metrics.json`, this run)

| Evaluation | AUROC | PR-AUC |
|---|---|---|
| Held-out patients (32 patients, 157 rows, 26 flares), longitudinal logistic regression | 0.65 (0.49-0.77) | 0.33 (0.15-0.53) |
| Held-out, best static model (Random Forest by CV rule) | 0.56 (0.40-0.71) | 0.25 |
| Held-out, chance | 0.50 | 0.17 |
| Out-of-bag over all 111 patients (deployed ensemble) | 0.63 (0.58-0.68) | 0.27 (0.21-0.35) |
| All patients, repeated grouped CV, logistic: static vs longitudinal | 0.67 vs 0.64 | 0.30 vs 0.26 |

- The held-out test set is small (26 flares); intervals are wide and the lower AUROC bound is about 0.5.
- Static minus longitudinal differences are within noise (paired intervals include 0 for random forest and
  XGBoost; for logistic regression the longitudinal model is slightly *worse*, AUROC difference -0.03,
  95 % CI -0.06 to 0.00).
- Platt scaling improved held-out calibration (ECE 0.09 -> 0.03; Brier 0.147 -> 0.132).
- The F1-optimal alert threshold flags most visits (out-of-bag recall 83 %, precision 21 % vs 17 % base rate),
  so it is a weak screening rule.
- Forward-in-time check (72 rows, 10 flares) is too small to interpret.

**Bottom line:** the pipeline works and is leak-audited, but next-visit flare is hard to predict from this
small cohort, and the data do not show that history improves accuracy.

## 6. Novelty - what can and cannot be claimed

A quick literature check (web search, not a systematic review) found existing work, so **"first SLE flare
prediction model" cannot be claimed**:
- FLAME, "A fair machine learning model to predict flares of systemic lupus erythematosus" (PMC12296391):
  EHR cohort of 28,433 adults, flare within 3 months of an outpatient visit defined as a SLEDAI-2K rise of
  >= 4 points over baseline, XGBoost and logistic regression on 12 months of prior data, AUROC about 0.66.
- Pediatric SLE flare prediction from gene-expression data and machine learning (NHSJS, 2026) - appears to
  use the same kind of public paediatric data (I only saw the title and listing).
- "Comparing two machine learning approaches in predicting lupus hospitalization using longitudinal data"
  (PMC9525268) - title seen only.
- Other flare-prediction abstracts (ACR, PAGE meeting, JMIR Formative Research 2025) exist.

Defensible contributions of this project:
1. A **leak-audited, patient-split next-visit pipeline** on a public SLE cohort with unit tests, including the
   specific leaking columns found in the public file.
2. A **direct, paired comparison of static vs longitudinal features** with patient-level intervals - and an
   honest negative/neutral finding in this cohort.
3. A single tool combining **out-of-bag per-patient prediction, a reliability band and trend-level
   explanation**, with an "insufficient history" rule and no display of future data as known.

Not claimed: improved accuracy over existing models, causal conclusions, or clinical utility.
A fuller novelty statement would need a systematic literature search.

## 7. Limitations

- Small, single-study, mostly female paediatric cohort; no external validation.
- Flare is a SLEDAI-based proxy; "next record" may skip visits (visit numbers have gaps in 70 patients) and
  the gap to the next record varies (median 84 days, up to 735).
- No calendar dates, so true chronological (by date) validation is not possible.
- Several features are noisy or missing (anti-dsDNA 30 %, albumin 20 %, creatinine 17 % missing); imputation
  is simple (median, inside each training fold).
- Treatment flags are observational and confounded by indication; SHAP and any "what-if" reading are
  associations only.
- Only SLE (no RA); the base paper's multimodal inputs (CRP, vitals, cytokines) are not in this file.
- Reliability band reflects model disagreement, not real-world error.

## 8. Future enhancements

- Add a second cohort (for example MIMIC-IV, after credentialing) for external validation and richer
  labs/vitals; add RA.
- Sequence models (GRU / the Temporal Causal Transformer from the base paper) once more patients are available.
- Pre-declared secondary outcomes (next-visit high disease activity, renal involvement) and a sensitivity
  analysis for the flare threshold and for consecutive-visit pairs only.
- Add the gene-expression modality that ships with GSE65391.
- Causal modules: now built as Stage II - see `docs/CAUSAL_MODEL.md` (the statements above describe Stage I only).
- Fairness analysis by sex and ancestry; conformal prediction for guaranteed-coverage uncertainty.
