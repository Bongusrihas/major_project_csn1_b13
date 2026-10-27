# Current implementation and evidence audit

This audit covers the active **Paper framework versus Proposed framework** comparison. It replaces the earlier LECG confirmation benchmark as the current application description.

## Scope and evidence

- Publication: [supplied paper](paper/reference.pdf), Sections III-IV and Algorithm 1.
- Implementation: [Architecture](CAUSAL_MODEL.md) and [Methodology](METHODOLOGY.md), checked against active source files.
- Evidence: [Measured results](MEASURED_RESULTS.md) and [JSON snapshot](results/current_comparison.json), generated from completed runs.

Only two methods are active. The proposed extension is TRB; LECG is disabled in both backbones. Earlier GRU, classical and LECG experiments are not current comparison rows.

The stored evaluation compares the same fitted backbone with and without validation-selected correction on identical held-out patients. Public and simulated cohorts are separate. The authors' published clinical scores are historical reference values, not our local baseline scores.

The result generator verifies training-artifact hashes before writing documentation. The snapshot records run IDs, report / prediction hashes, sample sizes, thresholds, all metrics, correction settings and paired bootstrap intervals. It contains no patient IDs or raw clinical records.

## Controls and remaining limits

| Concern | Current control | Remaining limit |
|---|---|---|
| Patient overlap | Disjoint patient groups and pairing assertions. | A small holdout is not external validation. |
| Scaling | Training-only fitted transforms. | Uploaded units and coding must match. |
| Future information | Causal attention and past-only history / pooling. | Measurement availability itself may reflect disease / clinical practice. |
| Graph | Statistical edges use training measurements only. | Prior edges and proxy correlations do not identify a true clinical DAG. |
| Neural choice | Validation-loss checkpoint and early stopping. | Validation is reused later. |
| Head fitting | Trees fit training targets; validation selects stopping / settings / strengths. | Training embeddings and predictions are in-sample, not cross-fitted. |
| Seed choice | Validation search followed by freeze; no test-based seed choice. | Objective optimizes Proposed, not each framework independently. |
| Calibration / thresholds | Fit on validation and fixed on test / uploads. | Separate calibration / nested validation would reduce reuse. |
| Uploaded labels | Separate from model inputs; unknown labels remain unscored. | New IDs alone cannot establish independent provenance. |
| Seen patients | Recognized training / validation IDs excluded from summaries by default. | Renaming IDs defeats that recognition. |
| Reproducibility | Frozen seeds, stored settings, content caches and hashes. | Hardware / library versions can still change results. |
| Interruption | Only sealed completed runs are reused. | Incomplete training restarts rather than resuming an epoch checkpoint. |

## Interpret the improvement responsibly

Generated tables identify unfavorable changes as well as gains, including calibration or F1 regressions when they occur. They also state whether each AUROC-change interval includes zero. These observations are part of the evidence, not exceptions to omit.

The interval is conditional on fixed fitted models, not retraining or population uncertainty. The holdout has also been inspected during project development; a fresh locked protocol and external cohort would be stronger confirmation.

Ranking improvement does not establish treatment benefit, correct treatment effects, prospective clinical utility, or complete uncertainty coverage of the combined predictor.

## Future validation

1. Independent real-cohort evaluation with matching inputs and a predeclared protocol.
2. Nested patient-grouped validation or separate calibration / head-selection patients.
3. Repeated paired training seeds without favorable test-based selection.
4. Ablations separating history groups, embeddings, trees and severity clipping.
5. Calibration of the final Proposed probability without test-label access.
6. Explanations and uncertainty covering both neural and correction stages.

These are future studies, not additional active methods or already measured gains.

## Refresh the documentation

Preparation calls `src.project.reporting.sync_results()`. It updates the README and methodology result blocks, full measured-results document and JSON snapshot from the selected completed runs.

To refresh existing results without retraining or opening Streamlit:

```shell
uv run python -m src.project.reporting
```

For an intentional rebuild with the same defaults and frozen seed:

```shell
uv run python run.py --update
```
