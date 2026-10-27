# Architecture: Paper framework and Proposed framework

This describes the **active code path**, replacing the earlier Stage II / LECG account. See [Methodology](METHODOLOGY.md) for the publication and [Measured results](MEASURED_RESULTS.md) for scores.

## 1. The two architectures

Paper is our local adaptation of Sharma and Kaur's framework. Proposed shares its fitted network and adds Temporal Residual Boosting (TRB). Each paired run has one neural checkpoint and one correction artifact.

```text
Available visits: normalized values X, observation masks M, transformed gaps g
                                |
Fixed typed graph -> 2-layer R-GCN, node width 32 + direct [X,M] skip
                                |
          visit code [z_c:32 | z_s:16]  (z_s -> nuisance objective)
                                |
                 z_c -> MRTF: windows 1 / 3 / 6 visits
                                |
                 2-layer, 4-head TCT -> h_t:64
                                |
       treatment-action path + noise posterior:16 -> shared trunk:64
                                |
          flare logit + 3-component severity mixture
                                |
         deterministic inference / validation Platt calibration
                                |
                                +------> PAPER OUTPUT
                                |
h_t + past-only history summaries + Paper margin / severity starting point
                                |
           TRB classifier + severity-residual regressor
                                |
   validation-selected strengths; corrected severity clipped to [0,4]
                                |
                                +------> PROPOSED OUTPUT
```

CDI and other auxiliary losses act during neural training. The intended causal / nuisance names are modeling objectives, not proof of identified clinical causes.

## 2. Dimensions and source files

`B` = patients per batch; `T` = padded visits; `V` = input variables. Public data use `V=29`, main simulation `V=52`, random cohorts `V=18`.

| Block | Shape / configuration | Implementation |
|---|---|---|
| Values / masks | `B x T x V` | `src/causal/dataset.py` |
| Gap / padding mask | `B x T`; gap = `log1p(days/30)` | dataset loaders / `to_torch_batch()` |
| Relation graph | `14 x (V+1) x (V+1)`, including latent node | `src/causal/graph.py` |
| R-GCN | Width 32, two layers | `src/encoders/rgcn_original.py` |
| Visit latent | `B x T x 48`, split into 32 + 16 | structural encoder |
| MRTF | `B x T x 64`, windows 1 / 3 / 6 | `src/temporal/mrtf.py` |
| TCT | Width 64, two layers, four heads, dropout 0.15 | `src/temporal/tct.py` |
| Noise posterior | 16 means and log-variances | `src/models/causal_deep_model.py` |
| Prediction trunk | Width 64 | `CausalDeepModel` |
| Flare | One logit per visit | `flare_head` |
| Severity | Three weights, means and log scales | `sev_head` |
| TRB input | `9V + 65`: 326 / 533 / 227 by cohort | `src/project/improvement.py` |

TCT positional embeddings support 128 visits; upload validation enforces that limit. Padding masks handle variable histories. The last visit remains a valid inference input even when its future target is unknown.

## 3. Shared neural blocks

### Fixed graph and R-GCN

`pair.build_graph()` combines a clinical prior and training-only statistical graph with prior weight `0.6` and threshold `0.35`. Normalized relation channels contain seven forward types plus reverse message types. The relation graph is fixed during neural fitting.

Measurements are embedded with their scaled value, observation mask and node identity. Two layers mix neighbors through relation-specific weights. Attention pooling over observed-variable nodes and a latent-node readout form the graph representation. A direct `[X,M]` skip path is added before projection to 48 dimensions.

Reverse messages and the skip path make this a graph-informed predictor, not proof that every learned dependency follows an identified clinical causal pathway.

### CDI, MRTF and TCT

The visit code splits into `z_c` (32) and `z_s` (16). Training penalizes dependence between `z_c` and nuisance covariates, subgroup distribution differences, and failure of `z_s` to encode nuisance values. The predictive temporal path uses `z_c`.

MRTF computes causal rolling means over 1 / 3 / 6 visits, projects them to width 64 and uses sigmoid fusion gates. The TCT adds position and gap embeddings, applies past-only attention with padding masks, and uses treatment / gap gating to interpolate new and previous representations. Its output `h_t` is the representation supplied to TRB.

### Noise / action path and heads

Training samples a damped Gaussian latent noise value; deterministic inference uses the posterior mean. A treatment-action projection is added to `h_t`, concatenated with the noise code and passed through a shared trunk.

Flare uses a logit. Severity uses a three-component Gaussian mixture; the point estimate is `sum_k pi_k * mu_k`, not an integer category. Current paired scores use deterministic inference and validation-fitted Platt scaling, not MC-dropout samples.

## 4. Actual local training loss

`src/causal/train.py::compute_composite_loss()` combines:

```text
L = L_reconstruction + 0.3 * L_severity_NLL + 0.01 * L_noise_KL
    + 1.0 * L_focal + 0.8 * L_factual_noise_consistency
    + 0.1 * L_correlation + 0.05 * L_contrastive + 0.5 * L_CDI
```

Terms depend on enabled modules. CDI combines HSIC, `0.2 * MMD` and `0.1 * nuisance-MSE`. CDI and contrastive calculations require enough valid visits and use at most 256 sampled training points; validation uses a deterministic subset.

Reconstruction targets next observed scaled measurements under their masks. TC uses squared off-diagonal latent correlations instead of full KL total correlation. The consistency approximation compares factual sampled-noise and mean-noise predictions rather than verified alternative treatment effects.

Adam uses learning rate and weight decay `1e-4`, gradient clipping at norm 2, maximum 30 epochs and patience six. The best validation-loss weights are restored before calibration / correction fitting.

## 5. The added Proposed stage

[`TemporalResidualBoosting`](../src/project/improvement.py) combines nine past-only groups per variable with `h_t` and time gaps. [Methodology, section 4](METHODOLOGY.md#4-the-new-implementation-temporal-residual-boosting) defines each group and the equations.

The calibrated Paper probability supplies the classifier base margin. Raw Paper severity supplies the regression residual starting point. These are two task heads inside **one Proposed framework**, fitted after the neural weights freeze; gradients do not return through the trees.

Classification blends margins, not probabilities. Severity adds a residual and clips to [0,4]. Validation chooses zero or nonzero strengths and may select different settings per task. Selected strengths, tree counts and configurations appear in [Measured results](MEASURED_RESULTS.md).

Neural attention / integrated gradients and mixture variance describe the backbone. The project has not established complete attribution or uncertainty coverage for the combined neural / tree predictor.

## 6. Differences from the publication

| Topic | Publication | Active local implementation |
|---|---|---|
| Clinical data | Authors' SLE / RA cohort, 2,000 patients, 54 features. | Public pediatric SLE plus generated cohorts; original records unavailable. |
| Preprocessing | GRU-D / alignment in III.A; carry-forward / domain imputation and patient min-max in IV.A. | Recorded gaps, training-fitted transforms, masks. |
| Discovery | Prior plus PC / kernel independence tests. | Prior plus partial-correlation constraints on training measurements. |
| Attention | DAG-path / temporal admissibility. | Lower-triangular temporal and padding masks; no full DAG-path mask. |
| MRTF | Several temporal resolutions. | Last 1 / 3 / 6 visits. |
| Causal objectives | Variational, counterfactual and invariance objectives. | Explicit approximations above, evaluated predictively. |
| Uncertainty | MC dropout and Gaussian mixture. | Mixture retained; paired scores are deterministic. |
| Our extension | TRB is not described. | One added history-aware correction stage. |

Source: [supplied PDF](paper/reference.pdf), III.C-L and Algorithm 1. These distinctions prevent a paper-inspired implementation from being mistaken for the authors' exact released model.

## 7. Code and artifacts

| Responsibility | Source |
|---|---|
| Paired training and evaluation | [`pair.py`](../src/project/pair.py) |
| Single TRB improvement | [`improvement.py`](../src/project/improvement.py) |
| Network / configuration | [`causal_deep_model.py`](../src/models/causal_deep_model.py), [`config.py`](../src/models/config.py) |
| Graph and transforms | [`graph.py`](../src/causal/graph.py), [`dataset.py`](../src/causal/dataset.py), [`data.py`](../src/project/data.py) |
| Fitting / losses | [`train.py`](../src/causal/train.py) |
| Cache orchestration | [`pipeline.py`](../src/project/pipeline.py) |
| Generated result docs | [`reporting.py`](../src/project/reporting.py) |
| Uploaded-record checks | [`custom_data.py`](../src/project/custom_data.py) |

A run saves `paper.pt`, `improved.joblib`, `results.json`, `predictions.csv`, `embeddings.npz`, figures / projections and an integrity manifest. `.pt` is shared; Proposed adds a correction artifact, not another trained Transformer.

`CausalCfg` still contains legacy LECG / ablation switches. Active `fit_candidate()` explicitly sets `use_lecg=False`. Presence of a class or old experiment file does not create an additional active method.
