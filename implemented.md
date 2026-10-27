# Major project implementation guide

Exactly two methods are active:

1. **Paper framework:** a local paper-inspired R-GCN, MRTF and Temporal Causal Transformer with flare and severity heads.
2. **Proposed framework:** the same fitted backbone plus Temporal Residual Boosting using past-only values, missingness, timing and temporal embeddings.

The extension adds a prediction stage after the network. It does not replace the Transformer or change graph edges. LECG is disabled in both active backbones.

## Read the updated documentation

- [Methodology](docs/METHODOLOGY.md): paper design, motivation, correction equations, training stages and evaluation.
- [Architecture](docs/CAUSAL_MODEL.md): blocks, dimensions, code mapping and differences from the publication.
- [Measured results](docs/MEASURED_RESULTS.md): scores, gains, regressions, confidence intervals and run provenance.
- [Architecture sheet](docs/architecture_sheet.html): visual summary; also [PDF](docs/architecture_sheet.pdf) and [PNG](docs/architecture_sheet.png).
- [Audit](docs/PROJECT_AUDIT.md): controls, limitations and remaining validation.
- [README](README.md): installation, GPU setup and app instructions.

```shell
uv sync --locked
uv run python run.py --check-device
uv run python run.py
```

Preparation downloads / generates data as needed, trains both frameworks when required, generates curves and 3D projections and opens Streamlit. Completed work is cached; `--update` rebuilds with the frozen seed. The app compares public records, uploaded CSVs, manual visits and random cohorts with optional actual flare outcomes.

Local gains compare paired predictors on identical held-out patients. They do not establish superiority on the paper's unavailable clinical cohort. Earlier Stage I / Stage II accounts have been replaced; a nonexistent `legacy/` folder is not required.
