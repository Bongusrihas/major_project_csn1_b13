# Major project implementation guide

The current project has exactly two methods:

1. **Paper framework:** structural R-GCN, MRTF, temporal causal transformer and flare/severity prediction heads.
2. **Proposed framework:** the same network plus a history-aware nonlinear correction trained on training patients and selected on validation patients.

Run `python run.py`. The launcher prepares the public and simulated datasets, trains both methods, creates the figures and 3D projections, prepares random-cohort tests and opens Streamlit. Subsequent runs use verified caches. Use `python run.py --update` to rebuild without deleting previous runs.

See [README.md](README.md) for the architecture, current measured scores, fixed defaults, dataset provenance and app sections.

The original paper's reported results and original graphics are in the Paper vs proposed section. Locally measured scores appear in Proposed framework; they compare methods on the same held-out patients. Published clinical results are not compared directly with simulated-cohort scores.

The previous UI and LECG guide are retained in `legacy/` for reference. They are not additional methods in the current app.
