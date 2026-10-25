"""Train a validation-selected graph model on the local 54-feature simulation.

The original checkpoints are preserved. This is not the paper's clinical cohort.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from src.causal import dataset as D, train as T, graph as G
from src.models.config import CausalCfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output", default="models/causal_validated_model.pt")
    args = parser.parse_args()
    sd = D.load_multimodal_cohort()
    tr, va, te = [D.subset_dataset(sd, sd.splits[key]) for key in ("train", "val", "test")]
    prior = G.build_prior_graph(sd.names, sd.roles)
    statistical, _ = G.build_statistical_graph(np.where(tr.M > 0, tr.X, np.nan), tr.n_vis, sd.names, sd.roles)
    A = G.build_relation_tensor(G.synthesize_causal_dag(prior, statistical, lam=.6, threshold=.35), sd.names, sd.roles)
    model = T.train_graph_candidates(tr, va, A, cfg=CausalCfg(), epochs=args.epochs,
                                    seed=42, device=T.DEVICE, verbose=True)
    pred = T.predict_outcomes(model, te, mc_samples=0, device=T.DEVICE)
    metrics = T.calculate_metrics(te.y_flare[te.valid], pred["p"][te.valid], te.y_sev[te.valid], pred["sev"][te.valid])
    T.save_model(model, args.output, A, sd)
    report = {"dataset": "Locally generated simulation (52 inputs and two outcomes)", "normalization": "training-fitted scaler",
              "selected_method": model.selected_method, "validation_brier": model.candidate_validation_brier,
              "test_metrics": metrics, "training_history": model.candidate_training_history,
              "split_patients": {key: len(value) for key, value in sd.splits.items()}}
    path = Path("reports/validated_benchmark/deployed_model.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "training_history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
