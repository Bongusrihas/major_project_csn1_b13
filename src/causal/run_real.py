"""Causal model on the REAL SLE cohort (GSE65391), patient-grouped 5-fold CV, repeated over 2 seeds.
Counterfactual outputs on real data CANNOT be verified (no ground truth) and are not evaluated here."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from src.causal import dataset as D, pipeline as P, train as T
from src.causal.model import CausalCfg
from src.causal.run_synthetic import flat_feats

OUT = Path("reports/causal")
CFG = dict(d=16, d_c=16, d_s=6, d_h=32, n_heads=2, n_layers=1, dropout=0.25)


def main(epochs=60):
    sd = D.make_real("data/processed/visits_clean.csv")
    N = sd.N
    folds = list(GroupKFold(5).split(np.arange(N), groups=np.arange(N)))
    names = {"Causal model (full)": CausalCfg(**CFG),
             "GRU baseline": CausalCfg(**CFG, use_graph=False, use_cdi=False, use_cse=False, use_tct=False),
             "w/o graph encoder": CausalCfg(**CFG, use_graph=False)}
    oof = {n: np.full((N, sd.X.shape[1]), np.nan) for n in list(names) + ["Logistic regression"]}
    oof_sev = {n: np.full((N, sd.X.shape[1]), np.nan) for n in names}
    per_fold = {n: [] for n in oof}
    for seed in (0,):
        for k, (tri, tei) in enumerate(folds):
            rng = np.random.default_rng(seed * 10 + k)
            tri = rng.permutation(tri); nva = max(12, int(0.15 * len(tri)))
            va_i, tr_i = np.sort(tri[:nva]), np.sort(tri[nva:])
            D.finalize_real(sd, tr_i)
            W, A, _ = P.build_graph(sd, tr_i, keep=0.8)
            tr, va, te = D.subset(sd, tr_i), D.subset(sd, va_i), D.subset(sd, tei)
            for n, cfg in names.items():
                m = T.fit(tr, va, A, cfg, epochs=epochs, patience=10, batch_size=16, learning_rate=2e-3, seed=seed, verbose=False)
                pr = T.predict(m, te, mc_samples=20)
                oof[n][tei] = np.where(te.valid, pr["p"], np.nan); oof_sev[n][tei] = pr["sev"]
                per_fold[n].append(T.metrics(te, pr)["auroc"])
            Xf = flat_feats(tr); y = tr.y_flare.reshape(-1); mk = tr.valid.reshape(-1)
            lr = LogisticRegression(C=0.05, max_iter=2000).fit(Xf[mk], y[mk])
            p = lr.predict_proba(flat_feats(te))[:, 1].reshape(te.N, -1)
            oof["Logistic regression"][tei] = np.where(te.valid, p, np.nan)
            from sklearn.metrics import roc_auc_score
            per_fold["Logistic regression"].append(float(roc_auc_score(te.y_flare[te.valid], p[te.valid])))
            print("fold", k, {n: round(v[-1], 3) for n, v in per_fold.items()}, flush=True)
    from sklearn.metrics import average_precision_score, roc_auc_score
    v = sd.valid; y = sd.y_flare[v]
    res = {"dataset": sd.name, "n_patients": N, "n_valid_steps": int(v.sum()), "prevalence": float(y.mean()), "cv": "5-fold patient-grouped, seed 0",
           "note": "Small cohort: treat all numbers as exploratory. Counterfactuals on real data are unverified.", "models": {}}
    rng = np.random.default_rng(0)
    for n in oof:
        p = oof[n][v]
        pat = np.where(v)[0]
        boots = []
        for _ in range(300):                                            # patient-cluster bootstrap
            ps = rng.integers(0, N, N)
            idx = np.concatenate([np.where(v[i])[0] * 0 + 0 for i in []] or [np.array([], int)])
            mm = np.isin(np.where(v)[0], ps)
            rows = np.concatenate([np.where(v[i])[0] * 0 + i for i in ps])
            cols = np.concatenate([np.where(v[i])[0] for i in ps])
            yy, pp = sd.y_flare[rows, cols], oof[n][rows, cols]
            if 0 < yy.mean() < 1:
                boots.append(roc_auc_score(yy, pp))
        res["models"][n] = dict(auroc=float(roc_auc_score(y, p)), auroc_ci95=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                                auprc=float(average_precision_score(y, p)), fold_auroc_mean=float(np.mean(per_fold[n])),
                                fold_auroc_sd=float(np.std(per_fold[n])))
        if n in oof_sev:
            s = oof_sev[n][v]; ys = sd.y_sev[v]
            res["models"][n]["sev_mae"] = float(np.abs(s - ys).mean())
    res["models"]["Baseline: predict mean severity"] = {"sev_mae": float(np.abs(sd.y_sev[v] - sd.y_sev[v].mean()).mean())}
    json.dump(res, open(OUT / "real_metrics.json", "w"), indent=1)
    print(json.dumps(res["models"], indent=1))
    # final model on all patients (15% val) for the app demo
    rng = np.random.default_rng(1); perm = rng.permutation(N); va_i, tr_i = np.sort(perm[:24]), np.sort(perm[24:])
    D.finalize_real(sd, tr_i)
    W, A, _ = P.build_graph(sd, tr_i, keep=0.8); np.save(OUT / "graph_W_real.npy", W)
    m = T.fit(D.subset(sd, tr_i), D.subset(sd, va_i), A, CausalCfg(**CFG), epochs=epochs, patience=10, batch_size=16, seed=0, verbose=False)
    T.save_model(m, "models/causal_real.pt", A, sd)
    import joblib; joblib.dump({"scaler": sd.scaler, "train": tr_i, "val": va_i}, "models/causal_real_scaler.joblib")


if __name__ == "__main__":
    main()
