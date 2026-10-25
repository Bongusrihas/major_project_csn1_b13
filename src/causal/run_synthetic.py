"""Full experiment on the SYNTHETIC cohort: model, baselines, ablations, graph recovery and
ground-truth counterfactual validation.   python -m src.causal.run_synthetic [--fast]"""
from __future__ import annotations

import argparse, json, time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.causal import dataset as D, graph as G, pipeline as P, synthetic as SYN, train as T
from src.causal.model import CausalCfg

OUT = Path("reports/causal")
OUT.mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)


def flat_feats(sd, X=None):
    X = sd.X if X is None else X
    N, Tn, V = X.shape
    prev = np.concatenate([np.zeros((N, 1, V)), X[:, :-1]], 1)
    cm = np.cumsum(X, 1) / np.arange(1, Tn + 1)[None, :, None]
    return np.concatenate([X, sd.M, prev, cm, sd.gap[..., None]], -1).reshape(N * Tn, -1)


def cf_effect_truth(sd_all, pos_global, t, new):
    f, c = SYN.true_effect(sd_all.truth, t, pos_global, new, n_mc=600)
    return f, c


def cf_validation(models, xgb, te, glob, sd_all, which):
    """Compare predicted vs TRUE effect of a treatment change on next-visit flare risk."""
    rows = {}
    for label, new, cond in which:
        p, t = np.where(te.valid & cond(te))
        if len(p) > 1500:
            sel = np.random.default_rng(0).choice(len(p), 1500, replace=False); p, t = p[sel], t[sel]
        f_true, c_true = cf_effect_truth(sd_all, glob[p], t, new)
        eff_true = c_true - f_true
        res = {"n_points": int(len(p)), "true_mean_effect": float(eff_true.mean()), "true_sd_effect": float(eff_true.std())}
        for name, m in models.items():
            r = T.counterfactual(m, te, p, t, {k: v for k, v in new.items()}, mc_samples=10)
            e = r["effect"]
            res[name] = dict(mean_effect=float(e.mean()), corr=float(pearsonr(e, eff_true)[0]) if e.std() > 0 else 0.0,
                             rmse=float(np.sqrt(((e - eff_true) ** 2).mean())),
                             sign_acc=float(((e > 0) == (eff_true > 0)).mean()))
        if xgb is not None:                                         # S-learner: change the treatment inputs, re-predict
            X2 = te.X.copy()
            for k, v in new.items():
                X2[p, t, te.names.index(k)] = D.treatment_value(te.scaler, k, v)
            pf = xgb.predict_proba(flat_feats(te).reshape(te.N, te.X.shape[1], -1)[p, t])[:, 1]
            pc = xgb.predict_proba(flat_feats(te, X2).reshape(te.N, te.X.shape[1], -1)[p, t])[:, 1]
            e = pc - pf
            res["XGBoost S-learner"] = dict(mean_effect=float(e.mean()), corr=float(pearsonr(e, eff_true)[0]) if e.std() > 0 else 0.0,
                                            rmse=float(np.sqrt(((e - eff_true) ** 2).mean())), sign_acc=float(((e > 0) == (eff_true > 0)).mean()))
        rows[label] = res
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fast", action="store_true"); a = ap.parse_args()
    EP = 4 if a.fast else 20
    seeds = [0] if a.fast else [0, 1, 2]
    sd = D.make_synthetic(); sp = sd.extra["split"]
    tr, va, te = (D.subset(sd, sp[k]) for k in ("train", "val", "test"))
    W, A, info = P.build_graph(sd, sp["train"])
    R = {"dataset": sd.name, "SYNTHETIC": True, "n_patients": sd.N, "n_train": len(sp["train"]),
         "n_val": len(sp["val"]), "n_test": len(sp["test"]), "epochs_max": EP}
    # --- graph recovery
    te_edges = SYN.TRUE_TREATMENT_EDGES
    R["graph"] = {"acyclic": info["acyclic"], "n_edges": info["n_edges"],
                  "prior_only": G.evaluate_treatment_edges(info["P"], sd.names, sd.roles, te_edges),
                  "stat_only": G.evaluate_treatment_edges(info["S"], sd.names, sd.roles, te_edges),
                  "fused": G.evaluate_treatment_edges(info["W"], sd.names, sd.roles, te_edges)}
    np.save(OUT / "graph_W.npy", W)
    # --- oracle reference: AUROC of the TRUE next-visit risk
    pp, tt = np.where(te.valid)
    f_true, _ = SYN.true_effect(sd.truth, tt * 0 + tt, sp["test"][pp], {}, n_mc_samples=300)
    R["oracle_auroc"] = float(roc_auc_score(te.y_flare[pp, tt], f_true))
    # --- main model (several seeds) + baselines + ablations
    cfgs = {"Full causal model": CausalCfg()}
    base = {"GRU (no graph, no causal parts)": CausalCfg(use_graph=False, use_cdi=False, use_cse=False, use_tct=False, use_mog=True),
            "Transformer (no graph, no gate, no CSE/CDI)": CausalCfg(use_graph=False, use_cdi=False, use_cse=False, use_gate=False)}
    abl = {"w/o graph encoder": CausalCfg(use_graph=False), "w/o CSE (counterfactual engine)": CausalCfg(use_cse=False),
           "w/o CDI (disentanglement)": CausalCfg(use_cdi=False), "w/o intervention gate": CausalCfg(use_gate=False),
           "w/o mixture UQ head": CausalCfg(use_mog=False)}
    trained, results = {}, {}
    def run(name, cfg, seed, mc=30):
        t0 = time.time(); print(f"== {name} seed {seed}", flush=True)
        m = T.fit(tr, va, A, cfg, epochs=EP, patience=4, seed=seed, verbose=False)
        pr = T.predict(m, te, mc_samples=mc); met = T.metrics(te, pr)
        met["train_s"] = time.time() - t0; met["epochs"] = len(m.training_history)
        print(f"   AUROC {met['auroc']:.3f} AUPRC {met['auprc']:.3f} sevMAE {met['sev_mae']:.3f} ({met['train_s']:.0f}s)", flush=True)
        return m, met
    for name, cfg in {**cfgs, **base, **abl}.items():
        ss = seeds if name in ("Full causal model", "GRU (no graph, no causal parts)") else [0]
        mets = []
        for s in ss:
            m, met = run(name, cfg, s); mets.append(met)
            if name not in trained:
                trained[name] = m
        results[name] = {k: ([x[k] for x in mets] if False else float(np.mean([x[k] for x in mets]))) for k in mets[0]}
        if len(mets) > 1:
            results[name]["_sd"] = {k: float(np.std([x[k] for x in mets])) for k in ("auroc", "auprc", "sev_mae")}
        R["models"] = results
        json.dump(R, open(OUT / "synthetic_metrics.json", "w"), indent=1)
        if name == "Full causal model":
            T.save_model(trained[name], "models/causal_synthetic.pt", A, sd)
    # --- XGBoost baseline (also the S-learner for counterfactual comparison)
    Xtr = flat_feats(tr); ytr = tr.y_flare.reshape(-1); mtr = tr.valid.reshape(-1)
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6, n_jobs=2, verbosity=0)
    xgb.fit(Xtr[mtr], ytr[mtr])
    pxg = xgb.predict_proba(flat_feats(te))[:, 1].reshape(te.N, -1)
    results["XGBoost (lag features)"] = dict(auroc=float(roc_auc_score(te.y_flare[te.valid], pxg[te.valid])))
    R["models"] = results
    # --- ground-truth counterfactual validation
    on_st = lambda s: s.raw[:, :, s.names.index("steroid_dose")] > 0
    on_im = lambda s: (s.raw[:, :, s.names.index("dmard")] + s.raw[:, :, s.names.index("biologic")]) > 0
    cmp = {"Full causal model": trained["Full causal model"],
           "GRU (no graph, no causal parts)": trained["GRU (no graph, no causal parts)"],
           "w/o graph encoder": trained["w/o graph encoder"], "w/o CSE (counterfactual engine)": trained["w/o CSE (counterfactual engine)"]}
    R["counterfactual"] = cf_validation(cmp, xgb, te, sp["test"], sd, [
        ("Stop steroids (on steroid -> 0 mg)", {"steroid_dose": 0.0}, on_st),
        ("Stop DMARD + biologic", {"dmard": 0.0, "biologic": 0.0}, on_im)])
    json.dump(R, open(OUT / "synthetic_metrics.json", "w"), indent=1)
    # --- uncertainty example & calibration data for plots
    pr = T.predict(trained["Full causal model"], te, mc_samples=30)
    np.savez(OUT / "synthetic_test_preds.npz", p=pr["p"], p_sd=pr["p_sd"], valid=te.valid, y=te.y_flare, sev=pr["sev"],
             sev_sd=pr["sev_sd"], ys=te.y_sev, sev_epi=pr["sev_epi"], sev_alea=pr["sev_alea"])
    print(json.dumps(R["counterfactual"], indent=1))


if __name__ == "__main__":
    main()
