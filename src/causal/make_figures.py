"""Figure: predicted vs TRUE effect of stopping steroids on the SYNTHETIC test set."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.causal import dataset as D, synthetic as SYN, train as T

sd = D.make_synthetic(); sp = sd.extra["split"]; te = D.subset(sd, sp["test"])
m = T.load_model("models/causal_synthetic.pt")
st = te.names.index("steroid_dose")
p, t = np.where(te.valid & (te.raw[:, :, st] > 0))
sel = np.random.default_rng(0).choice(len(p), 800, replace=False); p, t = p[sel], t[sel]
f, c = SYN.true_effect(sd.truth, t, sp["test"][p], {"steroid_dose": 0.0}, n_mc=600)
e = T.counterfactual(m, te, p, t, {"steroid_dose": 0.0}, mc_samples=10)["effect"]
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.scatter(c - f, e, s=8, alpha=.5)
lim = [min((c - f).min(), e.min()), max((c - f).max(), e.max())]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("TRUE effect on next-visit flare risk (simulator)"); ax.set_ylabel("Model's predicted effect")
ax.set_title(f"SYNTHETIC data - stop steroids\ncorr = {np.corrcoef(c - f, e)[0, 1]:.2f}")
fig.tight_layout(); fig.savefig("reports/causal/cf_steroid_scatter.png", dpi=150)
