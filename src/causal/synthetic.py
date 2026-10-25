"""SYNTHETIC autoimmune cohort with a KNOWN causal structure (structural causal model, SCM).

!!! EVERYTHING IN THIS FILE IS SIMULATED. Nothing here is real patient data. !!!

Why simulate? Real clinical data never shows what would have happened under a different
treatment, so a "counterfactual" model can not be checked on it. In a simulation we know the
true data-generating process, so we can compute the TRUE effect of an intervention and test
whether the counterfactual engine recovers it.

The structure follows the paper's Figure 2: an unobserved latent "immune dysregulation" state U
drives biomarkers, symptoms and disease activity; treatments suppress U; environmental triggers
raise it; doctors choose treatments from what they OBSERVE (confounding by indication).

Cohort size mirrors the paper's description (1,024 RA + 976 SLE patients, 6-32 visits each) but the
numbers are generated, not measured.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# (name, role). Roles drive the prior causal graph and the plots.
VARS = [
    ("crp", "lab"), ("esr", "lab"), ("dsdna", "immuno"), ("wbc", "lab"), ("platelets", "lab"),
    ("hemoglobin", "lab"), ("c3", "immuno"), ("creatinine", "lab"),
    ("il6", "immuno"), ("tnfa", "immuno"), ("cd4cd8", "immuno"), ("ana", "immuno"), ("rf", "immuno"),
    ("heart_rate", "vital"), ("sbp", "vital"), ("temperature", "vital"),
    ("fatigue", "symptom"), ("joint_swelling", "symptom"), ("rash", "symptom"),
    ("steroid_dose", "treatment"), ("dmard", "treatment"), ("biologic", "treatment"),
    ("hydroxychloroquine", "treatment"), ("nsaid", "treatment"),
    ("severity", "outcome"), ("flare", "outcome"),
]
NAMES = [v for v, _ in VARS]
ROLES = {v: r for v, r in VARS}
TREAT = [v for v, r in VARS if r == "treatment"]
LOG_VARS = {"crp", "esr", "dsdna", "il6", "creatinine"}      # skewed: modelled on log scale
STATIC = ["age", "female", "sle", "comorbidity", "smoker"]
NUISANCE = ["age", "female", "site0", "site1", "site2", "site3"]
STEROID_LEVELS = np.array([0.0, 5.0, 10.0, 20.0])            # mg/day

# --- true parameters of the SCM (used both to simulate and to compute true counterfactuals) ---
K_STEROID, K_BIOLOGIC, K_HCQ, K_DMARD = 1.5, 1.0, 0.35, 0.6
RHO45 = 0.6               # persistence of U per 45 days
SHOCK = 1.6               # effect of an environmental trigger on U
TAU, U0 = 0.5, -0.2      # flare = rise in U above TAU while U above U0 (gives ~14 % flares per visit)

# True direct edges (observed-variable level). Latent U -> every biomarker/symptom/outcome is implied.
TRUE_TREATMENT_EDGES = {("steroid_dose", "crp"), ("steroid_dose", "esr"), ("steroid_dose", "il6"),
                        ("steroid_dose", "wbc"), ("steroid_dose", "sbp"),
                        ("biologic", "tnfa"), ("nsaid", "fatigue"), ("nsaid", "joint_swelling")}
LATENT_CHILDREN = [v for v, r in VARS if r in ("lab", "immuno", "vital", "symptom", "outcome")
                   and v not in ("sbp",)]


@dataclass
class SimConfig:
    n_patients: int = 2000
    seed: int = 7
    t_max: int = 32
    n_sle: int = 976


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _equilibrium(mu, steroid_norm, biologic, hcq, sle, md):
    """Level U is pulled towards: patient baseline minus the treatment effects."""
    return mu - (K_STEROID * (1 - np.exp(-3.0 * steroid_norm)) + K_BIOLOGIC * biologic
                 + K_HCQ * hcq * sle + K_DMARD * md)


def flare_risk(U_t, md_t, mu, sle, treat, n_mc=400, seed=0):
    """TRUE probability of a flare at the next visit given the hidden state and a treatment vector.

    treat: dict with steroid_dose (mg), dmard, biologic, hydroxychloroquine, nsaid (arrays or scalars).
    Everything random at the next step (gap, noise, trigger) is averaged out by Monte Carlo.
    """
    rng = np.random.default_rng(seed)
    U_t = np.asarray(U_t, float)[:, None]
    n = U_t.shape[0]
    steroid_norm = np.broadcast_to(np.asarray(treat["steroid_dose"], float) / 20.0, (n,))[:, None]
    biologic = np.broadcast_to(np.asarray(treat["biologic"], float), (n,))[:, None]
    hcq = np.broadcast_to(np.asarray(treat["hydroxychloroquine"], float), (n,))[:, None]
    dmard = np.broadcast_to(np.asarray(treat["dmard"], float), (n,))[:, None]
    md_next = 0.75 * np.asarray(md_t, float)[:, None] + dmard
    eq = _equilibrium(np.asarray(mu)[:, None], steroid_norm, biologic, hcq, np.asarray(sle)[:, None], md_next)
    gap = np.clip(np.exp(rng.normal(np.log(45), 0.5, size=(n, n_mc))), 14, 180)
    rho = RHO45 ** (gap / 45.0)
    eps = rng.standard_normal((n, n_mc))
    p_trig = 0.06 + 0.02 * np.broadcast_to(np.asarray(treat.get("_smoker", 0.0), float), (n,))[:, None]
    trig = rng.random((n, n_mc)) < p_trig
    u_new = eq + rho * (U_t - eq) + np.sqrt(1 - rho ** 2) * 0.9 * eps + SHOCK * trig * rng.uniform(0.5, 1.0, (n, n_mc))
    flare = (u_new - U_t > TAU) & (u_new > U0)
    return flare.mean(axis=1)


def simulate(cfg: SimConfig = SimConfig()):
    """Simulate the cohort. Returns (data, truth).

    data  : what a researcher would observe (arrays N x T x V, masks, static covariates ...)
    truth : hidden quantities for EVALUATION ONLY (latent state, true counterfactual risks)
    """
    rng = np.random.default_rng(cfg.seed)
    N, T, V = cfg.n_patients, cfg.t_max, len(VARS)
    idx = {n: i for i, n in enumerate(NAMES)}

    sle = np.zeros(N); sle[: cfg.n_sle] = 1; rng.shuffle(sle)
    age = np.where(sle == 1, rng.normal(41.7, 12, N), rng.normal(53.1, 11, N)).clip(18, 85)
    female = (rng.random(N) < np.where(sle == 1, 0.849, 0.765)).astype(float)
    comorb = rng.poisson(1.0, N).clip(0, 4).astype(float)
    smoker = (rng.random(N) < 0.2).astype(float)
    site = rng.integers(0, 4, N)
    n_vis = np.clip(np.round(rng.normal(16.4, 5.9, N)), 6, T).astype(int)

    age_z = (age - 47) / 12
    mu = 0.25 * age_z + 0.4 * smoker + 0.12 * comorb + rng.normal(0, 0.5, N)   # patient baseline of U
    site_shift = rng.normal(0, 0.35, (4, V))                                    # batch / site effects

    # --- allocate
    X = np.full((N, T, V), np.nan)
    U = np.zeros((N, T)); MD = np.zeros((N, T)); E = np.zeros((N, T)); GAP = np.zeros((N, T))
    steroid_mg = np.zeros((N, T)); dm = np.zeros((N, T)); bio = np.zeros((N, T))
    hcq = np.zeros((N, T)); nsa = np.zeros((N, T))
    sev = np.zeros((N, T)); flare = np.zeros((N, T))

    # initial treatments
    steroid_lvl = rng.choice(4, N, p=[0.35, 0.25, 0.25, 0.15])
    dm_cur = (rng.random(N) < 0.35).astype(float)
    bio_cur = (rng.random(N) < 0.08).astype(float)
    hcq_cur = (rng.random(N) < np.where(sle == 1, 0.7, 0.2)).astype(float)
    md = dm_cur.copy()
    U_prev = mu + rng.normal(0, 0.8, N)

    obs_names = [n for n in NAMES if ROLES[n] in ("lab", "immuno", "vital")]
    for t in range(T):
        alive = t < n_vis
        gap = np.zeros(N) if t == 0 else np.clip(np.exp(rng.normal(np.log(45), 0.5, N)), 14, 180)
        GAP[:, t] = gap
        # --- latent dynamics (treatments chosen at visit t-1 act on U_t)
        if t == 0:
            U_t = U_prev
        else:
            s_norm = steroid_mg[:, t - 1] / 20.0
            md = 0.75 * md + dm[:, t - 1]
            eq = _equilibrium(mu, s_norm, bio[:, t - 1], hcq[:, t - 1], sle, md)
            rho = RHO45 ** (gap / 45.0)
            trig = (rng.random(N) < 0.06 + 0.02 * smoker)
            E[:, t] = trig
            U_t = eq + rho * (U_prev - eq) + np.sqrt(1 - rho ** 2) * 0.9 * rng.standard_normal(N) \
                + SHOCK * trig * rng.uniform(0.5, 1.0, N)
            flare[:, t] = ((U_t - U_prev > TAU) & (U_t > U0)).astype(float)
        MD[:, t] = md
        U[:, t] = U_t

        # --- measurements at visit t
        def load(sle_l, ra_l):
            return np.where(sle == 1, sle_l, ra_l)
        s_prev = steroid_mg[:, t - 1] / 20.0 if t > 0 else steroid_lvl * 0 + STEROID_LEVELS[steroid_lvl] / 20.0
        b_prev = bio[:, t - 1] if t > 0 else bio_cur
        sh = site_shift[site]                                   # (N, V)
        nz = lambda s=1.0: rng.normal(0, s, N)
        col = lambda n: sh[:, idx[n]]
        meas = {
            "crp": 0.9 * U_t - 0.8 * s_prev + col("crp") + nz(0.45),
            "esr": 0.8 * U_t - 0.3 * s_prev + col("esr") + nz(0.5),
            "dsdna": load(0.9, 0.15) * U_t + col("dsdna") + nz(0.6),
            "wbc": 0.3 * U_t + 0.9 * s_prev + col("wbc") + nz(0.6),
            "platelets": -0.3 * U_t + col("platelets") + nz(0.7),
            "hemoglobin": -0.45 * U_t - 0.1 * (1 - female) + col("hemoglobin") + nz(0.7),
            "c3": load(-0.75, -0.1) * U_t + col("c3") + nz(0.6),
            "creatinine": 0.3 * U_t + 0.15 * comorb / 2 + col("creatinine") + nz(0.6),
            "il6": 0.9 * U_t - 0.3 * s_prev + col("il6") + nz(0.6),
            "tnfa": 0.7 * U_t - 0.7 * b_prev + col("tnfa") + nz(0.6),
            "cd4cd8": -0.5 * U_t + col("cd4cd8") + nz(0.7),
            "ana": load(0.8, 0.2) * U_t + col("ana") + nz(0.7),
            "rf": load(0.2, 0.9) * U_t + col("rf") + nz(0.7),
            "heart_rate": 0.4 * U_t + 0.2 * (0.5 * U_t) + col("heart_rate") + nz(0.8),
            "sbp": 0.25 * age_z + 0.6 * s_prev + col("sbp") + nz(0.8),
            "temperature": 0.5 * U_t + col("temperature") + nz(0.8),
        }
        nsaid_prev = nsa[:, t - 1] if t > 0 else np.zeros(N)
        sym = {
            "fatigue": 0.7 * U_t - 0.25 * nsaid_prev + nz(0.6),
            "joint_swelling": load(0.5, 1.0) * U_t - 0.35 * nsaid_prev + nz(0.6),
            "rash": load(0.8, 0.05) * U_t + nz(0.7),
        }
        S_lat = 1.7 + 1.0 * U_t + 0.15 * rng.standard_normal(N)
        sev_t = np.clip(np.round(S_lat), 0, 4)
        sev[:, t] = sev_t

        # --- doctor's treatment decision, based on what is OBSERVED at visit t (confounding by indication)
        d = 0.45 * (sev_t - 1.7) / 1.1 + 0.30 * sym["joint_swelling"] + 0.25 * sym["fatigue"]
        d = d + rng.normal(0, 0.25, N)
        up = (d > 0.7) & (rng.random(N) < 0.5)
        down = (d < -0.2) & (rng.random(N) < 0.4)
        steroid_lvl = np.clip(steroid_lvl + up.astype(int) - down.astype(int), 0, 3)
        interrupt = rng.random(N) < 0.05                       # exogenous interruption (adherence, supply)
        steroid_lvl = np.where(interrupt, 0, steroid_lvl)
        steroid_mg[:, t] = STEROID_LEVELS[steroid_lvl]
        start_dm = (dm_cur == 0) & (d > 0.4) & (rng.random(N) < 0.15)
        stop_dm = (dm_cur == 1) & ((rng.random(N) > 0.95) | (rng.random(N) < 0.04))
        dm_cur = np.where(start_dm, 1.0, np.where(stop_dm, 0.0, dm_cur))
        start_b = (bio_cur == 0) & (d > 1.0) & (rng.random(N) < 0.07)
        stop_b = (bio_cur == 1) & ((rng.random(N) > 0.97) | (rng.random(N) < 0.03))
        bio_cur = np.where(start_b, 1.0, np.where(stop_b, 0.0, bio_cur))
        stop_h = (hcq_cur == 1) & (rng.random(N) > 0.97)
        hcq_cur = np.where(stop_h, 0.0, np.where((hcq_cur == 0) & (rng.random(N) < 0.02), 1.0, hcq_cur))
        nsaid_cur = (rng.random(N) < _sigmoid(-0.6 + 1.0 * sym["joint_swelling"])).astype(float)
        dm[:, t], bio[:, t], hcq[:, t], nsa[:, t] = dm_cur, bio_cur, hcq_cur, nsaid_cur

        # --- store observed matrix with informative missingness
        p_lab = _sigmoid(0.2 + 0.7 * (sev_t - 1.7))
        for n in obs_names:
            v = meas[n]
            if ROLES[n] == "vital":
                keep = rng.random(N) < 0.92
            elif ROLES[n] == "immuno" and n in ("il6", "tnfa", "cd4cd8", "ana", "rf"):
                keep = rng.random(N) < 0.45 * p_lab + 0.1
            else:
                keep = rng.random(N) < 0.55 + 0.4 * p_lab
            if n in LOG_VARS:
                v = np.exp(v * 0.7)                              # skewed raw scale
            X[:, t, idx[n]] = np.where(keep & alive, v, np.nan)
        for n in sym:
            X[:, t, idx[n]] = np.where(alive, sym[n], np.nan)
        X[:, t, idx["steroid_dose"]] = np.where(alive, steroid_mg[:, t], np.nan)
        X[:, t, idx["dmard"]] = np.where(alive, dm[:, t], np.nan)
        X[:, t, idx["biologic"]] = np.where(alive, bio[:, t], np.nan)
        X[:, t, idx["hydroxychloroquine"]] = np.where(alive, hcq[:, t], np.nan)
        X[:, t, idx["nsaid"]] = np.where(alive, nsa[:, t], np.nan)
        X[:, t, idx["severity"]] = np.where(alive, sev_t, np.nan)
        X[:, t, idx["flare"]] = np.where(alive, flare[:, t], np.nan)
        U_prev = U_t

    static = np.stack([age, female, sle, comorb, smoker], 1)
    nuis = np.concatenate([np.stack([age_z, female], 1), np.eye(4)[site]], 1)
    data = dict(X=X, n_vis=n_vis, gap=GAP, static=static, nuisance=nuis, site=site, names=NAMES,
                roles=[ROLES[n] for n in NAMES], static_names=STATIC, nuisance_names=NUISANCE,
                log_vars=sorted(LOG_VARS), is_sle=sle)
    truth = dict(U=U, MD=MD, mu=mu, sle=sle, smoker=smoker, flare=flare, severity=sev,
                 steroid=steroid_mg, dmard=dm, biologic=bio, hcq=hcq, nsaid=nsa, n_vis=n_vis)
    return data, truth


def true_effect(truth: dict, t_idx: np.ndarray, p_idx: np.ndarray, new_treat: dict, n_mc: int = 400, seed: int = 0):
    """TRUE next-visit flare risk (factual and under do(new treatment)) for patient-visit points.

    p_idx, t_idx: patient and visit indices. new_treat overrides treatments; unspecified ones stay factual.
    Returns (risk_factual, risk_counterfactual).
    """
    tr = {"steroid_dose": truth["steroid"][p_idx, t_idx], "dmard": truth["dmard"][p_idx, t_idx],
          "biologic": truth["biologic"][p_idx, t_idx], "hydroxychloroquine": truth["hcq"][p_idx, t_idx],
          "nsaid": truth["nsaid"][p_idx, t_idx], "_smoker": truth["smoker"][p_idx]}
    cf = dict(tr); cf.update(new_treat)
    args = (truth["U"][p_idx, t_idx], truth["MD"][p_idx, t_idx], truth["mu"][p_idx], truth["sle"][p_idx])
    return flare_risk(*args, tr, n_mc, seed), flare_risk(*args, cf, n_mc, seed)


if __name__ == "__main__":
    d, tr = simulate()
    X = d["X"]; nv = d["n_vis"]
    print("patients", X.shape[0], "SLE", int(d["is_sle"].sum()), "mean visits", nv.mean().round(1))
    m = np.arange(X.shape[1])[None, :] < nv[:, None]
    fl = tr["flare"][m]
    print("flare rate per visit %.3f" % fl.mean(), "| severity dist", np.bincount(tr["severity"][m].astype(int), minlength=5) / m.sum())
    print("missing in labs: %.2f" % np.isnan(X[:, :, 0][m]).mean())
