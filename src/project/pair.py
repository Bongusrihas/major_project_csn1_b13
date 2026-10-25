"""Train and evaluate exactly two project methods on identical patient splits."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import f1_score

from src.causal import dataset as D, train as T, graph as G
from src.models.config import CausalCfg
from src.project.improvement import TemporalResidualBoosting
from src.project.settings import METHODS, EPOCHS, BATCH_SIZE, LEARNING_RATE, SEED, training_seed


@torch.no_grad()
def infer(model, sd, batch_size=32):
    model.eval()
    probabilities, severities, representations = [], [], []
    device = next(model.parameters()).device
    for start in range(0, sd.N, batch_size):
        batch = T.to_torch_batch(sd, np.arange(start, min(start + batch_size, sd.N)), device=device)
        output = model(batch['X'], batch['M'], batch['gap'], batch['pad'], sample_noise=False)
        logits = output['flare'].cpu().numpy()
        platt = getattr(model, 'platt_calibrator', None)
        if platt is not None:
            logits = logits * platt.coef_[0, 0] + platt.intercept_[0]
        probabilities.append(expit(logits))
        severities.append(model.sev_mean(output).cpu().numpy())
        representations.append(output['h'].cpu().numpy())
    return {'p': np.concatenate(probabilities), 'sev': np.concatenate(severities)}, np.concatenate(representations)


def threshold(y, probability):
    return float(max(np.linspace(.05, .95, 37), key=lambda value: f1_score(y, probability >= value, zero_division=0)))


def build_graph(sd, train):
    prior = G.build_prior_graph(sd.names, sd.roles)
    statistical, _ = G.build_statistical_graph(np.where(train.M > 0, train.X, np.nan), train.n_vis, sd.names, sd.roles)
    return G.build_relation_tensor(G.synthesize_causal_dag(prior, statistical, lam=.6, threshold=.35), sd.names, sd.roles)


def patient_bootstrap(frame, n_boot=500, seed=42):
    """Paired patient resampling, conditional on fitted methods and this cohort."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    patients = frame.patient_id.unique()
    groups = {patient: np.flatnonzero(frame.patient_id.to_numpy() == patient) for patient in patients}
    deltas = []
    y = frame.actual_flare.to_numpy()
    base = frame.paper_risk.to_numpy()
    improved = frame.improved_risk.to_numpy()
    for _ in range(n_boot):
        ix = np.concatenate([groups[patient] for patient in rng.choice(patients, len(patients), replace=True)])
        if np.unique(y[ix]).size == 2:
            deltas.append(roc_auc_score(y[ix], improved[ix]) - roc_auc_score(y[ix], base[ix]))
    if not deltas:
        return {'lower': None, 'upper': None, 'draws': 0}
    return {'lower': float(np.quantile(deltas, .025)), 'upper': float(np.quantile(deltas, .975)), 'draws': len(deltas)}


def fit_candidate(train, val, adjacency, epochs=EPOCHS, seed=SEED, progress=print, device=None):
    """Fit/select using training and validation patients only, with no test input."""
    model = T.train_causal_model(train, val, adjacency, cfg=CausalCfg(use_lecg=False),
        epochs=epochs, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seed=seed,
        device=device or T.DEVICE, verbose=True)
    progress('Learning the history-aware residual correction')
    train_base, train_h = infer(model, train)
    val_base, val_h = infer(model, val)
    improvement = TemporalResidualBoosting().fit(train, val, train_base, val_base, train_h, val_h, seed)
    val_improved = improvement.predict(val, val_base, val_h)
    validation, thresholds = {}, {}
    for key, predictions in (('paper', val_base), ('improved', val_improved)):
        thresholds[key] = threshold(val.y_flare[val.valid], predictions['p'][val.valid])
        validation[key] = T.calculate_metrics(val.y_flare[val.valid], predictions['p'][val.valid],
            val.y_sev[val.valid], predictions['sev'][val.valid], thresholds[key])
    return model, improvement, thresholds, validation


def train_pair(sd, folder, epochs=EPOCHS, seed=None, adjacency=None, progress=print, device=None):
    seed = training_seed() if seed is None else seed
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    train, val, test = [D.subset_dataset(sd, sd.splits[key]) for key in ('train', 'val', 'test')]
    assert not (set(train.patient_ids) & set(val.patient_ids) or set(train.patient_ids) & set(test.patient_ids) or set(val.patient_ids) & set(test.patient_ids))
    adjacency = adjacency if adjacency is not None else build_graph(sd, train)
    progress(f'Training paper implementation: {sd.name}, {train.N}/{val.N}/{test.N} patients')
    model, improvement, thresholds, validation = fit_candidate(
        train, val, adjacency, epochs, seed, progress, device)
    # Test predictions are made after fitting and validation decisions finish.
    test_base, test_h = infer(model, test)
    test_improved = improvement.predict(test, test_base, test_h)
    metrics = {}
    for key, predictions in (('paper', test_base), ('improved', test_improved)):
        metrics[key] = T.calculate_metrics(test.y_flare[test.valid], predictions['p'][test.valid],
            test.y_sev[test.valid], predictions['sev'][test.valid], thresholds[key])
    rows, visits = np.where(test.valid)
    frame = pd.DataFrame({'patient_id': np.asarray(test.patient_ids)[rows], 'visit': visits + 1,
        'actual_flare': test.y_flare[test.valid].astype(int), 'actual_severity': test.y_sev[test.valid]})
    for key, predictions in (('paper', test_base), ('improved', test_improved)):
        frame[f'{key}_risk'] = predictions['p'][test.valid]
        frame[f'{key}_prediction'] = (frame[f'{key}_risk'] >= thresholds[key]).astype(int)
        frame[f'{key}_severity'] = predictions['sev'][test.valid]
    frame.to_csv(folder / 'predictions.csv', index=False)
    report = {'framework_revision': 3, 'completed_at': time.time(), 'dataset': sd.name,
        'methods': METHODS, 'epochs': epochs, 'seed': seed, 'split_seed': SEED,
        'patients': sd.N, 'input_features': sd.V, 'visits': int(sd.n_vis.sum()),
        'splits': {key: [sd.patient_ids[i] for i in indices] for key, indices in sd.splits.items()},
        'thresholds': thresholds, 'metrics': metrics, 'improvement': improvement.validation,
        'validation_metrics': validation,
        'delta_auroc': metrics['improved']['auroc'] - metrics['paper']['auroc'],
        'paired_patient_delta_ci95': patient_bootstrap(frame),
        'history': model.training_history, 'feature_names': sd.names}
    (folder / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    T.save_model(model, folder / 'paper.pt', adjacency, sd)
    joblib.dump(improvement, folder / 'improved.joblib')
    # Keep only valid test visits; no synthetic labels enter the projection.
    np.savez_compressed(folder / 'embeddings.npz', representation=test_h[test.valid],
        patient_id=np.asarray(test.patient_ids)[rows], visit=visits + 1,
        outcome=test.y_flare[test.valid], paper_risk=test_base['p'][test.valid],
        improved_risk=test_improved['p'][test.valid])
    return report
