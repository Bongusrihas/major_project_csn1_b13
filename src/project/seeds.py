"""One-time model-seed search on a fixed validation split; no test scoring."""
import json
import math
import time

from src.project.settings import SEED, SEED_CANDIDATES, FROZEN_SEED_FILE, RUNS, EPOCHS


def select_seed(trials):
    """Rank only proposed-framework validation scores; test fields are ignored."""
    def rank(trial):
        values = trial['validation_metrics']['improved']
        def finite(key, default):
            value = values.get(key)
            return float(value) if value is not None and math.isfinite(value) else default
        return (finite('auroc', -1.), finite('auprc', -1.),
                -finite('brier', float('inf')), -finite('mae', float('inf')))
    if not trials:
        raise ValueError('No completed validation seed trials')
    return max(trials, key=rank)['seed']


def ensure_frozen_seed(real_path, epochs=EPOCHS, force=False, progress=print):
    from src.project.pipeline import atomic_json, checksum, training_fingerprint
    if FROZEN_SEED_FILE.exists() and not force:
        record = json.loads(FROZEN_SEED_FILE.read_text(encoding='utf-8'))
        if select_seed(record['trials']) != record['selected_seed']:
            raise ValueError('Frozen seed record does not match its validation scores')
        progress(f"Reusing frozen model seed {record['selected_seed']} (patient split seed {SEED}).")
        return record
    from src.project.data import split_real
    from src.causal import dataset as D
    from src.project.pair import build_graph, fit_candidate
    import pandas as pd
    import hashlib

    sd = split_real(real_path, seed=SEED)
    train, val = [D.subset_dataset(sd, sd.splits[key]) for key in ('train', 'val')]
    # Even graph discovery receives only training observations.
    adjacency = build_graph(train, train)
    source = {'training': training_fingerprint(), 'data': checksum(real_path),
              'split_seed': SEED, 'epochs': epochs, 'candidates': list(SEED_CANDIDATES)}
    signature = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    folder = RUNS / 'seed_search' / signature[:20]
    folder.mkdir(parents=True, exist_ok=True)
    trials = []
    for seed in SEED_CANDIDATES:
        path = folder / f'seed_{seed}.json'
        if path.exists():
            trial = json.loads(path.read_text(encoding='utf-8'))
            progress(f'Cached validation trial: model seed {seed}')
        else:
            progress(f'Validation-only seed search: {seed}; patient split stays {SEED}')
            model, proposed, _, validation = fit_candidate(train, val, adjacency, epochs, seed, progress)
            trial = {'seed': seed, 'validation_metrics': validation,
                     'correction': proposed.validation}
            atomic_json(path, trial)
            del model, proposed
        trials.append(trial)
    selected = select_seed(trials)
    record = {'selected_seed': selected, 'split_seed': SEED, 'epochs': epochs,
        'selection': 'Proposed validation AUROC; ties: AUPRC, Brier, severity MAE',
        'test_used_for_selection': False, 'source_signature': signature,
        'selected_at': time.time(), 'trials': trials,
        'train_patient_ids': train.patient_ids, 'validation_patient_ids': val.patient_ids}
    atomic_json(FROZEN_SEED_FILE, record)
    frame = pd.DataFrame([{'seed': trial['seed'], **trial['validation_metrics']['improved']} for trial in trials])
    frame.to_csv(folder / 'validation_scores.csv', index=False)
    progress(f'Frozen model seed {selected}; all future runs reuse it.')
    return record
