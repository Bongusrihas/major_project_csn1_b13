"""Local CSV prediction and outcome comparison with the two frozen frameworks.

Uploaded outcome columns are kept separate from model inputs. Neither fitting,
calibration nor threshold selection happens here.
"""
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

MAX_ROWS = 10000
MAX_VISITS = 128
FLARE_ALIASES = ('actual_next_flare', 'next_flare_event', 'actual_flare', 'flare_event')
SEVERITY_ALIASES = ('actual_next_severity', 'next_severity_score', 'actual_severity', 'severity_score')


def binary_labels(series):
    values = series.astype('string').str.strip().str.lower()
    mapping = {'yes': 1., 'y': 1., 'true': 1., '1': 1., '1.0': 1.,
               'no': 0., 'n': 0., 'false': 0., '0': 0., '0.0': 0.}
    missing = values.isna() | values.isin(['', 'na', 'n/a', 'nan', 'unknown', 'not applicable'])
    invalid = ~missing & ~values.isin(mapping)
    if invalid.any():
        raise ValueError('Flare outcomes must be Yes/No, True/False, 1/0, or blank. '
                         f"Invalid values: {series[invalid].astype(str).unique()[:5].tolist()}")
    return values.map(mapping).astype(float)


def prepare_records(frame, features, flare_column=None, severity_column=None, outcome_timing='next'):
    """Validate/sort input records; retain labels independently of measurements."""
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError(f'Provide between 1 and {MAX_ROWS:,} visit rows.')
    frame = frame.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.columns.duplicated().any():
        raise ValueError('Column names must be unique.')
    aliases = {'subject': 'patient_id', 'visit_no': 'visit', 'visit_index': 'visit', 'visit_number': 'visit'}
    for old, new in aliases.items():
        if new not in frame and old in frame:
            frame[new] = frame[old]
            if old == 'visit_index':
                frame[new] = pd.to_numeric(frame[new], errors='raise') + 1
    if not {'patient_id', 'visit'}.issubset(frame.columns):
        raise ValueError('Required columns: patient_id and visit (or subject and visit_no).')
    if frame.patient_id.isna().any() or frame.patient_id.astype(str).str.strip().eq('').any():
        raise ValueError('Every row needs a patient_id.')
    frame['patient_id'] = frame.patient_id.astype(str).str.strip()
    visit = pd.to_numeric(frame.visit, errors='coerce')
    if (~np.isfinite(visit) | (visit < 1) | (visit % 1 != 0)).any():
        raise ValueError('visit must contain positive whole-number visit identifiers.')
    frame['visit'] = visit.astype(int)
    frame['_source_row'] = np.arange(len(frame)) + 2  # CSV header is row 1.
    if frame.duplicated(['patient_id', 'visit']).any():
        raise ValueError('Each patient_id + visit pair must be unique.')
    frame = frame.sort_values(['patient_id', 'visit'], kind='stable').reset_index(drop=True)
    if frame.groupby('patient_id').size().max() > MAX_VISITS:
        raise ValueError(f'The model supports at most {MAX_VISITS} uploaded visits per patient.')
    for column in (flare_column, severity_column):
        if column is not None and (column not in frame or column in features):
            raise ValueError('Outcome columns must be separate from the model input measurements.')
    if flare_column == severity_column and flare_column is not None:
        raise ValueError('Flare and severity must use separate outcome columns.')
    if 'gap_days' not in frame:
        if 'cumulative_time' not in frame:
            raise ValueError('Provide gap_days, or cumulative_time in days, to describe visit intervals.')
        days = pd.to_numeric(frame.cumulative_time, errors='coerce')
        if not np.isfinite(days).all():
            raise ValueError('cumulative_time must contain finite numeric days.')
        frame['gap_days'] = days.groupby(frame.patient_id).diff().fillna(0.)
    gaps = pd.to_numeric(frame.gap_days, errors='coerce')
    if (~np.isfinite(gaps) | (gaps < 0)).any():
        raise ValueError('gap_days must contain finite, nonnegative days; visits must be chronological.')
    frame['gap_days'] = gaps.astype(float)
    supplied = [name for name in features if name in frame]
    if not supplied:
        raise ValueError('No recognized clinical measurements found. Use the selected model’s CSV template.')
    for name in supplied:
        original = frame[name].replace(r'^\s*$', np.nan, regex=True)
        numeric = pd.to_numeric(original, errors='coerce')
        if (original.notna() & (numeric.isna() | ~np.isfinite(numeric))).any():
            raise ValueError(f'{name} must contain numeric measurements or blanks.')
        frame[name] = numeric.astype(float)
    if frame[supplied].isna().all(axis=1).any():
        raise ValueError('Every visit needs at least one clinical measurement; leave other measurements blank.')
    for name in set(features) - set(supplied):
        frame[name] = np.nan
    flare = binary_labels(frame[flare_column]) if flare_column else pd.Series(np.nan, index=frame.index)
    severity = pd.Series(np.nan, index=frame.index)
    if severity_column:
        original = frame[severity_column].replace(r'^\s*$', np.nan, regex=True)
        severity = pd.to_numeric(original, errors='coerce')
        invalid = original.notna() & (severity.isna() | ~severity.isin([0, 1, 2, 3, 4]))
        if invalid.any():
            raise ValueError('Actual severity must be an ordinal category 0–4, or blank; raw SLEDAI belongs in sledai.')
    if outcome_timing == 'current':
        flare = flare.groupby(frame.patient_id).shift(-1)
        severity = severity.groupby(frame.patient_id).shift(-1)
    elif outcome_timing != 'next':
        raise ValueError('Outcome timing must be current or next.')
    frame['_actual_next_flare'] = flare
    frame['_actual_next_severity'] = severity
    return frame, {'present_features': supplied, 'missing_features': [name for name in features if name not in supplied]}


def input_template(features):
    """Blank values avoid accidentally presenting invented measurements as data."""
    columns = ['patient_id', 'visit', 'gap_days', *features, 'actual_next_flare', 'actual_next_severity']
    row = {name: '' for name in columns}
    row.update(patient_id='PATIENT-001', visit=1, gap_days=0)
    return pd.DataFrame([row], columns=columns)


def actual_heldout_records(report, root):
    """Real clinical records with known following-visit labels, never training rows."""
    frame = pd.read_csv(Path(root) / 'data/processed/visits_clean.csv')
    frame = frame[frame.subject.astype(str).isin(report['splits']['test'])].copy()
    frame = frame.sort_values(['subject', 'cumulative_time', 'visit_no'])
    output = pd.DataFrame({'patient_id': frame.subject.astype(str), 'visit': frame.visit_no.astype(int),
                          'gap_days': frame.groupby('subject').cumulative_time.diff().fillna(0.)})
    for name in report['feature_names']:
        output[name] = frame[name]
    next_score = frame.groupby('subject').sledai.shift(-1)
    flare = (next_score - frame.sledai >= 4).where(next_score.notna())
    output['actual_next_flare'] = flare.map({True: 'Yes', False: 'No'})
    output['actual_next_severity'] = pd.Series(
        np.digitize(next_score.fillna(0), [0, 5, 10, 19], right=True), index=frame.index).where(next_score.notna())
    return output.reset_index(drop=True)


@dataclass
class ModelBundle:
    model: object
    proposed: object
    metadata: dict
    report: dict


def load_bundle(folder, device=None):
    """Only load hash-verified project models, never models supplied in an upload."""
    from src.project.pipeline import valid_artifacts
    from src.causal import train as T
    import torch
    import joblib
    folder = Path(folder)
    if not valid_artifacts(folder, training_only=True):
        raise ValueError('The saved model artifacts failed their integrity check. Run python run.py to repair them.')
    metadata = torch.load(folder / 'paper.pt', map_location='cpu', weights_only=False)
    if metadata.get('scaler') is None:
        raise ValueError('This model has no training-fitted scaler; rebuild it with python run.py --update.')
    model = T.load_model(folder / 'paper.pt', device=device or T.DEVICE)
    proposed = joblib.load(folder / 'improved.joblib')
    report = json.loads((folder / 'results.json').read_text(encoding='utf-8'))
    return ModelBundle(model, proposed, metadata, report)


def predict_records(records, bundle, patients_per_batch=16):
    """Bounded, read-only inference. Outcome columns never enter a model tensor."""
    from src.causal.dataset import SequenceDataset, apply_scaler
    from src.project.pair import infer
    names, roles = bundle.metadata['names'], bundle.metadata['roles']
    ids = records.patient_id.unique()
    groups = {patient: group for patient, group in records.groupby('patient_id', sort=False)}
    outputs = []
    for start in range(0, len(ids), patients_per_batch):
        selected = ids[start:start + patients_per_batch]
        parts = [groups[patient] for patient in selected]
        lengths = np.array([len(part) for part in parts])
        n, t, v = len(parts), int(lengths.max()), len(names)
        # Match training preprocessing precision before converting scaled values.
        raw = np.full((n, t, v), np.nan, dtype=np.float64)
        gap = np.zeros((n, t), dtype=np.float32)
        pad = np.arange(t)[None] < lengths[:, None]
        for i, part in enumerate(parts):
            raw[i, :len(part)] = part[names].to_numpy(dtype=np.float64)
            gap[i, :len(part)] = np.log1p(part.gap_days.to_numpy(dtype=np.float32) / 30.)
        x, mask = apply_scaler(raw, names, bundle.metadata['scaler'])
        sd = SequenceDataset('Uploaded records', x, mask, raw, gap, lengths,
            np.zeros((n, t), np.float32), np.zeros((n, t), np.float32), pad,
            np.zeros((n, bundle.metadata['n_nuisance']), np.float32), np.zeros(n, int),
            names, roles, bundle.metadata['treat_idx'], list(selected), scaler=bundle.metadata['scaler'])
        paper, h = infer(bundle.model, sd, batch_size=patients_per_batch)
        proposed = bundle.proposed.predict(sd, paper, h)
        result = pd.concat(parts, ignore_index=True)[['patient_id', 'visit', '_source_row',
            '_actual_next_flare', '_actual_next_severity']].copy()
        result['history_visits'] = np.concatenate([np.arange(1, length + 1) for length in lengths])
        for name, predictions in (('paper', paper), ('improved', proposed)):
            result[f'{name}_risk'] = predictions['p'][pad]
            result[f'{name}_severity'] = predictions['sev'][pad]
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def compare_results(predictions, report, include_seen=False):
    """Score only supplied labels, using previously frozen validation thresholds."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    result = predictions.copy()
    actual = pd.to_numeric(result['_actual_next_flare'], errors='coerce')
    severity = pd.to_numeric(result['_actual_next_severity'], errors='coerce')
    labeled = actual.notna()
    train, val, test = [set(report['splits'][key]) for key in ('train', 'val', 'test')]
    result['patient_exposure'] = result.patient_id.map(lambda patient: 'Training patient' if patient in train else
        'Validation patient' if patient in val else 'Held-out patient' if patient in test else 'New / unrecognized patient ID')
    eligible = ~result.patient_id.isin(train | val) if not include_seen else pd.Series(True, index=result.index)
    result['used_for_flare_metrics'] = labeled & eligible
    result['actual_next_flare'] = actual.map({1.: 'Yes', 0.: 'No'}).fillna('Unknown')
    result['actual_next_severity'] = severity
    correctness = {}
    metrics = []
    for name, label in (('paper', 'Paper framework'), ('improved', 'Proposed framework')):
        predicted = (result[f'{name}_risk'] >= report['thresholds'][name]).astype(int)
        correct = predicted.eq(actual) & labeled
        correctness[name] = correct
        result[f'{name}_flare'] = predicted.map({1: 'Yes', 0: 'No'})
        result[f'{name}_correct'] = np.where(~labeled, 'Not scored', np.where(correct, 'Correct', 'Wrong'))
        result[f'{name}_absolute_error'] = (result[f'{name}_severity'] - severity).abs()
        fm, sm = labeled & eligible, severity.notna() & eligible
        item = {'Method': label, 'Scored flare visits': int(fm.sum()), 'Accuracy': None,
                'AUROC': None, 'AUPRC': None, 'F1': None, 'Precision': None, 'Recall': None,
                'Brier': None, 'Scored severity visits': int(sm.sum()), 'Severity MAE': None, 'Severity RMSE': None}
        if fm.any():
            y, p, decision = actual[fm].astype(int), result.loc[fm, f'{name}_risk'], predicted[fm]
            item.update(Accuracy=float(accuracy_score(y, decision)), F1=float(f1_score(y, decision, zero_division=0)),
                Precision=float(precision_score(y, decision, zero_division=0)), Recall=float(recall_score(y, decision, zero_division=0)),
                Brier=float(np.mean((p - y) ** 2)))
            if y.nunique() == 2:
                item['AUROC'] = float(roc_auc_score(y, p))
            if y.sum() > 0:
                item['AUPRC'] = float(average_precision_score(y, p))
        if sm.any():
            errors = result.loc[sm, f'{name}_absolute_error']
            item['Severity MAE'], item['Severity RMSE'] = float(errors.mean()), float(np.sqrt((errors ** 2).mean()))
        metrics.append(item)
    result['who_was_correct'] = np.select([~labeled, correctness['paper'] & correctness['improved'],
        correctness['paper'], correctness['improved']], ['Not scored', 'Both', 'Paper only', 'Proposed only'], default='Neither')
    result['closer_severity'] = np.where(severity.isna(), 'Not scored', np.where(
        np.isclose(result.paper_absolute_error, result.improved_absolute_error), 'Tie', np.where(
        result.paper_absolute_error < result.improved_absolute_error, 'Paper', 'Proposed')))
    return result.drop(columns=['_actual_next_flare', '_actual_next_severity']), pd.DataFrame(metrics)
