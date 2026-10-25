"""Public dataset preparation; downloads commit only after gzip verification."""
import csv
import gzip
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.project.settings import ROOT


def atomic_download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with gzip.open(destination, 'rb') as stream:
            stream.read(1024)
        return destination
    partial = destination.with_suffix(destination.suffix + '.partial')
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'LongitudinalHealthResearch/1.0'})
        with urllib.request.urlopen(request, timeout=120) as source, partial.open('wb') as target:
            shutil.copyfileobj(source, target)
        # Verify gzip CRC and completion before replacing the final filename.
        with gzip.open(partial, 'rb') as stream:
            while stream.read(1024 * 1024):
                pass
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def prepare_real():
    raw = ROOT / 'data/raw/GSE65391_series_matrix.txt.gz'
    atomic_download('https://ftp.ncbi.nlm.nih.gov/geo/series/GSE65nnn/GSE65391/matrix/GSE65391_series_matrix.txt.gz', raw)
    destination = ROOT / 'data/processed/visits_clean.csv'
    stamp = destination.with_suffix('.source.json')
    signature = {'raw_bytes': raw.stat().st_size, 'raw_mtime_ns': raw.stat().st_mtime_ns,
                 'parser': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if destination.exists() and stamp.exists():
        try:
            if json.loads(stamp.read_text()) == signature:
                return destination
        except ValueError:
            pass
    records = []
    with gzip.open(raw, 'rt', encoding='latin1') as stream:
        for line in stream:
            if line.startswith('!series_matrix_table_begin'):
                break
            if not line.startswith('!Sample_'):
                continue
            values = next(csv.reader([line], delimiter='\t'))
            if values[0] == '!Sample_geo_accession':
                records = [{'gsm': value} for value in values[1:]]
            elif values[0] == '!Sample_characteristics_ch1':
                for record, value in zip(records, values[1:]):
                    if ':' in value:
                        key, item = value.split(':', 1)
                        record[key.strip().lower().replace(' ', '_')] = item.strip()
    df = pd.DataFrame(records)
    df = df[(df.disease_state == 'SLE') & (df['set'] != 'Technical_Replicate')].copy()
    df['visit_no'] = pd.to_numeric(df['visit'], errors='coerce')
    keep_text = {'gsm', 'subject', 'gender', 'race', 'set', 'disease_state'}
    for column in df.columns.difference(list(keep_text)):
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=['subject', 'visit_no', 'cumulative_time', 'sledai'])
    df = df.sort_values(['subject', 'cumulative_time', 'visit_no', 'gsm'])
    df = df.drop_duplicates(['subject', 'visit_no'], keep='first')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.csv.partial')
    df.to_csv(partial, index=False)
    os.replace(partial, destination)
    stamp_partial = stamp.with_suffix('.json.partial')
    stamp_partial.write_text(json.dumps(signature))
    os.replace(stamp_partial, stamp)
    return destination


def prepare_synthetic():
    from src.download_data import generate_exact_paper_cohort, ALL_54_FEATURES, LAB_FEATURES, IMMUNO_FEATURES, DEMO_VITAL_FEATURES, TREATMENT_FEATURES
    import json
    folder = ROOT / 'data/processed'
    expected = [folder / name for name in ('autoimmune_multimodal_54dim.csv', 'cohort_splits.json', 'feature_metadata.json')]
    if all(path.exists() for path in expected):
        return expected[0]
    df, splits = generate_exact_paper_cohort()
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(expected[0], index=False)
    expected[1].write_text(json.dumps({key: sorted(value) for key, value in splits.items()}))
    expected[2].write_text(json.dumps({'features_54': ALL_54_FEATURES,
        'categories': {'laboratory': LAB_FEATURES, 'immunological': IMMUNO_FEATURES,
                       'demographic_vitals': DEMO_VITAL_FEATURES, 'pharmacological': TREATMENT_FEATURES}}))
    return expected[0]


def split_real(path, seed=42):
    from src.causal import dataset as D
    sd = D.make_real(path)
    order = np.random.default_rng(seed).permutation(sd.N)
    a, b = int(.7 * sd.N), int(.85 * sd.N)
    sd.splits = dict(zip(('train', 'val', 'test'), np.split(order, [a, b])))
    # Refit nuisance-age normalization on training patients as well as features.
    age_records = pd.read_csv(path, usecols=['subject', 'age']).drop_duplicates('subject').set_index('subject')
    age = pd.to_numeric(age_records.reindex(sd.patient_ids)['age'], errors='coerce').to_numpy(dtype=float)
    train_age = age[sd.splits['train']]
    age_mean = float(np.nanmean(train_age)) if np.isfinite(train_age).any() else 0.
    age_std = max(float(np.nanstd(train_age)), 1e-6) if np.isfinite(train_age).any() else 1.
    sd.nuisance[:, 0] = np.nan_to_num((age - age_mean) / age_std)
    sd.subgroup = (sd.nuisance[:, 0] > 0).astype(int)
    # Explicit research bins: remission 0, then 1–5, 6–10, 11–19, >=20.
    # The legacy loader used left-closed boundaries and mapped SLEDAI=0 to 1.
    activity = sd.raw[:, :, sd.names.index('sledai')]
    severity = np.digitize(np.nan_to_num(activity), [0, 5, 10, 19], right=True)
    sd.y_sev[:, :-1] = severity[:, 1:]
    sd.y_sev[~sd.valid] = 0
    D.finalize_real(sd, sd.splits['train'])
    sd.extra['normalization'] = 'train'
    return sd
