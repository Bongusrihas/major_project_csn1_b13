"""Resumable preparation with content-addressed, integrity-checked caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import webbrowser
import socket
import urllib.request
from contextlib import contextmanager

import numpy as np
from filelock import FileLock, Timeout

from src.project.settings import ROOT, RUNS, EPOCHS, RANDOM_DATASETS, RANDOM_PATIENTS, RANDOM_VISITS, EDGE_NOISE, SEED, FROZEN_SEED_FILE, training_seed


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def training_fingerprint():
    files = [p for p in (ROOT / 'src').rglob('*.py') if p.parent.name != 'project']
    files += [ROOT / 'src/project' / name for name in ('settings.py', 'data.py', 'pair.py', 'improvement.py', 'seeds.py')]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    # Different numerical-library versions should not reuse a stale model cache.
    from importlib.metadata import version
    for package in ('torch', 'numpy', 'scikit-learn', 'xgboost'):
        digest.update(f'{package}:{version(package)}'.encode())
    return digest.hexdigest()


TRAINING_ARTIFACTS = {'paper.pt', 'improved.joblib', 'results.json', 'predictions.csv', 'embeddings.npz'}


def valid_artifacts(folder, training_only=False):
    try:
        manifest = json.loads((Path(folder) / 'complete.json').read_text())
        entries = manifest['sha256']
        if not TRAINING_ARTIFACTS.issubset(entries):
            return False
        selected = {name: value for name, value in entries.items() if not training_only or name in TRAINING_ARTIFACTS}
        return all((Path(folder) / name).is_file() and checksum(Path(folder) / name) == value
                   for name, value in selected.items()) and bool(selected)
    except (OSError, ValueError, KeyError):
        return False


def seal(folder, key):
    files = [path for path in folder.rglob('*') if path.is_file() and path.name != 'complete.json' and not path.name.endswith('.partial')]
    atomic_json(folder / 'complete.json', {'created_at': time.time(), 'signature': key,
        'plots_source': checksum(ROOT / 'src/project/plots.py'),
        'sha256': {str(path.relative_to(folder)): checksum(path) for path in files}})


def cached_pair(sd, identity, epochs=EPOCHS, adjacency=None, refresh=False, progress=print):
    from src.project.pair import train_pair
    from src.project.plots import generate_figures
    model_seed = training_seed()
    key = hashlib.sha256(json.dumps({'identity': identity, 'training': training_fingerprint(),
        'epochs': epochs, 'model_seed': model_seed}, sort_keys=True).encode()).hexdigest()
    index_path = RUNS / 'cache_index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    if not refresh and key in index and valid_artifacts(RUNS / index[key], training_only=True):
        folder = RUNS / index[key]
        metadata = json.loads((folder / 'complete.json').read_text())
        if not valid_artifacts(folder) or metadata.get('plots_source') != checksum(ROOT / 'src/project/plots.py'):
            progress(f'Regenerating figures from cached predictions: {sd.name}')
            generate_figures(folder)
            seal(folder, key)
        else:
            progress(f'Cache hit: {sd.name}')
        return folder
    RUNS.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=key[:12] + '-', dir=RUNS))
    # An interrupted run has no complete.json and can never masquerade as ready.
    train_pair(sd, folder, epochs=epochs, seed=model_seed, adjacency=adjacency, progress=progress)
    generate_figures(folder)
    seal(folder, key)
    index[key] = folder.name
    atomic_json(index_path, index)
    return folder


def random_dataset(index, seed, patients=RANDOM_PATIENTS, visits=RANDOM_VISITS):
    from src.synthetic.generator import generate_single_disease_dataset
    from src.causal import dataset as D
    sd, adjacency, _, meta = generate_single_disease_dataset(index, patients, visits, EDGE_NOISE, seed)
    order = np.random.default_rng(seed).permutation(sd.N)
    sd.splits = dict(zip(('train', 'val', 'test'), np.split(order, [int(.6 * sd.N), int(.8 * sd.N)])))
    sd.scaler = D.fit_scaler(np.where(sd.M > 0, sd.raw, np.nan), sd.n_vis, sd.names, sd.roles, set(), sd.splits['train'])
    transformed, _ = D.apply_scaler(sd.raw, sd.names, sd.scaler)
    sd.X = np.where(sd.M > 0, transformed, 0).astype(np.float32)
    sd.extra.update(normalization='train', simulation=meta)
    return sd, adjacency


def ready_signature(epochs, random_count):
    """Cheap launch check: inspect code/data/artifacts without importing torch."""
    paths = [ROOT / 'data/processed' / name for name in
             ('visits_clean.csv', 'visits_clean.source.json', 'autoimmune_multimodal_54dim.csv',
              'cohort_splits.json', 'feature_metadata.json')]
    paths += [ROOT / 'src/project/plots.py', ROOT / 'src/project/paper.py', ROOT / 'docs/paper/reference.pdf']
    paths += [FROZEN_SEED_FILE]
    paths += [ROOT / 'docs/paper' / f'fig{index}.png' for index in range(6, 14)]
    raw = ROOT / 'data/raw/GSE65391_series_matrix.txt.gz'
    value = {'training': training_fingerprint(), 'epochs': epochs, 'random_count': random_count,
             'files': {str(path.relative_to(ROOT)): checksum(path) if path.exists() else None for path in paths},
             'raw': [raw.stat().st_size, raw.stat().st_mtime_ns] if raw.exists() else None}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def preparation_lock(progress):
    lock = FileLock(str(RUNS / '.prepare.lock'))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        progress('Another preparation is active; waiting for its cache to finish...')
        lock.acquire(timeout=60)
    try:
        yield
    finally:
        lock.release()


def prepare(refresh=False, epochs=EPOCHS, random_count=RANDOM_DATASETS, progress=print, reselect_seed=False):
    from src.project.data import prepare_real, prepare_synthetic, split_real
    from src.project.paper import prepare_paper
    RUNS.mkdir(parents=True, exist_ok=True)
    with preparation_lock(progress):
        dashboard_path = RUNS / 'dashboard.json'
        if not refresh and not reselect_seed and FROZEN_SEED_FILE.exists() and dashboard_path.exists():
            previous = json.loads(dashboard_path.read_text())
            folders = [*previous['datasets'].values(), *previous['random_runs']]
            if previous.get('ready_signature') == ready_signature(epochs, random_count) and all(
                    valid_artifacts(ROOT / folder) for folder in folders):
                from src.project.reporting import sync_results
                sync_results(previous)
                progress('Verified cache hit: datasets, both methods, figures and random tests are ready.')
                return previous
        from src.causal import dataset as D
        progress('Preparing public data and original paper figures')
        paper = prepare_paper()
        real = prepare_real()
        simulation = prepare_synthetic()
        from src.project.seeds import ensure_frozen_seed
        seed_record = ensure_frozen_seed(real, epochs, force=reselect_seed, progress=progress)
        datasets = {}
        for name, path, loader in (('real', real, split_real), ('simulation', simulation, D.load_multimodal_cohort)):
            sd = loader(path)
            identity = {'dataset': name, 'data': checksum(path), 'seed': SEED}
            if name == 'simulation':
                identity['splits'] = checksum(path.parent / 'cohort_splits.json')
                identity['metadata'] = checksum(path.parent / 'feature_metadata.json')
            folder = cached_pair(sd, identity, epochs, refresh=refresh, progress=progress)
            datasets[name] = str(folder.relative_to(ROOT))
        random_runs = []
        for index in range(random_count):
            seed = SEED + index * 17
            sd, adjacency = random_dataset(index, seed)
            folder = cached_pair(sd, {'random_dataset': index, 'seed': seed, 'patients': sd.N,
                'visits': sd.T, 'edge_noise': EDGE_NOISE}, epochs, adjacency, refresh, progress)
            random_runs.append(str(folder.relative_to(ROOT)))
        dashboard = {'schema': 1, 'methods': ['paper', 'improved'], 'datasets': datasets,
                     'random_runs': random_runs, 'defaults': {'epochs': epochs, 'random_datasets': random_count},
                     'published': paper, 'prepared_at': time.time(),
                     'training_seed': seed_record['selected_seed'], 'split_seed': SEED,
                     'seed_selection': seed_record}
        dashboard['ready_signature'] = ready_signature(epochs, random_count)
        atomic_json(RUNS / 'dashboard.json', dashboard)
        from src.project.reporting import sync_results
        sync_results(dashboard)
        progress('Ready. Training, figures and projections are cached.')
        return dashboard


def launch():
    """Reuse this project's live server, otherwise start a free loopback port."""
    import psutil
    marker = RUNS / 'server.json'
    app = ROOT / 'app/streamlit_app.py'
    ui_files = sorted((ROOT / 'app').rglob('*.py')) + [ROOT / '.streamlit/config.toml']
    ui_source = hashlib.sha256(''.join(checksum(path) for path in ui_files if path.exists()).encode()).hexdigest()
    if marker.exists():
        try:
            state = json.loads(marker.read_text())
            command = psutil.Process(state['pid']).cmdline()
            if str(app) in command and state.get('ui_source') == ui_source:
                url = f"http://127.0.0.1:{state['port']}"
                with urllib.request.urlopen(url + '/_stcore/health', timeout=2) as response:
                    if response.status == 200:
                        print(f'Opening the existing dashboard: {url}', flush=True)
                        webbrowser.open(url)
                        return
            elif str(app) in command:
                # A healthy Streamlit server can still hold old imported page
                # modules. Restart only the process tree verified as this app.
                print('Dashboard UI changed; restarting this project server.', flush=True)
                owned = psutil.Process(state['pid'])
                processes = owned.children(recursive=True) + [owned]
                for process in processes:
                    try:
                        process.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(processes, timeout=5)
                for process in alive:
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
        except (OSError, ValueError, KeyError, psutil.Error):
            pass
    port = None
    for candidate in range(8501, 8521):
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', candidate))
                port = candidate
                break
            except OSError:
                continue
    if port is None:
        raise RuntimeError('No free local dashboard port between 8501 and 8520')
    process = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', str(app),
        '--server.address=127.0.0.1', f'--server.port={port}', '--server.headless=false',
        '--browser.gatherUsageStats=false'], cwd=ROOT)
    atomic_json(marker, {'pid': process.pid, 'port': port, 'app': str(app), 'ui_source': ui_source})
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
    finally:
        if marker.exists() and json.loads(marker.read_text()).get('pid') == process.pid:
            marker.unlink()


def main():
    parser = argparse.ArgumentParser(description='Prepare and launch the two-method research dashboard')
    parser.add_argument('--update', action='store_true', help='Rebuild artifacts without deleting previous runs')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--reselect-seed', action='store_true', help='Explicitly repeat validation seed selection; ordinary runs keep the frozen seed')
    args = parser.parse_args()
    prepare(refresh=args.update, reselect_seed=args.reselect_seed)
    if not args.prepare_only:
        launch()


if __name__ == '__main__':
    main()
