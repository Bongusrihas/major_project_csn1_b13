"""Four focused views; exactly two project methods; no automatic UI training."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.project.settings import METHODS, RUNS, PAPER_DIR, EPOCHS, RANDOM_PATIENTS, SEED
from src.project.paper import REFERENCE

def apply_style():
    st.markdown('''<style>
    .block-container {max-width:1320px;padding-top:2.1rem;padding-bottom:3rem;}
    h1 {font-family:Georgia,serif;font-weight:500!important;letter-spacing:-.045em;font-size:3.3rem!important;}
    h2,h3 {letter-spacing:-.025em;}
    p {line-height:1.65;}
    [data-testid="stMetricValue"], [data-testid="stDataFrame"] {font-variant-numeric:tabular-nums;}
    [data-testid="stMetric"] {border-top:2px solid #c9ddd8;padding-top:1rem;}
    [data-testid="stSidebar"] {background:#edf2f1;}
    .eyebrow {font-size:.73rem;letter-spacing:.16em;color:#518176;text-transform:uppercase;font-weight:600;margin-bottom:.5rem;}
    .subtext {max-width:700px;color:#657176;margin-bottom:1.8rem;}
    button {transition:background-color .2s ease,transform .2s ease;}
    button:active {transform:translateY(1px);}
    button:focus-visible {outline:2px solid #178878!important;outline-offset:3px;}
    </style>''', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def read_json(path, version):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@st.cache_data(show_spinner=False)
def read_csv(path, version):
    return pd.read_csv(path)


def json_file(path):
    path = Path(path)
    return read_json(str(path), path.stat().st_mtime_ns)


def csv_file(path):
    path = Path(path)
    return read_csv(str(path), path.stat().st_mtime_ns)


def manifest():
    path = RUNS / 'dashboard.json'
    if not path.exists():
        st.info('Prepare the project once with `python run.py`. It downloads data, trains both methods, generates figures and opens this app.')
        st.stop()
    return json_file(path)


def heading(number, title, description):
    st.markdown(f'<div class="eyebrow">Research studio / {number}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subtext">{description}</div>', unsafe_allow_html=True)


def select_run(key):
    data = manifest()
    labels = {'real': 'Public SLE cohort · GSE65391', 'simulation': 'Simulated SLE / RA cohort'}
    dataset = st.radio('Evaluation cohort', list(data['datasets']), format_func=lambda value: labels[value],
                       horizontal=True, key=key)
    folder = ROOT / data['datasets'][dataset]
    required = ['results.json', 'predictions.csv', 'projection.npz']
    if not all((folder / name).exists() for name in required):
        st.info('This cached run is incomplete. Run `python run.py` to repair it; completed training is reused where possible.')
        st.stop()
    return folder, json_file(folder / 'results.json')


def score_table(report):
    frame = pd.DataFrame([{'Method': label, **report['metrics'][key]} for key, label in METHODS.items()])
    return frame.rename(columns={'auroc': 'AUROC', 'auprc': 'AUPRC', 'f1': 'F1', 'mae': 'Severity MAE',
                                 'rmse': 'Severity RMSE', 'brier': 'Brier', 'ece': 'ECE'})


def prediction_view(folder, key):
    predictions = csv_file(folder / 'predictions.csv')
    patient = st.selectbox('Patient trajectory', predictions.patient_id.unique(), key=key)
    selected = predictions[predictions.patient_id == patient]
    chart = go.Figure()
    for method, label in METHODS.items():
        chart.add_trace(go.Scatter(x=selected.visit, y=selected[f'{method}_risk'], name=label,
            mode='lines+markers', line=dict(color='#5b6c82' if method == 'paper' else '#148575', width=3)))
    chart.add_trace(go.Scatter(x=selected.visit, y=selected.actual_flare, mode='markers', name='Observed next-visit flare',
                               marker=dict(symbol='diamond', size=10, color='#c28e44')))
    chart.update_layout(height=320, xaxis_title='Current visit', yaxis_title='Next-visit flare risk / outcome',
                        yaxis=dict(range=[-.03, 1.03]), legend=dict(orientation='h'), margin=dict(t=15, b=15))
    st.plotly_chart(chart, width='stretch')
    st.dataframe(selected.rename(columns={'paper_risk': 'Paper risk', 'improved_risk': 'Proposed risk',
        'paper_prediction': 'Paper prediction', 'improved_prediction': 'Proposed prediction',
        'actual_flare': 'Actual next flare', 'paper_severity': 'Paper severity', 'improved_severity': 'Proposed severity'}), hide_index=True, width='stretch')


def run_details(folder, report):
    stamp = report.get('completed_at', (folder / 'results.json').stat().st_mtime)
    updated = pd.Timestamp(stamp, unit='s', tz='UTC').tz_convert('Asia/Calcutta')
    st.caption(f"Run {folder.name} · updated {updated:%d %b %Y, %H:%M %Z} · "
               f"training seed {report['seed']} ·  split seed {report.get('split_seed', SEED)}")


@st.fragment(run_every='5s')
def watch_latest_run():
    path = RUNS / 'dashboard.json'
    if path.exists():
        version = (path.stat().st_mtime_ns, path.stat().st_size)
        previous = st.session_state.get('_visible_run_version')
        st.session_state['_visible_run_version'] = version
        if previous is not None and previous != version:
            st.session_state.pop('two_method_random_runs', None)
            st.rerun(scope='app')


def paper_reference():
    st.subheader('Original paper figures')
    available = {name: caption for name, caption in REFERENCE['figures'].items() if (PAPER_DIR / f'{name}.png').exists()}
    if available:
        chosen = st.selectbox('Figure', list(available), format_func=lambda name: f"Figure {name[3:]} · {available[name]}")
        st.image(str(PAPER_DIR / f'{chosen}.png'), caption='Original published graphic · Sharma & Kaur (2025) · CC BY 4.0', width='stretch')
    else:
        st.info('Original figure images are generated once by `python run.py`.')
    st.markdown(f"[Read the published paper]({REFERENCE['doi']}) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)")
    source = PAPER_DIR / 'reference.pdf'
    if source.exists():
        st.download_button('Download source paper', source.read_bytes(), 'reference_paper.pdf', 'application/pdf')



def paper_page():
    heading('01', 'Paper vs proposed', 'Current run results, computed from predictions on the same held-out patients.')
    folder, report = select_run('main_comparison_cohort')
    run_details(folder, report)
    left, right = st.columns(2)
    left.metric('Paper framework · AUROC', f"{report['metrics']['paper']['auroc']:.4f}")
    right.metric('Proposed framework · AUROC', f"{report['metrics']['improved']['auroc']:.4f}",
                 f"{report['delta_auroc'] * 100:+.2f} percentage points")
    st.dataframe(score_table(report), hide_index=True, width='stretch')
    chart = go.Figure(go.Bar(x=list(METHODS.values()),
        y=[report['metrics'][method]['auroc'] for method in METHODS], marker_color=['#59677c', '#168575']))
    chart.update_layout(title='Current held-out AUROC · same evaluation cohort', yaxis=dict(range=[0, 1]),
                        height=330, margin=dict(t=50, b=20))
    st.plotly_chart(chart, width='stretch')
    st.subheader('Curves from this run')
    st.image(str(folder / 'figures' / 'discrimination.png'), width='stretch')
    seed_selection = manifest().get('seed_selection')
    if seed_selection:
        st.caption(f"Model seed {seed_selection['selected_seed']} was frozen after comparing "
                   f"{len(seed_selection['trials'])} seeds on validation patients. Test scores did not select the seed.")


def improvement_page():
    heading('02', 'Proposed framework', 'A history-aware residual correction with missingness, irregular visit timing and validation-only tuning.')
    folder, report = select_run('improvement_cohort')
    run_details(folder, report)
    gain = report['delta_auroc'] * 100
    a, b, c = st.columns(3)
    a.metric('Paper framework · AUROC', f"{report['metrics']['paper']['auroc']:.3f}")
    b.metric('Proposed framework · AUROC', f"{report['metrics']['improved']['auroc']:.3f}", f'{gain:+.2f} percentage points')
    c.metric('Held-out patients', len(report['splits']['test']))
    st.caption('These scores compare our two frameworks on this run. The original publication used a separate clinical cohort.')
    st.subheader('What changed')
    st.write('The proposed framework keeps the paper network and adds a temporal residual correction. It uses short- and long-term history, changes since the last observed measurement, observation age, and the temporal embedding. Patient-balanced fitting is considered during a fixed validation-only tuning step. The test set does not select the correction or its strength.')
    st.latex(r'\ell_{\mathrm{improved}} = \ell_{\mathrm{paper}} + \alpha f_{\mathrm{history}}(x_{\leq t}, h_t),\quad \alpha\in[0,1]')
    st.dataframe(score_table(report), hide_index=True, width='stretch')
    interval = report.get('paired_patient_delta_ci95', {})
    if interval.get('lower') is not None:
        st.caption(f"AUROC change: {gain:+.2f} points; paired patient bootstrap 95% interval "
                   f"[{100 * interval['lower']:+.2f}, {100 * interval['upper']:+.2f}] points. "
                   'The interval is conditional on this fitted model and holdout; it does not establish performance on other cohorts.')
    st.caption(f"Maximum {report['epochs']} epochs · patient-level splits · training-only scaling · validation-only thresholds. A 4–5 point AUROC gain is a target, not a promised outcome.")
    st.subheader('Measured curves')
    choices = {'discrimination': 'ROC, precision–recall and calibration', 'confusion': 'Confusion matrices',
               'severity': 'Severity errors', 'training': 'Backbone training curve'}
    chosen = st.radio('Chart', list(choices), format_func=choices.get, horizontal=True)
    st.image(str(folder / 'figures' / f'{chosen}.png'), width='stretch')
    st.subheader('What each method predicted')
    prediction_view(folder, 'improvement_patient')
    st.download_button('Download held-out predictions', (folder / 'predictions.csv').read_bytes(), 'paired_predictions.csv', 'text/csv')
    with st.expander('Original paper source · historical reference'):
        st.caption('These source figures belong to the original publication; training a new model does not change them.')
        paper_reference()


def explorer_page():
    heading('03', 'Inside the representation', 'An embedding-projector-style view of held-out patient visits. Rotate, zoom, search a patient and move through visit history.')
    folder, report = select_run('projector_cohort')
    from src.project.plots import projection_figure
    data = np.load(folder / 'projection.npz')
    c1, c2, c3 = st.columns([1, 1.4, 1.4])
    algorithm = c1.selectbox('Projection', ['tsne', 'pca'], format_func=lambda value: 't-SNE' if value == 'tsne' else 'PCA')
    color = c2.selectbox('Color by', ['improved_risk', 'paper_risk', 'outcome'],
                         format_func=lambda value: {'improved_risk': 'Proposed risk', 'paper_risk': 'Paper risk', 'outcome': 'Observed next flare'}[value])
    patient = c3.selectbox('Highlight patient', ['All patients', *np.unique(data['patient_id']).tolist()])
    latest = st.slider('Reveal visits up to', 1, max(2, int(data['visit'].max())), max(2, int(data['visit'].max())))
    st.plotly_chart(projection_figure(folder / 'projection.npz', algorithm, color,
        None if patient == 'All patients' else patient, latest), width='stretch')
    st.caption('Each point is a model embedding from one held-out visit. Both color views use the same coordinates. Projection distance has no clinical unit; this visualization is not evidence of causal separation.')


def random_page():
    heading('04', 'Dataset tests. Side by side.', 'Use actual patient visits or random cohorts. Compare Paper and Proposed predictions with known outcomes.')
    prepared = manifest()
    source = st.radio('Dataset source', ['Actual public dataset', 'Upload CSV', 'Enter patient data', 'Random datasets'],
                      horizontal=True, key='test_dataset_source')
    if source == 'Random datasets':
        random_cohort_tests(prepared)
    else:
        from app.dataset_testing import actual_data_page
        actual_data_page(source, prepared)


def random_cohort_tests(prepared):
    st.caption(f'Default: 3 cohorts · {RANDOM_PATIENTS} patients each · 6 visits · 20% added graph edges · maximum {EPOCHS} epochs. Results are cached.')
    with st.form('random_test'):
        c1, c2, c3 = st.columns([1, 1, 1.4])
        count = c1.number_input('Number of datasets', min_value=1, max_value=5, value=3)
        seed = c2.number_input('Dataset seed', min_value=0, max_value=1000000, value=42)
        with c3:
            st.write('')
            submit = st.form_submit_button('Generate and test both methods', type='primary')
    if submit:
        from src.project.pipeline import valid_artifacts
        use_prepared = (int(seed) == SEED and int(count) == len(prepared['random_runs'])
                        and all(valid_artifacts(ROOT / path) for path in prepared['random_runs']))
        if use_prepared:
            st.session_state['two_method_random_runs'] = prepared['random_runs']
            st.success('Loaded the verified cached comparison.')
        else:
            from filelock import FileLock, Timeout
            from src.project.pipeline import random_dataset, cached_pair
            lock = FileLock(str(RUNS / '.prepare.lock'))
            try:
                lock.acquire(timeout=0)
            except Timeout:
                st.info('Another preparation run is active. Its completed results remain available; try generating again when it finishes.')
                return
            try:
                folders = []
                with st.status('Preparing random-cohort tests…', expanded=True) as status:
                    for index in range(int(count)):
                        current_seed = int(seed) + index * 17
                        sd, adjacency = random_dataset(index, current_seed)
                        folder = cached_pair(sd, {'random_dataset': index, 'seed': current_seed, 'patients': sd.N,
                            'visits': sd.T, 'edge_noise': .2}, adjacency=adjacency, progress=st.write)
                        folders.append(str(folder.relative_to(ROOT)))
                    status.update(label='Both methods evaluated. Results are cached.', state='complete', expanded=False)
                st.session_state['two_method_random_runs'] = folders
            finally:
                lock.release()
    folders = st.session_state.get('two_method_random_runs', prepared['random_runs'])
    rows = []
    for index, folder in enumerate(folders):
        report = json_file(ROOT / folder / 'results.json')
        row = {'Dataset': index + 1}
        for metric in ('auroc', 'auprc', 'f1', 'mae'):
            row[f'Paper {metric.upper()}'] = report['metrics']['paper'][metric]
            row[f'Proposed {metric.upper()}'] = report['metrics']['improved'][metric]
        row['AUROC change (points)'] = report['delta_auroc'] * 100
        rows.append(row)
    st.subheader('Test scores')
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    selected = st.selectbox('Inspect dataset predictions', range(len(folders)), format_func=lambda index: f'Dataset {index + 1}')
    folder = ROOT / folders[selected]
    prediction_view(folder, 'random_patient')
    st.download_button('Download both methods’ predictions', (folder / 'predictions.csv').read_bytes(),
                       f'dataset_{selected + 1}_predictions.csv', 'text/csv')

