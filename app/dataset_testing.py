"""Actual-data prediction UI; uploaded records/results stay in this session."""
import hashlib
import json
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.project.custom_data import (FLARE_ALIASES, SEVERITY_ALIASES, MAX_ROWS,
    actual_heldout_records, input_template, prepare_records, compare_results)
from src.project.settings import ROOT, METHODS


def model_version(folder):
    return tuple((name, (folder / name).stat().st_mtime_ns, (folder / name).stat().st_size)
                 for name in ('paper.pt', 'improved.joblib', 'results.json', 'complete.json'))


@st.cache_resource(show_spinner=False, max_entries=2)
def model_bundle(folder, version):
    from src.project.custom_data import load_bundle
    return load_bundle(folder)


def csv_bytes(frame):
    return frame.to_csv(index=False).encode('utf-8')


def instructions(report, example, dataset):
    features = report['feature_names']
    with st.expander('Input format, column details and CSV templates'):
        st.markdown('One row is one patient visit. Include **all available earlier visits** for each patient, '
                    'in chronological order. Both frameworks predict a flare at the **following visit**.')
        st.dataframe(pd.DataFrame([
            {'Column': 'patient_id', 'Enter': 'A consistent patient identifier; use anonymous IDs.'},
            {'Column': 'visit', 'Enter': 'Positive whole-number visit order; unique within a patient.'},
            {'Column': 'gap_days', 'Enter': 'Days since the previous supplied visit; 0 for the first visit.'},
            {'Column': 'Clinical inputs below', 'Enter': 'Numeric raw values in the training source’s units/coding. Leave missing values blank.'},
            {'Column': 'actual_next_flare (optional)', 'Enter': 'Yes / No, 1 / 0 or True / False; blank means unknown.'},
            {'Column': 'actual_next_severity (optional)', 'Enter': 'Known following-visit severity category 0–4; blank means unknown.'},
        ]), hide_index=True, width='stretch')
        st.caption('Aliases: subject → patient_id; visit_no → visit; zero-based visit_index → visit. '
                   'cumulative_time in days can replace gap_days. Maximum 10,000 rows and 128 visits per patient. '
                   'At least one recognized measurement is required per row; more complete histories match training better.')
        rows = []
        for name in features:
            values = example[name].dropna().unique()[:4] if name in example else []
            kind = ('Source category code' if name.endswith('_category') else
                    'Source symptom flag' if name in ('renal', 'musculoskeletal', 'serology', 'fever',
                                                     'thrombocytopenia', 'leukopenia') else 'Numeric measurement')
            rows.append({'Input column': name, 'Format': kind,
                         'Actual source examples': ', '.join(f'{v:g}' for v in values)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.caption('Source examples describe encoding, not suggested patient values. Do not substitute medication doses '
                   'for category codes or convert laboratory units without matching the training source. '
                   'Current measurements and following-visit outcomes must use separate columns.')
        if dataset == 'real':
            st.caption('Public-cohort severity uses the same saved evaluation definition: '
                       'SLEDAI 0, 1–5, 6–10, 11–19, ≥20 map to categories 0–4. '
                       'Current SLEDAI is an input; actual_next_severity is the following-visit category.')
        st.download_button('Download blank input CSV', csv_bytes(input_template(features)),
                           'patient_input_template.csv', 'text/csv', key='input_template_download')
        st.download_button('Download column guide', csv_bytes(pd.DataFrame(rows)),
                           'input_column_guide.csv', 'text/csv', key='column_guide_download')


def outcome_controls(frame, features, key):
    choices = ['No outcome column'] + [name for name in frame.columns if name not in features
        and name not in ('patient_id', 'subject', 'visit', 'visit_no', 'visit_index', 'visit_number',
                         'gap_days', 'cumulative_time')]
    def default(aliases):
        return next((choices.index(name) for name in aliases if name in choices), 0)
    a, b = st.columns(2)
    flare = a.selectbox('Actual flare column (optional)', choices, index=default(FLARE_ALIASES), key=f'{key}_flare')
    severity = b.selectbox('Actual severity column (optional)', choices, index=default(SEVERITY_ALIASES), key=f'{key}_severity')
    timing = st.radio('What do these outcome columns describe?', ['next', 'current'],
        format_func=lambda x: 'Next visit — label is already on the prediction row' if x == 'next' else
                             'Current visit — shift labels back to check the preceding prediction',
        key=f'{key}_timing')
    st.caption('Example: a Yes recorded on visit 2 checks the visit 1 prediction in current-visit mode. '
               'The final visit then has no known future outcome. With no labels, both models still predict.')
    return (None if flare == choices[0] else flare, None if severity == choices[0] else severity, timing)


def result_view(predictions, report, key):
    include_seen = st.checkbox('Include training / validation patients in summary (in-sample demonstration)',
                               value=False, key=f'{key}_include_seen')
    rows, metrics = compare_results(predictions, report, include_seen=include_seen)
    scored = rows.used_for_flare_metrics
    a, b, c = st.columns(3)
    a.metric('Patient visits predicted', len(rows))
    b.metric('Visits with actual flare labels', int(rows.actual_next_flare.ne('Unknown').sum()))
    c.metric('Visits scored in summary', int(scored.sum()))
    st.subheader('Paper and Proposed · measured on your outcomes')
    st.dataframe(metrics[['Method', 'Accuracy', 'AUROC', 'AUPRC', 'F1', 'Precision', 'Recall', 'Brier']],
        hide_index=True, width='stretch', column_config={
        name: st.column_config.NumberColumn(format='%.4f') for name in
        ('Accuracy', 'AUROC', 'AUPRC', 'F1', 'Precision', 'Recall', 'Brier', 'Severity MAE', 'Severity RMSE')})
    st.caption('Higher is better: Accuracy, AUROC, AUPRC, F1, Precision and Recall. '
               'Lower is better: Brier, severity MAE and RMSE. Blank scores are undefined or have no supplied outcomes. '
               'AUROC needs both Yes and No labels. Summary scores exclude recognized training and validation patients by default; '
               'renaming their IDs does not make them independent test patients.')
    if metrics['Scored severity visits'].max() > 0:
        st.dataframe(metrics[['Method', 'Scored severity visits', 'Severity MAE', 'Severity RMSE']],
                     hide_index=True, width='stretch', column_config={
                         name: st.column_config.NumberColumn(format='%.4f') for name in ('Severity MAE', 'Severity RMSE')})
    if not scored.any():
        st.info('No eligible flare outcomes to score. Add actual Yes/No outcomes; unknown outcomes are never treated as No.')
    else:
        counts = rows.loc[scored, 'who_was_correct'].value_counts().reindex(
            ['Paper only', 'Proposed only', 'Both', 'Neither'], fill_value=0)
        st.bar_chart(counts.rename('Visits'), horizontal=True, color='#208574')
    thresholds = report['thresholds']
    st.caption(f"Saved validation thresholds: Paper {thresholds['paper']:.3f}; Proposed {thresholds['improved']:.3f}. "
               'Uploaded outcomes never change either threshold or train a model.')
    st.subheader('Which framework was right?')
    patient = st.selectbox('Patient', ['All patients', *rows.patient_id.unique()], key=f'{key}_patient')
    view = rows if patient == 'All patients' else rows[rows.patient_id == patient]
    only_labeled = st.checkbox('Show only visits with known flare outcomes', key=f'{key}_labeled')
    if only_labeled:
        view = view[view.actual_next_flare.ne('Unknown')]
    columns = ['patient_id', 'visit', 'actual_next_flare', 'paper_flare', 'improved_flare',
               'who_was_correct', 'paper_risk', 'improved_risk']
    labels = {'patient_id': 'Patient', 'visit': 'Visit', 'actual_next_flare': 'Actual flare',
              'paper_risk': 'Paper risk', 'paper_flare': 'Paper prediction', 'paper_correct': 'Paper correct?',
              'improved_risk': 'Proposed risk', 'improved_flare': 'Proposed prediction',
              'improved_correct': 'Proposed correct?', 'who_was_correct': 'Who was right?',
              'patient_exposure': 'Patient exposure', 'used_for_flare_metrics': 'Scored in summary?'}
    st.dataframe(view[columns].rename(columns=labels), hide_index=True, width='stretch', column_config={
        'Paper risk': st.column_config.NumberColumn(format='percent'),
        'Proposed risk': st.column_config.NumberColumn(format='percent')})
    if patient != 'All patients' and len(view):
        fig = go.Figure()
        for method, label in METHODS.items():
            fig.add_scatter(x=view.visit, y=view[f'{method}_risk'], name=label, mode='lines+markers')
        truth = view.actual_next_flare.map({'Yes': 1., 'No': 0.})
        fig.add_scatter(x=view.visit, y=truth, name='Actual next flare', mode='markers',
                        marker=dict(symbol='x', size=11, color='#233637'))
        fig.update_layout(xaxis_title='Prediction visit', yaxis_title='Following-visit flare probability',
                          yaxis=dict(range=[-.05, 1.05]), height=330)
        st.plotly_chart(fig, width='stretch')
    with st.expander('Severity predictions and absolute errors'):
        st.dataframe(view[['patient_id', 'visit', 'actual_next_severity', 'paper_severity', 'improved_severity',
                           'paper_absolute_error', 'improved_absolute_error', 'closer_severity']],
                     hide_index=True, width='stretch')
    with st.expander('Correctness details and patient exposure'):
        st.dataframe(view[['patient_id', 'visit', 'history_visits', 'paper_correct', 'improved_correct',
                           'patient_exposure', 'used_for_flare_metrics']], hide_index=True, width='stretch')
    st.download_button('Download both predictions and correctness', csv_bytes(rows),
                       'paper_proposed_patient_comparison.csv', 'text/csv', key=f'{key}_download_results')
    st.download_button('Download comparison scores', csv_bytes(metrics),
                       'paper_proposed_scores.csv', 'text/csv', key=f'{key}_download_scores')


def evaluate(frame, folder, report, flare, severity, timing, key):
    """Only cache patient predictions in the current browser session."""
    version = model_version(folder)
    signature = hashlib.sha256(csv_bytes(frame) + repr((str(folder), version, flare, severity, timing)).encode()).hexdigest()
    state_key = f'{key}_prediction_result'
    cached = st.session_state.get(state_key)
    if cached and cached['signature'] != signature:
        del st.session_state[state_key]
    try:
        records, details = prepare_records(frame, report['feature_names'], flare, severity, timing)
    except (ValueError, TypeError, OverflowError) as error:
        st.error(str(error))
        return
    st.caption(f"{len(records):,} visits · {records.patient_id.nunique()} patients · "
               f"{len(details['present_features'])}/{len(report['feature_names'])} input columns supplied. "
               'Missing measurements remain missing; the saved training scaler is reused.')
    if details['missing_features']:
        with st.expander('Missing input columns'):
            st.write(', '.join(details['missing_features']))
    if st.button('Predict with Paper and Proposed', type='primary', key=f'{key}_predict'):
        try:
            with st.spinner('Running both saved frameworks on these visits…'):
                from src.project.custom_data import predict_records
                bundle = model_bundle(str(folder), version)
                predictions = predict_records(records, bundle)
            st.session_state[state_key] = {'signature': signature, 'predictions': predictions}
        except (ValueError, OSError, RuntimeError) as error:
            st.error(f'Prediction could not complete: {error}')
            return
    cached = st.session_state.get(state_key)
    if cached and cached['signature'] == signature:
        result_view(cached['predictions'], report, key)


def actual_data_page(source, prepared):
    labels = {'real': 'Public SLE model · actual clinical inputs',
              'simulation': 'Simulation model · simulated input format'}
    dataset = 'real' if source == 'Actual public dataset' else st.selectbox(
        'Use the model trained for this input format', list(prepared['datasets']),
        format_func=lambda value: labels[value], key='custom_model_format')
    folder = ROOT / prepared['datasets'][dataset]
    report = json.loads((folder / 'results.json').read_text(encoding='utf-8'))
    real_folder = ROOT / prepared['datasets']['real']
    real_report = json.loads((real_folder / 'results.json').read_text(encoding='utf-8'))
    example = actual_heldout_records(real_report, ROOT) if dataset == 'real' else pd.DataFrame()
    st.caption(f"Saved model run: {folder.name} · model seed {report['seed']}. "
               'Your CSV and entered patient records are used in memory for this session; no patient file is saved by this page.')
    instructions(report, example, dataset)
    if source == 'Actual public dataset':
        st.subheader('Actual public SLE dataset · held-out patients')
        st.write('These are the real GSE65391 records reserved for testing. Actual next-visit flare is '
                 'derived from a SLEDAI increase of at least 4; the final visit has no known next outcome.')
        st.caption(f"{example.patient_id.nunique()} held-out patients · {len(example)} visits. "
                   'The original paper’s private clinical cohort is unavailable; this is our public evaluation cohort.')
        st.dataframe(example, hide_index=True, width='stretch', height=230)
        st.download_button('Download actual dataset with known outcomes', csv_bytes(example),
                           'actual_SLE_heldout_patients.csv', 'text/csv')
        evaluate(example, folder, report, 'actual_next_flare', 'actual_next_severity', 'next', 'public')
    elif source == 'Upload CSV':
        if dataset == 'real':
            st.download_button('Download filled example from actual held-out patients', csv_bytes(example),
                               'actual_SLE_example.csv', 'text/csv')
        upload = st.file_uploader('Upload patient visits as CSV', type=['csv'], key='patient_csv', max_upload_size=10)
        if upload is None:
            st.info('Download the blank template above, enter your measurements and optional actual outcomes, then upload it here.')
            return
        if upload.size > 10 * 1024 * 1024:
            st.error('CSV size must be at most 10 MB.')
            return
        try:
            frame = pd.read_csv(BytesIO(upload.getvalue()), nrows=MAX_ROWS + 1,
                                dtype={'patient_id': 'string', 'subject': 'string'})
        except (ValueError, UnicodeError, pd.errors.ParserError) as error:
            st.error(f'Cannot read this CSV: {error}')
            return
        frame.columns = frame.columns.astype(str).str.strip()
        st.dataframe(frame.head(50), hide_index=True, width='stretch', height=220)
        file_key = 'upload_' + hashlib.sha256(upload.getvalue()).hexdigest()[:12]
        flare, severity, timing = outcome_controls(frame, report['feature_names'], file_key)
        evaluate(frame, folder, report, flare, severity, timing, 'upload')
    else:
        st.subheader('Enter actual patient visits')
        defaults = [name for name in ('sledai', 'wbc', 'esr', 'ds_dna', 'c3', 'c4') if name in report['feature_names']]
        if not defaults:
            defaults = report['feature_names'][:5]
        features = st.multiselect('Clinical input columns to enter', report['feature_names'],
                                 default=defaults, key=f'manual_features_{dataset}')
        if not features:
            st.info('Choose at least one clinical input column.')
            return
        blank = input_template(features)
        blank['actual_next_severity'] = np.nan
        for name in features:
            blank[name] = np.nan
        blank['gap_days'] = blank.gap_days.astype(float)
        st.caption('Add one row per visit using the table’s + row control. Leave unknown outcomes blank. '
                   'For multiple visits, repeat the patient ID and increase the visit number.')
        editor_key = f'manual_editor_{dataset}_' + hashlib.sha256('|'.join(features).encode()).hexdigest()[:8]
        frame = st.data_editor(blank, num_rows='dynamic', hide_index=True, width='stretch', key=editor_key,
            column_config={'actual_next_flare': st.column_config.SelectboxColumn('Actual flare', options=['Yes', 'No']),
                'actual_next_severity': st.column_config.NumberColumn('Actual severity (optional)', min_value=0, max_value=4, step=1),
                'visit': st.column_config.NumberColumn(min_value=1, step=1),
                'gap_days': st.column_config.NumberColumn(min_value=0)})
        timing = st.radio('Entered outcomes describe', ['next', 'current'],
            format_func=lambda x: 'Following visit (next-visit outcome)' if x == 'next' else 'This visit (shift to previous prediction)',
            horizontal=True, key=f'manual_timing_{dataset}')
        if not frame[features].notna().any().any():
            st.info('Enter at least one clinical measurement in each visit row to enable prediction.')
            return
        evaluate(frame, folder, report, 'actual_next_flare', 'actual_next_severity', timing, f'manual_{dataset}')
