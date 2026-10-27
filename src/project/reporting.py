"""Generate result documentation from completed runs, never score constants."""
import json
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from src.project.settings import ROOT, RUNS

START = '<!-- RUN_RESULTS:START -->'
END = '<!-- RUN_RESULTS:END -->'

LABELS = {'real': 'Public pediatric SLE cohort, GSE65391',
          'simulation': 'Simulated SLE / RA cohort'}


def comparison_snapshot(dashboard, root=ROOT):
    """Read verified completed results without copying patient records into docs."""
    from src.project.pipeline import valid_artifacts
    root = Path(root)
    cohorts = {}
    entries = list(dashboard['datasets'].items()) + [
        (f'random_{i + 1}', path) for i, path in enumerate(dashboard['random_runs'])]
    for name, relative in entries:
        folder = root / relative
        if not valid_artifacts(folder, training_only=True):
            raise ValueError(f'Cannot document unverified or incomplete run: {folder.name}')
        report_path, predictions_path = folder / 'results.json', folder / 'predictions.csv'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        with predictions_path.open(newline='', encoding='utf-8') as stream:
            predictions = list(csv.DictReader(stream))
        cohorts[name] = {
            'label': LABELS.get(name, f'Random cohort {name.split("_")[-1]}'),
            'run_id': folder.name,
            'completed_at': report['completed_at'],
            'patients': report['patients'], 'visits': report['visits'],
            'input_features': report['input_features'],
            'split_counts': {key: len(value) for key, value in report['splits'].items()},
            'scored_visits': len(predictions),
            'flare_visits': sum(int(float(row['actual_flare'])) for row in predictions),
            'epochs_completed': len(report['history']),
            'thresholds': report['thresholds'], 'metrics': report['metrics'],
            'delta_auroc': report['delta_auroc'],
            'paired_patient_delta_ci95': report.get('paired_patient_delta_ci95', {}),
            'correction': report['improvement'],
            'sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (report_path, predictions_path)},
        }
    return {'schema': 1, 'prepared_at': dashboard['prepared_at'],
            'documentation_generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'training_seed': dashboard['training_seed'], 'split_seed': dashboard['split_seed'],
            'max_epochs': dashboard['defaults']['epochs'],
            'methods': ['Paper framework', 'Proposed framework'], 'cohorts': cohorts}


def render_measured_results(snapshot):
    stamp = datetime.fromtimestamp(snapshot['prepared_at'], timezone.utc).astimezone(ZoneInfo('Asia/Kolkata'))
    lines = ['# Measured results: Paper framework versus Proposed framework', '',
        'Generated from hash-verified completed runs. Normal preparation refreshes this file and '
        '[the JSON snapshot](results/current_comparison.json); do not edit performance numbers manually.', '',
        f"Source preparation: **{stamp:%d %b %Y, %H:%M IST}**. Model seed **{snapshot['training_seed']}**, "
        f"patient / dataset split seed **{snapshot['split_seed']}**, maximum neural epochs **{snapshot['max_epochs']}**.", '',
        'These are paired local evaluations. The paper publication reports AUROC 0.867, AUPRC 0.726, '
        'F1 0.702, severity MAE 0.74, RMSE 0.95 and ECE 0.032 on its separate clinical cohort '
        '(supplied PDF, Tables 2-3 and IV.F, pages 14-15). Those values are historical references, '
        'not an additional model row or an evaluation on our test patients.', '',
        '![Paired local AUROC and severity MAE comparison](results/comparison.png)', '',
        '## Main-cohort AUROC improvement', '',
        '| Cohort | Paper framework | Proposed framework | Absolute change |', '|---|---:|---:|---:|']
    for name, cohort in snapshot['cohorts'].items():
        if name.startswith('random_'):
            continue
        p, q = cohort['metrics']['paper'], cohort['metrics']['improved']
        lines.append(f"| {cohort['label']} | {p['auroc']:.4f} | {q['auroc']:.4f} | {100 * cohort['delta_auroc']:+.2f} percentage points |")
    lines += ['', '`AUROC points = (Proposed AUROC - Paper AUROC) * 100`. A four-point gain '
              'means +0.04 AUROC; it does not mean a 4% relative gain. Cohorts are not pooled.', '']
    metrics = [('auroc', 'AUROC', True), ('auprc', 'AUPRC', True), ('f1', 'F1', True),
               ('mae', 'Severity MAE', False), ('rmse', 'Severity RMSE', False),
               ('brier', 'Brier', False), ('ece', 'ECE', False)]
    for name, cohort in snapshot['cohorts'].items():
        p, q = cohort['metrics']['paper'], cohort['metrics']['improved']
        split = cohort['split_counts']
        prevalence = cohort['flare_visits'] / max(1, cohort['scored_visits'])
        lines += [f"## {cohort['label']}", '',
            f"{cohort['patients']:,} total patients, {cohort['visits']:,} visits, {cohort['input_features']} inputs; "
            f"patient split **{split['train']} / {split['val']} / {split['test']}** (train / validation / test). "
            f"Held-out scoring uses **{cohort['scored_visits']:,} visit targets**, including "
            f"**{cohort['flare_visits']:,} flares ({prevalence:.1%})**. "
            f"The chosen neural training ran {cohort['epochs_completed']} epochs.", '',
            '| Metric | Better | Paper framework | Proposed framework | Proposed minus Paper |',
            '|---|---|---:|---:|---:|']
        worsened = []
        for key, label, higher in metrics:
            delta = q[key] - p[key]
            lines.append(f"| {label} | {'Higher' if higher else 'Lower'} | {p[key]:.4f} | {q[key]:.4f} | {delta:+.4f} |")
            if (higher and delta < 0) or (not higher and delta > 0):
                worsened.append(label)
        reduction = 100 * (p['mae'] - q['mae']) / p['mae'] if p['mae'] else None
        if reduction is not None:
            lines += ['', f'Approximate severity-MAE reduction: **{reduction:+.2f}%**, calculated from the reported rounded scores.']
        lines += ['', ('**Trade-offs:** ' + ', '.join(worsened) + ' worsened in this run.') if worsened
                  else 'None of the listed rounded metrics worsened in this run; that does not guarantee future gains.']
        interval = cohort['paired_patient_delta_ci95']
        if interval.get('lower') is not None:
            lines += ['', f"Paired patient-bootstrap 95% interval for AUROC change: "
                f"**[{100 * interval['lower']:+.2f}, {100 * interval['upper']:+.2f}] percentage points**, "
                f"from {interval['draws']} valid draws of 500 requested."]
            lines.append('The interval includes zero, so this holdout does not establish a reliable positive AUROC change.'
                         if interval['lower'] <= 0 <= interval['upper'] else
                         'This interval excludes zero conditional on the fitted models and this evaluation cohort; it is not external clinical confirmation.')
        correction = cohort['correction']
        lines += ['', f"Saved validation thresholds: Paper **{cohort['thresholds']['paper']:.3f}**, "
            f"Proposed **{cohort['thresholds']['improved']:.3f}**. Test outcomes did not choose these values.", '',
            '| Selected correction detail | Flare head | Severity head |', '|---|---:|---:|',
            f"| Strength | {correction['flare_strength']} | {correction['severity_strength']} |",
            f"| Trees | {correction['classifier_trees']} | {correction['severity_trees']} |",
            f"| Internal configuration ID (0-based) | {correction['selected_flare_configuration']} | {correction['selected_severity_configuration']} |", '',
            f"Run ID: `{cohort['run_id']}`. Its report and prediction hashes are recorded in the JSON snapshot.", '']
    lines += ['## What this means', '',
        'The observed gains compare Proposed against the shared local Paper backbone. They do not establish '
        'superiority on the publication\'s unavailable cohort or guarantee a 4-5-point gain on every new dataset. '
        'Random cohorts are controlled simulations, not independent clinical replications.', '',
        'The bootstrap resamples held-out patients for fixed models. It excludes training, tuning and '
        'population uncertainty. Validation is reused for early stopping, Platt scaling, head tuning, '
        'seed selection and thresholds; the frozen seed criterion favored Proposed validation AUROC. '
        'The holdout has been inspected during development. Independent / nested validation and a fresh '
        'clinical cohort remain necessary.', '',
        'See [Methodology](METHODOLOGY.md), [Architecture](CAUSAL_MODEL.md) and '
        '[Audit](PROJECT_AUDIT.md) for the implementation and limitations.', '']
    return '\n'.join(lines)


def write_comparison_figure(snapshot, path):
    """Render stored paired scores; no model inference or fitting takes place."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    cohorts = [cohort for name, cohort in snapshot['cohorts'].items() if not name.startswith('random_')]
    figure = Figure(figsize=(12, 4.6), dpi=180, facecolor='#f8faf9')
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    labels = ['Public SLE' if 'GSE65391' in cohort['label'] else cohort['label'].replace(' cohort', '')
              for cohort in cohorts]
    palette = {'paper': '#67798b', 'improved': '#168575'}
    for axis, key, title in zip(axes, ('auroc', 'mae'), ('Flare AUROC - higher is better', 'Severity MAE - lower is better')):
        for offset, (method, name) in zip((-.18, .18), (('paper', 'Paper framework'), ('improved', 'Proposed framework'))):
            values = [cohort['metrics'][method][key] for cohort in cohorts]
            bars = axis.bar([i + offset for i in range(len(cohorts))], values, width=.32,
                            color=palette[method], label=name, zorder=3)
            axis.bar_label(bars, labels=[f'{value:.4f}' if key == 'auroc' else f'{value:.3f}' for value in values],
                           padding=5, fontsize=10)
        axis.set_xticks(range(len(cohorts)), labels, fontsize=10)
        axis.set_title(title, fontsize=13, loc='left', pad=14)
        axis.set_facecolor('#f8faf9')
        maximum = max(cohort['metrics'][m][key] for cohort in cohorts for m in ('paper', 'improved'))
        axis.set_ylim(0, 1 if key == 'auroc' else max(1, 1.2 * maximum))
        axis.grid(axis='y', color='#dce5e2', linewidth=.7, zorder=0)
        for side in ('top', 'right', 'left'):
            axis.spines[side].set_visible(False)
        axis.spines['bottom'].set_color('#b8c8c3')
        axis.tick_params(axis='both', length=0)
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, 1), ncol=2, frameon=False)
    figure.text(.06, .025, f"Paired local holdouts | model seed {snapshot['training_seed']} | "
                f"split seed {snapshot['split_seed']} | authors' original clinical cohort is unavailable",
                fontsize=9, color='#536868')
    figure.subplots_adjust(top=.79, bottom=.16, left=.06, right=.98, wspace=.22)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=figure.get_facecolor())
    figure.clear()


def render_results(dashboard, root=ROOT):
    lines = [START, '', 'Generated from the latest completed run. Retraining updates this section automatically.', '',
        f"Frozen model seed: **{dashboard['training_seed']}**. Patient split seed: **{dashboard['split_seed']}**. "
        f"Maximum neural epochs: **{dashboard['defaults']['epochs']}**.", '',
        '| Evaluation cohort | Paper framework AUROC | Proposed framework AUROC | Change |',
        '|---|---:|---:|---:|']
    reports = {}
    labels = {'real': 'Public pediatric SLE cohort, GSE65391', 'simulation': 'Simulated SLE / RA cohort'}
    for name, folder in dashboard['datasets'].items():
        report = json.loads((root / folder / 'results.json').read_text(encoding='utf-8'))
        reports[name] = report
        paper, proposed = report['metrics']['paper'], report['metrics']['improved']
        lines.append(f"| {labels.get(name, name)} | {paper['auroc']:.4f} | {proposed['auroc']:.4f} | {report['delta_auroc'] * 100:+.2f} points |")
    lines.append('')
    if 'real' in reports:
        report = reports['real']
        p, q = report['metrics']['paper'], report['metrics']['improved']
        lines += [f"Public SLE holdout: AUPRC **{p['auprc']:.4f} → {q['auprc']:.4f}**, "
                  f"F1 **{p['f1']:.4f} → {q['f1']:.4f}**, severity MAE **{p['mae']:.3f} → {q['mae']:.3f}**.", '']
    for name, report in reports.items():
        interval = report.get('paired_patient_delta_ci95', {})
        if interval.get('lower') is not None:
            lines += [f"{labels.get(name, name)}: **{len(report['splits']['test'])} held-out patients**; "
                      f"paired patient bootstrap 95% interval for AUROC change: "
                      f"**{interval['lower'] * 100:+.2f} to {interval['upper'] * 100:+.2f} points**.", '']
    lines += ['The seed was chosen using validation scores, not test scores. These are exploratory local comparisons; '
              'performance can vary across datasets and the original paper cohort is unavailable.', '', END]
    return '\n'.join(lines)


def sync_results(dashboard, root=ROOT, runs=RUNS):
    root, runs = Path(root), Path(runs)
    snapshot = comparison_snapshot(dashboard, root)
    rendered = render_results(dashboard, root)
    runs.mkdir(parents=True, exist_ok=True)
    summary = runs / 'latest_results.md'
    if not summary.exists() or summary.read_text(encoding='utf-8') != rendered:
        summary.write_text(rendered, encoding='utf-8')
    for relative in ('README.md', 'docs/METHODOLOGY.md'):
        document = root / relative
        if not document.exists():
            continue
        text = document.read_text(encoding='utf-8')
        if START in text and END in text:
            before, remainder = text.split(START, 1)
            _, after = remainder.split(END, 1)
            updated = before + rendered + after
            if updated != text:
                document.write_text(updated, encoding='utf-8')
    outputs = {
        root / 'docs/MEASURED_RESULTS.md': render_measured_results(snapshot),
        root / 'docs/results/current_comparison.json': json.dumps(snapshot, indent=2, allow_nan=False) + '\n',
    }
    snapshot_changed = False
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text(encoding='utf-8') != content:
            path.write_text(content, encoding='utf-8')
            if path.suffix == '.json':
                snapshot_changed = True
    chart = root / 'docs/results/comparison.png'
    if snapshot_changed or not chart.exists():
        write_comparison_figure(snapshot, chart)


if __name__ == '__main__':
    index = RUNS / 'dashboard.json'
    if not index.exists():
        raise SystemExit('No completed dashboard index. Prepare it with uv run python run.py --prepare-only.')
    sync_results(json.loads(index.read_text(encoding='utf-8')))
    print('Refreshed README, methodology, full measured results and comparison snapshot from completed runs.')
