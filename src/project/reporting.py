"""Generate result documentation from completed runs, never score constants."""
import json

from src.project.settings import ROOT, RUNS

START = '<!-- RUN_RESULTS:START -->'
END = '<!-- RUN_RESULTS:END -->'


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
    rendered = render_results(dashboard, root)
    runs.mkdir(parents=True, exist_ok=True)
    summary = runs / 'latest_results.md'
    if not summary.exists() or summary.read_text(encoding='utf-8') != rendered:
        summary.write_text(rendered, encoding='utf-8')
    readme = root / 'README.md'
    if readme.exists():
        text = readme.read_text(encoding='utf-8')
        if START in text and END in text:
            before, remainder = text.split(START, 1)
            _, after = remainder.split(END, 1)
            updated = before + rendered + after
            if updated != text:
                readme.write_text(updated, encoding='utf-8')
