"""Generate measured research charts and shared-coordinate 3D projections once."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate_figures(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
    from sklearn.calibration import calibration_curve
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from src.project.settings import METHODS

    folder = Path(folder)
    predictions = pd.read_csv(folder / 'predictions.csv')
    report = json.loads((folder / 'results.json').read_text())
    colors = {'paper': '#59677c', 'improved': '#168d80'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
    figures = folder / 'figures'
    figures.mkdir(exist_ok=True)
    y = predictions.actual_flare
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for key, label in METHODS.items():
        risk = predictions[f'{key}_risk']
        if y.nunique() == 2:
            fpr, tpr, _ = roc_curve(y, risk)
            axes[0].plot(fpr, tpr, label=f"{label}: {report['metrics'][key]['auroc']:.3f}", color=colors[key])
            precision, recall, _ = precision_recall_curve(y, risk)
            axes[1].plot(recall, precision, color=colors[key], label=label)
        observed, predicted = calibration_curve(y, risk, n_bins=8, strategy='quantile')
        axes[2].plot(predicted, observed, 'o-', color=colors[key], label=label)
    axes[0].plot([0, 1], [0, 1], ':', color='#b0b7c1')
    axes[1].axhline(y.mean(), ls=':', color='#b0b7c1')
    axes[2].plot([0, 1], [0, 1], ':', color='#b0b7c1')
    for ax, title, xlabel, ylabel in zip(axes, ['ROC', 'Precision–recall', 'Calibration'],
            ['False positive rate', 'Recall', 'Mean predicted risk'], ['True positive rate', 'Precision', 'Observed flare rate']):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=.15)
    fig.savefig(figures / 'discrimination.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, (key, label) in zip(axes, METHODS.items()):
        matrix = confusion_matrix(y, predictions[f'{key}_prediction'], labels=[0, 1])
        ax.imshow(matrix, cmap='Greens', vmin=0)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha='center', va='center', color='#15252e')
        ax.set(title=label, xlabel='Predicted flare', ylabel='Actual flare', xticks=[0, 1], yticks=[0, 1])
    fig.savefig(figures / 'confusion.png', dpi=170)
    plt.close(fig)
    history = pd.DataFrame(report['history'])
    fig, ax = plt.subplots(figsize=(8, 3), layout='constrained')
    ax.plot(history.epoch, history.train_loss, label='Training', color=colors['paper'])
    ax.plot(history.epoch, history.val_loss, label='Validation', color=colors['improved'])
    ax.set(xlabel='Epoch', ylabel='Shared neural objective', title='Paper backbone training')
    ax.legend()
    fig.savefig(figures / 'training.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3), layout='constrained')
    for key, label in METHODS.items():
        error = np.abs(predictions[f'{key}_severity'] - predictions.actual_severity)
        ax.hist(error, bins=20, alpha=.5, label=label, color=colors[key])
    ax.set(xlabel='Absolute severity error', ylabel='Held-out visits', title='Severity prediction')
    ax.legend()
    fig.savefig(figures / 'severity.png', dpi=170)
    plt.close(fig)
    # The same embedding coordinates are used when switching prediction colors.
    data = np.load(folder / 'embeddings.npz')
    order = np.random.default_rng(42).choice(len(data['outcome']), min(1800, len(data['outcome'])), replace=False)
    h = data['representation'][order]
    components = min(3, h.shape[0], h.shape[1])
    pca = PCA(n_components=components).fit_transform(h)
    pca = np.pad(pca, ((0, 0), (0, 3 - components)))
    if len(h) >= 8:
        tsne = TSNE(n_components=3, perplexity=min(30, max(2, (len(h) - 1) // 3)),
                    init='pca', learning_rate='auto', random_state=42, max_iter=500).fit_transform(h)
    else:
        tsne = pca.copy()
    result = {key: data[key][order] for key in data.files if key != 'representation'}
    np.savez_compressed(folder / 'projection.npz', pca=pca, tsne=tsne, **result)


def projection_figure(path, algorithm='tsne', color='improved_risk', patient=None, visit=None):
    """Embedding-projector-style exploration; distances have no clinical units."""
    import plotly.graph_objects as go
    data = np.load(path)
    xyz = data[algorithm]
    keep = np.ones(len(xyz), dtype=bool) if visit is None else data['visit'] <= visit
    figure = go.Figure()
    figure.add_trace(go.Scatter3d(x=xyz[keep, 0], y=xyz[keep, 1], z=xyz[keep, 2], mode='markers',
        customdata=np.column_stack([data['patient_id'][keep], data['visit'][keep], data['outcome'][keep]]),
        hovertemplate='Patient %{customdata[0]}<br>Visit %{customdata[1]}<br>Observed flare %{customdata[2]}<extra></extra>',
        marker=dict(size=3, opacity=.7, color=data[color][keep], colorscale='Tealrose', cmin=0, cmax=1,
                    colorbar=dict(title='Flare / risk')), name='Held-out visits'))
    if patient is not None:
        idx = np.flatnonzero(keep & (data['patient_id'] == patient))
        idx = idx[np.argsort(data['visit'][idx])]
        figure.add_trace(go.Scatter3d(x=xyz[idx, 0], y=xyz[idx, 1], z=xyz[idx, 2], mode='lines+markers',
            marker=dict(size=6, color='#f4bd65'), line=dict(width=5, color='#f4bd65'), name=str(patient)))
        figure.frames = [go.Frame(name=str(step), traces=[1], data=[go.Scatter3d(
            x=xyz[idx[:step], 0], y=xyz[idx[:step], 1], z=xyz[idx[:step], 2], mode='lines+markers',
            marker=dict(size=6, color='#f4bd65'), line=dict(width=5, color='#f4bd65'))])
            for step in range(1, len(idx) + 1)]
        figure.update_layout(updatemenus=[dict(type='buttons', direction='left', x=.02, y=1.02,
            buttons=[dict(label='Play trajectory', method='animate', args=[None,
                dict(frame=dict(duration=500, redraw=True), transition=dict(duration=0), fromcurrent=True)]),
                dict(label='Pause', method='animate', args=[[None], dict(mode='immediate', frame=dict(duration=0, redraw=False))])])])
    figure.update_layout(height=680, margin=dict(l=0, r=0, t=15, b=0), paper_bgcolor='#101a26',
        font=dict(color='#d9e3ed'), scene=dict(bgcolor='#101a26',
            xaxis=dict(title='Embedding 1'), yaxis=dict(title='Embedding 2'), zaxis=dict(title='Embedding 3')),
        uirevision='patient-projector')
    return figure
