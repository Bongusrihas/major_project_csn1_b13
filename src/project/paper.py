"""Published statistics, transcribed from Tables 2–4 of the supplied paper."""
import json
from pathlib import Path

from src.project.settings import PAPER_DIR

REFERENCE = {
    'title': 'Causal Representation Learning for Predicting Autoimmune Disease Progression From Longitudinal Multimodal Clinical Data',
    'authors': 'Harshit Sharma and Simran Kaur',
    'journal': 'IEEE Access 13, 150010–150027 (2025)',
    'doi': 'https://doi.org/10.1109/ACCESS.2025.3598276',
    'license': 'CC BY 4.0',
    'cohort_patients': 2000, 'visit_instances_approx': 33100, 'features': 54,
    'ece': .032,
    'tables': [
        {'Model': 'LSTM', 'AUROC': .782, 'AUPRC': .635, 'F1': .610, 'MAE': .89, 'RMSE': 1.11},
        {'Model': 'Transformer', 'AUROC': .804, 'AUPRC': .660, 'F1': .631, 'MAE': .85, 'RMSE': 1.05},
        {'Model': 'Variational RNN', 'AUROC': .820, 'AUPRC': .676, 'F1': .648, 'MAE': .83, 'RMSE': 1.01},
        {'Model': 'Paper framework', 'AUROC': .867, 'AUPRC': .726, 'F1': .702, 'MAE': .74, 'RMSE': .95},
    ],
    'clinical_table': [
        {'Model': 'SLEDAI', 'AUROC': .752, 'AUPRC': .598, 'F1': .580, 'MAE': 1.15},
        {'Model': 'DAS28', 'AUROC': .764, 'AUPRC': .610, 'F1': .595, 'MAE': 1.10},
        {'Model': 'Paper framework', 'AUROC': .867, 'AUPRC': .726, 'F1': .702, 'MAE': .74},
    ],
    'figures': {
        'fig6': 'Latent causal and spurious representations',
        'fig7': 'Uncertainty heatmap',
        'fig8': 'DMARD-withdrawal counterfactual',
        'fig9': 'Subgroup ROC curves',
        'fig10': 'Temporal feature attention',
        'fig11': 't-SNE by flare outcome',
        'fig12': 'Uncertainty over time',
        'fig13': 'Component ablations',
    },
}


def prepare_paper():
    """Crop original graphics once; never synthesize original-paper curves."""
    import pymupdf
    import hashlib
    pdf = PAPER_DIR / 'reference.pdf'
    if not pdf.exists():
        raise FileNotFoundError('Place the reference paper at docs/paper/reference.pdf; the original figures cannot be fabricated.')
    document = pymupdf.open(pdf)
    version = hashlib.sha256(pdf.read_bytes() + Path(__file__).read_bytes()).hexdigest()
    stamp = PAPER_DIR / '.render.sha256'
    refresh = not stamp.exists() or stamp.read_text() != version
    crops = {
        'table2': (13, (34, 61, 282, 144)),
        'table3': (13, (34, 194, 282, 270)),
        'table4': (14, (296, 397, 557, 459)),
        'fig8': (13, (35, 460, 282, 600)),
        'fig9': (13, (296, 61, 557, 274)),
        'fig10': (13, (296, 414, 557, 548)),
        'fig11': (14, (35, 61, 283, 235)),
        'fig12': (14, (35, 416, 283, 572)),
        'fig13': (14, (296, 61, 557, 227)),
    }
    # Preserve complete source pages for figures 6/7, whose layout differs.
    for index in (9, 11):
        page = document[index]
        blocks = page.get_text('blocks')
        captions = [block for block in blocks if block[4].startswith(('FIGURE 6.', 'FIGURE 7.'))]
        for block in captions:
            number = block[4].split('.')[0].split()[-1]
            # Find the raster graphic immediately above its caption.
            candidates = [info['bbox'] for info in page.get_image_info()
                          if info['bbox'][3] <= block[1] + 2 and abs(info['bbox'][0] - block[0]) < 45]
            if candidates:
                box = min(candidates, key=lambda value: abs(value[3] - block[1]))
                # A figure may be composed of multiple adjacent image fragments.
                fragments = [info['bbox'] for info in page.get_image_info()
                             if abs(info['bbox'][1] - box[1]) < 3 and abs(info['bbox'][3] - box[3]) < 3]
                crops[f'fig{number}'] = (index, (min(block[0], *(item[0] for item in fragments)) - 2,
                    min(item[1] for item in fragments) - 9,
                    max(block[2], *(item[2] for item in fragments)) + 2, block[3] + 3))
    for name, (index, coordinates) in crops.items():
        path = PAPER_DIR / f'{name}.png'
        if refresh or not path.exists():
            document[index].get_pixmap(matrix=pymupdf.Matrix(2.5, 2.5), clip=pymupdf.Rect(coordinates)).save(path)
    stamp.write_text(version)
    metadata = PAPER_DIR / 'statistics.json'
    metadata.write_text(json.dumps(REFERENCE, indent=2), encoding='utf-8')
    (PAPER_DIR / 'ATTRIBUTION.md').write_text(
        '# Original paper graphics\n\nHarshit Sharma and Simran Kaur (2025), ' + REFERENCE['title'] +
        '. IEEE Access 13, 150010–150027. ' + REFERENCE['doi'] +
        '\n\nLicensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). '
        'Figures and tables are cropped from the user-supplied published PDF. Cropping and rasterization are the only changes. '
        'Published curves are shown as source images; no individual-level ROC/PR data have been invented.\n', encoding='utf-8')
    return REFERENCE
