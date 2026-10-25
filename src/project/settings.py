"""Fixed defaults shared by the one-command launcher and dashboard."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
SEED = 42
# Dataset generation and the patient split stay fixed during seed selection.
SEED_CANDIDATES = (7, 21, 42, 73, 123)
FROZEN_SEED_FILE = ROOT / 'config' / 'frozen_seed.json'
RANDOM_DATASETS = 3
RANDOM_PATIENTS = 120
RANDOM_VISITS = 6
EDGE_NOISE = 0.20
METHODS = {"paper": "Paper framework", "improved": "Proposed framework"}
RUNS = ROOT / "reports" / "project"
PAPER_DIR = ROOT / "docs" / "paper"


def training_seed():
    """Use the once-selected model seed; the dataset/split seed remains SEED."""
    import json
    if FROZEN_SEED_FILE.exists():
        record = json.loads(FROZEN_SEED_FILE.read_text(encoding='utf-8'))
        selected = record['selected_seed']
        if isinstance(selected, bool) or not isinstance(selected, int):
            raise ValueError('Frozen training seed must be an integer')
        return selected
    return SEED
