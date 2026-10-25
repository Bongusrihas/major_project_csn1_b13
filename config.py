"""Central configuration for the Longitudinal Patient Health Prediction project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

RAW_FILE = RAW_DIR / "GSE65391_series_matrix.txt.gz"   # optional, only needed to rebuild the CSV
VISITS_CSV = PROC_DIR / "visits_clean.csv"
ENSEMBLE_FILE = MODEL_DIR / "ensemble.joblib"

SEED = 42

# ---- Prediction target -------------------------------------------------------
# Primary target: "flare at the next record" = SLEDAI rises by >= FLARE_DELTA points
# between the current visit and the next available visit. (A rise of >= 4 is a
# commonly used SLEDAI flare threshold; change it here to run a sensitivity analysis.)
FLARE_DELTA = 4

# A prediction needs at least this many records (current + earlier) so that
# between-visit changes can be computed.
MIN_HISTORY = 2

# ---- Variables ---------------------------------------------------------------
LAB_COLS = ["wbc", "lymphocyte_count", "neutrophil_count", "platelet_count", "esr",
            "hgb", "cr", "alb", "ds_dna", "c3", "c4", "ast", "alt", "ldh"]
TREAT_COLS = ["steroid_iv_category", "oral_steroids_category", "cyclophosphamide_category",
              "mycophenolate_category", "hydroxychloroquine_category",
              "metotrexate_category", "nsaid_category", "asa_category"]
SLEDAI_FLAGS = ["renal", "musculoskeletal", "serology", "fever",
                "thrombocytopenia", "leukopenia"]

# Variables for which trend features (previous value, change, slope, rolling mean) are built
TREND_COLS = ["sledai", "c3", "c4", "ds_dna", "esr", "wbc", "hgb",
              "platelet_count", "cr", "alb", "lymphocyte_count"]

# Demographics. Race/gender are NOT used by default (small subgroups, fairness concerns).
USE_DEMOGRAPHICS = False

# Columns in the raw GEO file that must NEVER be used as features because they
# describe the patient's whole follow-up (future information) or are derived from it.
LEAKY_COLUMNS = [
    "visit_count",                              # total number of visits of the patient
    "days_between_diagnosis_and_last_visit",    # span up to the patient's LAST visit
    "biopsy_history", "nephritis_class",        # assigned from biopsies, some after the visit
    "days_since_kidney_biopsy",                 # negative for visits BEFORE the biopsy
    "set", "batch", "batch_replicate",          # study split / technical columns
    "disease_activity", "mdg", "treatment", "treatment_lmm1",
    "sledai_component_class", "sledaic_lmm2", "neph_treat_lmm3",  # derived groupings
]

# ---- Evaluation --------------------------------------------------------------
HOLDOUT_FRACTION_FOLDS = 4      # 1 of 4 patient folds (~25 % of patients) = untouched test set
CV_SPLITS = 5
CV_REPEATS = 3                  # repeated patient-grouped CV used for model selection
FULL_CV_REPEATS = 5             # repeated grouped CV on all patients (static vs longitudinal)
N_BOOT = 1000                   # patient-level bootstrap resamples for confidence intervals
N_ENSEMBLE = 60                 # bootstrap ensemble used for uncertainty in the app
