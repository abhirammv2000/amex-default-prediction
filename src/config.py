"""Central configuration for the AMEX Default Prediction project.

Paths are resolved relative to the repository root so the code runs the same
way regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "amex-default-prediction"          # original Kaggle CSVs
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"              # parquet + engineered features
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
# Figures live under reports/ (version-controlled) so they render in the README
# on GitHub; outputs/ holds large, regenerable artifacts and is git-ignored.
REPORTS_DIR = ROOT / "reports"
FIGURE_DIR = REPORTS_DIR / "figures"

# Raw files
TRAIN_CSV = RAW_DIR / "train_data.csv"
TEST_CSV = RAW_DIR / "test_data.csv"
TRAIN_LABELS_CSV = RAW_DIR / "train_labels.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "sample_submission.csv"

# Parquet (downcast) versions produced by convert_to_parquet.py
TRAIN_PARQUET = PROCESSED_DIR / "train_data.parquet"
TEST_PARQUET = PROCESSED_DIR / "test_data.parquet"

# Engineered feature tables produced by feature_engineering.py
TRAIN_FEATURES = PROCESSED_DIR / "train_features.parquet"
TEST_FEATURES = PROCESSED_DIR / "test_features.parquet"

for _d in (PROCESSED_DIR, MODEL_DIR, SUBMISSION_DIR, REPORTS_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
ID_COL = "customer_ID"
DATE_COL = "S_2"
TARGET_COL = "target"

# The 11 categorical features called out in the competition data description.
CATEGORICAL_FEATURES = [
    "B_30", "B_38", "D_114", "D_116", "D_117", "D_120",
    "D_126", "D_63", "D_64", "D_66", "D_68",
]

# Non-feature columns that must never be fed to the model directly.
NON_FEATURE_COLS = [ID_COL, DATE_COL]

# ---------------------------------------------------------------------------
# Reproducibility / CV
# ---------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5

# Chunk size used when streaming the giant CSVs (rows per chunk).
CSV_CHUNK_SIZE = 500_000
