"""Settings for the Amex default prediction project. Paths are relative to the repo root, so it doesn't matter
where you run from."""
from __future__ import annotations

from pathlib import Path

# paths
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "amex-default-prediction"          # original Kaggle CSVs
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"              # parquet + engineered features
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
# figures go in reports/ so they show up in the README on GitHub, outputs/ is for big files that can be regenerated (gitignored)
REPORTS_DIR = ROOT / "reports"
FIGURE_DIR = REPORTS_DIR / "figures"

# raw files
TRAIN_CSV = RAW_DIR / "train_data.csv"
TEST_CSV = RAW_DIR / "test_data.csv"
TRAIN_LABELS_CSV = RAW_DIR / "train_labels.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "sample_submission.csv"

# parquet versions from convert_to_parquet.py
TRAIN_PARQUET = PROCESSED_DIR / "train_data.parquet"
TEST_PARQUET = PROCESSED_DIR / "test_data.parquet"

# feature tables from feature_engineering.py
TRAIN_FEATURES = PROCESSED_DIR / "train_features.parquet"
TEST_FEATURES = PROCESSED_DIR / "test_features.parquet"

for _d in (PROCESSED_DIR, MODEL_DIR, SUBMISSION_DIR, REPORTS_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# columns
ID_COL = "customer_ID"
DATE_COL = "S_2"
TARGET_COL = "target"

# the 11 categorical features from the competition data description
CATEGORICAL_FEATURES = [
    "B_30", "B_38", "D_114", "D_116", "D_117", "D_120",
    "D_126", "D_63", "D_64", "D_66", "D_68",
]

# columns that aren't features and must not go to the model
NON_FEATURE_COLS = [ID_COL, DATE_COL]

# seeds and cross validation
SEED = 42
N_FOLDS = 5

# rows per chunk when streaming the big csvs
CSV_CHUNK_SIZE = 500_000
