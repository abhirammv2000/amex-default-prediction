"""Proof of no training/serving skew.

Takes a sample of customers, pulls their *raw* statements, runs the serving
feature pipeline, and asserts the result matches the offline training feature
table (`train_features.parquet`) row-for-row within float tolerance. If this
passes, the online service computes exactly the features the model trained on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "serving"))
import config  # noqa: E402
from app.pipeline import engineer_features  # noqa: E402

ART = ROOT / "serving" / "artifacts"


@pytest.mark.skipif(not config.TRAIN_PARQUET.exists(),
                    reason="raw train parquet not present")
def test_serving_matches_training(n_customers: int = 40):
    feature_order = json.loads((ART / "feature_names.json").read_text())
    cat_maps = json.loads((ART / "categorical_maps.json").read_text())

    # reference rows from the offline feature table
    ref = pd.read_parquet(config.TRAIN_FEATURES)
    sample_ids = ref[config.ID_COL].iloc[:n_customers].tolist()
    ref = ref.set_index(config.ID_COL).loc[sample_ids]

    # pull the raw statements for those customers and run the serving pipeline
    tbl = pq.read_table(config.TRAIN_PARQUET).to_pandas()
    raw = tbl[tbl[config.ID_COL].isin(sample_ids)].copy()
    out = engineer_features(raw, cat_maps, feature_order).loc[sample_ids]

    assert list(out.columns) == feature_order
    a = out.to_numpy(dtype=np.float64)
    b = ref[feature_order].to_numpy(dtype=np.float64)
    # NaNs must align, finite values must match closely (float32 round-trip)
    assert np.array_equal(np.isnan(a), np.isnan(b)), "NaN pattern differs"
    diff = np.nanmax(np.abs(a - b))
    assert diff < 1e-3, f"max feature mismatch {diff} exceeds tolerance"


if __name__ == "__main__":
    test_serving_matches_training()
    print("PASS: serving features match training features (no skew)")
