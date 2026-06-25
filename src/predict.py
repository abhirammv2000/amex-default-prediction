"""Generate the Kaggle submission from the trained fold models.

Loads the engineered test feature table, averages the predictions of the five
fold models, and writes a submission CSV in the required
``customer_ID,prediction`` format.

Usage
-----
    python predict.py
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import config


def main() -> None:
    t0 = time.time()
    test = pd.read_parquet(config.TEST_FEATURES)
    ids = test[config.ID_COL]
    feature_cols = [c for c in test.columns if c != config.ID_COL]
    X = test[feature_cols]
    print(f"Test matrix: {X.shape[0]:,} customers x {len(feature_cols)} features")

    model_paths = sorted(config.MODEL_DIR.glob("lgbm_fold*.txt"))
    if not model_paths:
        raise FileNotFoundError("No fold models found — run train_baseline.py first.")

    preds = np.zeros(len(X))
    for p in model_paths:
        model = lgb.Booster(model_file=str(p))
        preds += model.predict(X) / len(model_paths)
        print(f"  scored with {p.name} ({time.time() - t0:.0f}s)")

    sub = pd.DataFrame({config.ID_COL: ids, "prediction": preds})
    out = config.SUBMISSION_DIR / "submission_lgbm_baseline.csv"
    sub.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(sub):,} rows) in {time.time() - t0:.0f}s")
    print(sub.head().to_string(index=False))
    print(f"\nPrediction stats: min={preds.min():.4f} mean={preds.mean():.4f} "
          f"max={preds.max():.4f}")


if __name__ == "__main__":
    main()
