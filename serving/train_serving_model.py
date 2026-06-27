"""Train the final production model served by the API.

Cross-validation (in src/) was for model *selection*; production uses a single
model trained on all data. We:
  1. hold out 10% to get an early-stopping iteration and fit an isotonic
     calibrator (so the API returns true probabilities of default),
  2. retrain on 100% of the data at that iteration,
  3. save the booster + calibrator + metadata to serving/artifacts/.

The served model is the calibrated LightGBM — single, fast, and explainable
(the API attaches SHAP reason codes) — the honest production choice over the
research blend.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402
from metric import amex_metric_np, lgb_amex_metric  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "learning_rate": 0.03,
    "num_leaves": 128, "min_child_samples": 40, "feature_fraction": 0.4,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 2.0,
    "n_jobs": -1, "seed": config.SEED, "verbosity": -1,
}


def main() -> None:
    t0 = time.time()
    feats = pd.read_parquet(config.TRAIN_FEATURES)
    labels = pd.read_csv(config.TRAIN_LABELS_CSV)
    df = feats.merge(labels, on=config.ID_COL, how="inner")
    cat_features = [c for c in (config.PROCESSED_DIR / "categorical_features.txt")
                    .read_text().split() if c in df.columns]
    feat_cols = [c for c in df.columns if c not in (config.ID_COL, config.TARGET_COL)]
    X, y = df[feat_cols], df[config.TARGET_COL].values
    print(f"final model on {len(df):,} x {len(feat_cols)} feats")

    # --- 1. holdout for early stopping + calibration -------------------------
    Xtr, Xca, ytr, yca = train_test_split(X, y, test_size=0.1, stratify=y,
                                          random_state=config.SEED)
    model = lgb.train(
        PARAMS, lgb.Dataset(Xtr, ytr, categorical_feature=cat_features),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(Xca, yca, categorical_feature=cat_features)],
        feval=lgb_amex_metric,
        callbacks=[lgb.early_stopping(100, verbose=False)])
    best_iter = model.best_iteration
    p_ca = model.predict(Xca)
    holdout_amex = amex_metric_np(yca, p_ca)
    print(f"holdout amex={holdout_amex:.5f} best_iter={best_iter} "
          f"({time.time()-t0:.0f}s)")

    # --- 2. isotonic calibrator on the holdout -------------------------------
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_ca, yca)

    # --- 3. retrain on ALL data at best_iter ---------------------------------
    final = lgb.train(PARAMS, lgb.Dataset(X, y, categorical_feature=cat_features),
                      num_boost_round=best_iter)
    final.save_model(str(ART / "model.txt"))
    (ART / "calibrator.json").write_text(json.dumps({
        "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}))
    (ART / "model_metadata.json").write_text(json.dumps({
        "model": "lightgbm", "n_features": len(feat_cols),
        "best_iteration": int(best_iter), "holdout_amex": float(holdout_amex),
        "categorical_features": cat_features, "default_rate": float(y.mean()),
        "trained_rows": int(len(df))}, indent=2))
    print(f"saved model.txt + calibrator.json + metadata ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
