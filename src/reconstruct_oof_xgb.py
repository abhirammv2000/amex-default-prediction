"""Rebuild data/processed/oof_xgb.parquet from the saved XGB fold models.

The fold models are truncated to their best iteration and the StratifiedKFold
split is deterministic (fixed seed), so predicting each fold's validation rows
with its model reproduces the original out-of-fold predictions exactly. Used to
recover the OOF when only the models (not the OOF parquet) were persisted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

import config
from metric import amex_metric_np
from train_baseline import load_training_data


def main() -> None:
    df, feature_cols, _ = load_training_data()
    X = df[feature_cols].astype(np.float32)
    y = df[config.TARGET_COL].values

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True,
                          random_state=config.SEED)
    oof = np.zeros(len(df))
    for fold, (_, va) in enumerate(skf.split(X, y), 1):
        booster = xgb.Booster()
        booster.load_model(str(config.MODEL_DIR / f"xgb_fold{fold}.json"))
        oof[va] = booster.predict(xgb.DMatrix(X.iloc[va]))
        print(f"  fold {fold}: amex={amex_metric_np(y[va], oof[va]):.5f}", flush=True)

    out = config.PROCESSED_DIR / "oof_xgb.parquet"
    pd.DataFrame({config.ID_COL: df[config.ID_COL], "target": y, "oof_pred": oof}) \
        .to_parquet(out, index=False)
    print(f"OOF amex = {amex_metric_np(y, oof):.5f} -> wrote {out}")


if __name__ == "__main__":
    main()
