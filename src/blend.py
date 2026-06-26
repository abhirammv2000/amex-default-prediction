"""Blend the tuned LightGBM and XGBoost models.

Finds the blend weight ``w`` that maximises the Amex metric on the aligned
out-of-fold (OOF) predictions, then applies the same ``w`` to the two models'
test predictions to produce the final submission. Because both models use the
same fold split, their OOF rows align by ``customer_ID``.

    final = w * lgb + (1 - w) * xgb

Usage:
    python blend.py
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

import config
from metric import amex_metric_np


def _load_oof(path):
    df = pd.read_parquet(path).set_index(config.ID_COL).sort_index()
    return df


def main() -> None:
    t0 = time.time()
    # ---- align OOF predictions ---------------------------------------------
    lgb_oof = _load_oof(config.PROCESSED_DIR / "oof_predictions.parquet")
    xgb_oof = _load_oof(config.PROCESSED_DIR / "oof_xgb.parquet")
    assert (lgb_oof.index == xgb_oof.index).all(), "OOF customer_IDs misaligned"
    y = lgb_oof["target"].values
    p_lgb, p_xgb = lgb_oof["oof_pred"].values, xgb_oof["oof_pred"].values

    s_lgb = amex_metric_np(y, p_lgb)
    s_xgb = amex_metric_np(y, p_xgb)

    # ---- search the blend weight on OOF ------------------------------------
    best_w, best_s = 1.0, s_lgb
    for w in np.linspace(0, 1, 51):
        s = amex_metric_np(y, w * p_lgb + (1 - w) * p_xgb)
        if s > best_s:
            best_w, best_s = w, s
    print(f"OOF  LGB={s_lgb:.5f}  XGB={s_xgb:.5f}  "
          f"BLEND(w={best_w:.2f})={best_s:.5f}")

    # ---- predict test with both model sets and blend -----------------------
    test = pd.read_parquet(config.TEST_FEATURES)
    ids = test[config.ID_COL]
    feats = [c for c in test.columns if c != config.ID_COL]
    X = test[feats]
    print(f"Scoring {len(X):,} test customers ... ({time.time() - t0:.0f}s)")

    lgb_models = sorted(config.MODEL_DIR.glob("lgbm_fold*.txt"))
    p_test_lgb = np.zeros(len(X))
    for m in lgb_models:
        p_test_lgb += lgb.Booster(model_file=str(m)).predict(X) / len(lgb_models)

    # Predict XGB in row-chunks: a single DMatrix over all 924K x 1628 float32
    # rows exhausts 16 GB RAM, so build it ~150K rows at a time from numpy.
    xgb_models = sorted(config.MODEL_DIR.glob("xgb_fold*.json"))
    boosters = []
    for m in xgb_models:
        b = xgb.Booster(); b.load_model(str(m)); boosters.append(b)
    p_test_xgb = np.zeros(len(X))
    chunk = 150_000
    for start in range(0, len(X), chunk):
        block = X.iloc[start:start + chunk].to_numpy(dtype=np.float32)
        # Models were trained from a named DataFrame, so the DMatrix needs the
        # same feature_names (numpy arrays carry none).
        dblock = xgb.DMatrix(block, feature_names=feats)
        for b in boosters:
            p_test_xgb[start:start + len(block)] += b.predict(dblock) / len(boosters)
        del block, dblock

    p_final = best_w * p_test_lgb + (1 - best_w) * p_test_xgb
    out = config.SUBMISSION_DIR / "submission_blend.csv"
    pd.DataFrame({config.ID_COL: ids, "prediction": p_final}).to_csv(out, index=False)
    print(f"Wrote {out} ({len(ids):,} rows) | blend w={best_w:.2f} "
          f"OOF={best_s:.5f} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
