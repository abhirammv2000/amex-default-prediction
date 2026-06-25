"""Train an XGBoost model with 5-fold CV — a second model family for blending.

Uses the *same* StratifiedKFold split (same seed) as the LightGBM models, so the
out-of-fold (OOF) predictions align row-for-row and can be blended directly
(see blend.py). Early stopping uses the official Amex metric.

Outputs: outputs/models/xgb_fold*.json, outputs/models/cv_metadata_xgb.json,
and data/processed/oof_xgb.parquet.

Usage:
    python train_xgb.py
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

import config
from metric import amex_metric_np
from train_baseline import load_training_data


def xgb_amex(y_pred, dmatrix):
    """XGBoost custom metric -> (name, value); maximised."""
    return "amex", amex_metric_np(dmatrix.get_label(), y_pred)


def main(args) -> None:
    t0 = time.time()
    df, feature_cols, cat_features = load_training_data()
    # XGBoost treats the label-encoded categorical 'last' codes as numeric
    # ordinals (already int); fine for a blend partner.
    X = df[feature_cols].astype(np.float32)
    y = df[config.TARGET_COL].values
    print(f"XGB train: {X.shape[0]:,} x {len(feature_cols)} | default {y.mean():.4f}")

    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": 0.8,
        "colsample_bytree": 0.4,
        "min_child_weight": 8,
        "reg_lambda": 2.0,
        "seed": config.SEED,
        "nthread": -1,
    }

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True,
                          random_state=config.SEED)
    oof = np.zeros(len(df))
    fold_scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        dtr = xgb.DMatrix(X.iloc[tr], label=y[tr])
        dva = xgb.DMatrix(X.iloc[va], label=y[va])
        model = xgb.train(
            params, dtr, num_boost_round=args.num_boost_round,
            evals=[(dva, "valid")], custom_metric=xgb_amex, maximize=True,
            early_stopping_rounds=args.early_stopping, verbose_eval=args.log_every,
        )
        oof[va] = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
        fold_scores.append(amex_metric_np(y[va], oof[va]))
        # Persist the model truncated to its best iteration so later test
        # predictions (which use all stored trees) match the OOF above.
        model[: model.best_iteration + 1].save_model(
            str(config.MODEL_DIR / f"xgb_fold{fold}.json"))
        print(f"[fold {fold}] amex={fold_scores[-1]:.5f} "
              f"best_iter={model.best_iteration} | {time.time() - t0:.0f}s", flush=True)

    cv = amex_metric_np(y, oof)
    print("\n================ XGB CV ================")
    print(f"Per-fold : {[round(s, 5) for s in fold_scores]}")
    print(f"OOF amex : {cv:.5f}")
    print("=======================================")

    pd.DataFrame({config.ID_COL: df[config.ID_COL], "target": y, "oof_pred": oof}) \
        .to_parquet(config.PROCESSED_DIR / "oof_xgb.parquet", index=False)
    meta = {
        "model": "xgboost",
        "cv_oof_amex": float(cv),
        "fold_scores": [float(s) for s in fold_scores],
        "mean_fold_amex": float(np.mean(fold_scores)),
        "std_fold_amex": float(np.std(fold_scores)),
        "n_features": len(feature_cols),
        "params": params,
    }
    (config.MODEL_DIR / "cv_metadata_xgb.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved XGB models, OOF, metadata. Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-boost-round", type=int, default=3000)
    ap.add_argument("--eta", type=float, default=0.03)
    ap.add_argument("--max-depth", type=int, default=7)
    ap.add_argument("--early-stopping", type=int, default=150)
    ap.add_argument("--log-every", type=int, default=300)
    main(ap.parse_args())
