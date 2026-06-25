"""Train the LightGBM baseline with 5-fold stratified cross-validation.

Loads the engineered per-customer feature table, joins the labels, and trains
one LightGBM model per fold using the official competition metric for early
stopping. Saves per-fold models, out-of-fold (OOF) predictions, the CV score,
and a feature-importance table.

Usage
-----
    python train_baseline.py
    python train_baseline.py --num-boost-round 2000 --learning-rate 0.03
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config
from metric import amex_metric_np, lgb_amex_metric


def load_training_data():
    feats = pd.read_parquet(config.TRAIN_FEATURES)
    labels = pd.read_csv(config.TRAIN_LABELS_CSV)
    df = feats.merge(labels, on=config.ID_COL, how="inner")
    cat_features = (config.PROCESSED_DIR / "categorical_features.txt").read_text().split()
    cat_features = [c for c in cat_features if c in df.columns]
    feature_cols = [c for c in df.columns if c not in (config.ID_COL, config.TARGET_COL)]
    return df, feature_cols, cat_features


def main(args) -> None:
    t0 = time.time()
    df, feature_cols, cat_features = load_training_data()
    X = df[feature_cols]
    y = df[config.TARGET_COL].values
    print(f"Train matrix: {X.shape[0]:,} customers x {len(feature_cols)} features")
    print(f"Default rate: {y.mean():.4f} | categorical features: {len(cat_features)}")

    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": 40,
        "feature_fraction": 0.4,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "n_jobs": -1,
        "seed": config.SEED,
        "verbosity": -1,
    }

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    oof = np.zeros(len(df))
    importances = np.zeros(len(feature_cols))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        dtrain = lgb.Dataset(X.iloc[tr_idx], y[tr_idx],
                             categorical_feature=cat_features)
        dvalid = lgb.Dataset(X.iloc[va_idx], y[va_idx],
                             categorical_feature=cat_features)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=args.num_boost_round,
            valid_sets=[dvalid],
            feval=lgb_amex_metric,
            callbacks=[
                lgb.early_stopping(args.early_stopping, verbose=False),
                lgb.log_evaluation(args.log_every),
            ],
        )
        oof[va_idx] = model.predict(X.iloc[va_idx])
        score = amex_metric_np(y[va_idx], oof[va_idx])
        fold_scores.append(score)
        importances += model.feature_importance(importance_type="gain") / config.N_FOLDS
        model.save_model(str(config.MODEL_DIR / f"lgbm_fold{fold}.txt"))
        print(f"[fold {fold}] amex={score:.5f} | best_iter={model.best_iteration} "
              f"| {time.time() - t0:.0f}s")

    cv_score = amex_metric_np(y, oof)
    print("\n================ CV RESULTS ================")
    print(f"Per-fold amex : {[round(s, 5) for s in fold_scores]}")
    print(f"Mean +/- std  : {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"OOF amex      : {cv_score:.5f}")
    print("============================================")

    # persist OOF predictions
    pd.DataFrame({config.ID_COL: df[config.ID_COL], "target": y, "oof_pred": oof}) \
        .to_parquet(config.PROCESSED_DIR / "oof_predictions.parquet", index=False)

    # feature importance
    imp = pd.DataFrame({"feature": feature_cols, "gain": importances}) \
        .sort_values("gain", ascending=False)
    imp.to_csv(config.OUTPUT_DIR / "feature_importance.csv", index=False)
    print("\nTop 20 features by gain:")
    print(imp.head(20).to_string(index=False))

    # metadata
    meta = {
        "cv_oof_amex": float(cv_score),
        "fold_scores": [float(s) for s in fold_scores],
        "mean_fold_amex": float(np.mean(fold_scores)),
        "std_fold_amex": float(np.std(fold_scores)),
        "n_features": len(feature_cols),
        "n_customers": int(len(df)),
        "params": params,
        "num_boost_round": args.num_boost_round,
    }
    (config.MODEL_DIR / "cv_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved models, OOF, importance, metadata. Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-boost-round", type=int, default=1500)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--num-leaves", type=int, default=128)
    ap.add_argument("--early-stopping", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=200)
    main(ap.parse_args())
