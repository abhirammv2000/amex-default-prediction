"""Hyperparameter tuning for the LightGBM model with Optuna.

Two stages:
  1. SEARCH  - a fast single-fold objective (train on 4 folds, validate on 1)
     so each trial is ~2-3 min; Optuna maximises the official Amex metric over
     a sensible LightGBM search space.
  2. RETRAIN - an honest 5-fold StratifiedKFold run with the best params,
     producing the OOF score, per-fold models and cv_metadata (same artifacts
     as train_baseline.py) so the result is directly comparable.

Outputs (under outputs/): best_params.json, models/lgbm_fold*.txt,
models/cv_metadata.json, feature_importance.csv; OOF to data/processed/.

Usage:
    python tune.py --n-trials 50 --timeout 3600
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config
from metric import amex_metric_np, lgb_amex_metric
from train_baseline import load_training_data  # reuse the exact loader


def run_cv(X, y, cat_features, params, num_boost_round, early_stopping):
    """Honest 5-fold CV; returns (oof, fold_scores, importances, best_iters)."""
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True,
                          random_state=config.SEED)
    oof = np.zeros(len(y))
    importances = np.zeros(X.shape[1])
    fold_scores, best_iters = [], []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_features)
        dva = lgb.Dataset(X.iloc[va], y[va], categorical_feature=cat_features)
        model = lgb.train(
            params, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
            feval=lgb_amex_metric,
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )
        oof[va] = model.predict(X.iloc[va])
        fold_scores.append(amex_metric_np(y[va], oof[va]))
        importances += model.feature_importance("gain") / config.N_FOLDS
        best_iters.append(model.best_iteration)
        model.save_model(str(config.MODEL_DIR / f"lgbm_fold{fold}.txt"))
        print(f"  [fold {fold}] amex={fold_scores[-1]:.5f} "
              f"best_iter={model.best_iteration}", flush=True)
    return oof, fold_scores, importances, best_iters


def main(args) -> None:
    t0 = time.time()
    df, feature_cols, cat_features = load_training_data()
    X = df[feature_cols]
    y = df[config.TARGET_COL].values
    print(f"Tuning on {X.shape[0]:,} x {len(feature_cols)} | "
          f"{len(cat_features)} categorical")

    # ---- single fixed split for the fast search objective -------------------
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True,
                          random_state=config.SEED)
    tr_idx, va_idx = next(iter(skf.split(X, y)))
    dtrain = lgb.Dataset(X.iloc[tr_idx], y[tr_idx], categorical_feature=cat_features)
    dvalid = lgb.Dataset(X.iloc[va_idx], y[va_idx], categorical_feature=cat_features)
    y_va = y[va_idx]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary", "verbosity": -1, "n_jobs": -1,
            "seed": config.SEED, "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 48, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 150),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.6),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "min_sum_hessian_in_leaf": trial.suggest_float(
                "min_sum_hessian_in_leaf", 1e-3, 10.0, log=True),
        }
        model = lgb.train(
            params, dtrain, num_boost_round=3000, valid_sets=[dvalid],
            feval=lgb_amex_metric,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        score = amex_metric_np(y_va, model.predict(X.iloc[va_idx]))
        trial.set_user_attr("best_iteration", model.best_iteration)
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   show_progress_bar=False)
    print(f"\nSearch done: {len(study.trials)} trials | best single-fold amex="
          f"{study.best_value:.5f} | {time.time() - t0:.0f}s")
    print("Best params:", json.dumps(study.best_params, indent=2))

    # ---- honest 5-fold retrain with the best params -------------------------
    best = {
        "objective": "binary", "verbosity": -1, "n_jobs": -1,
        "seed": config.SEED, "boosting_type": "gbdt", "bagging_freq": 1,
        **study.best_params,
    }
    (config.OUTPUT_DIR / "best_params.json").write_text(json.dumps(best, indent=2))

    print("\nRetraining 5-fold with best params ...")
    oof, fold_scores, importances, best_iters = run_cv(
        X, y, cat_features, best, num_boost_round=args.final_rounds,
        early_stopping=args.early_stopping)
    cv = amex_metric_np(y, oof)
    print("\n================ TUNED CV ================")
    print(f"Per-fold : {[round(s, 5) for s in fold_scores]}")
    print(f"OOF amex : {cv:.5f}  (baseline v2 = 0.79266)")
    print("=========================================")

    pd.DataFrame({config.ID_COL: df[config.ID_COL], "target": y, "oof_pred": oof}) \
        .to_parquet(config.PROCESSED_DIR / "oof_predictions.parquet", index=False)
    pd.DataFrame({"feature": feature_cols, "gain": importances}) \
        .sort_values("gain", ascending=False) \
        .to_csv(config.OUTPUT_DIR / "feature_importance.csv", index=False)
    meta = {
        "cv_oof_amex": float(cv),
        "fold_scores": [float(s) for s in fold_scores],
        "mean_fold_amex": float(np.mean(fold_scores)),
        "std_fold_amex": float(np.std(fold_scores)),
        "best_iters": best_iters,
        "n_features": len(feature_cols),
        "params": best,
        "n_trials": len(study.trials),
        "best_search_amex": float(study.best_value),
    }
    (config.MODEL_DIR / "cv_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved best_params, models, cv_metadata. Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--final-rounds", type=int, default=5000)
    ap.add_argument("--early-stopping", type=int, default=200)
    main(ap.parse_args())
