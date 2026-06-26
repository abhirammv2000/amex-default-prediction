"""Model explainability with SHAP — global drivers + per-customer reason codes.

Regulated lending (e.g. ECOA) requires an *adverse-action* explanation: the
specific reasons a customer was scored as high risk. Tree SHAP gives exact,
additive per-prediction attributions that support both a global view (which
features drive the model) and local "reason codes" for individual decisions.

Runs on a sample of customers (Tree SHAP on all 458K x 1,628 is unnecessary).
Outputs: reports/figures/shap_summary.png, shap_importance.png and printed
example reason codes.

Usage:
    python explain.py --sample 20000
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
import shap

import config
from train_baseline import load_training_data


def main(args) -> None:
    df, feature_cols, _ = load_training_data()
    model = lgb.Booster(model_file=str(config.MODEL_DIR / "lgbm_fold1.txt"))

    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(df), size=min(args.sample, len(df)), replace=False)
    X = df.iloc[idx][feature_cols]
    y = df.iloc[idx][config.TARGET_COL].values

    print(f"Computing Tree SHAP on {len(X):,} customers x {len(feature_cols)} feats ...")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):          # older shap returns [class0, class1]
        sv = sv[1]

    # ---- global: mean |SHAP| importance ------------------------------------
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(mean_abs)[::-1]
    print("\nTop 15 global drivers (mean |SHAP|):")
    for i in order[:15]:
        print(f"  {feature_cols[i]:24s} {mean_abs[i]:.4f}")

    plt.figure()
    shap.summary_plot(sv, X, max_display=20, show=False)
    plt.title("SHAP summary — top 20 features")
    plt.savefig(config.FIGURE_DIR / "shap_summary.png", bbox_inches="tight", dpi=110)
    plt.close()

    plt.figure()
    shap.summary_plot(sv, X, plot_type="bar", max_display=20, show=False)
    plt.title("SHAP global importance")
    plt.savefig(config.FIGURE_DIR / "shap_importance.png", bbox_inches="tight", dpi=110)
    plt.close()

    # ---- local: reason codes for example high-risk customers ---------------
    base = explainer.expected_value
    base = base[1] if isinstance(base, (list, np.ndarray)) and np.ndim(base) else base
    pred_logit = base + sv.sum(1)
    riskiest = np.argsort(pred_logit)[::-1][:3]
    print("\n--- Example adverse-action reason codes (top risk drivers) ---")
    for r in riskiest:
        contrib = sv[r]
        top = np.argsort(contrib)[::-1][:5]   # features pushing toward default
        print(f"\nCustomer {df.iloc[idx[r]][config.ID_COL][:12]}...  "
              f"actual default={y[r]}")
        for i in top:
            print(f"   +{contrib[i]:.3f}  {feature_cols[i]} = {X.iloc[r, i]:.3f}")

    print("\nSaved: shap_summary.png, shap_importance.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000)
    main(ap.parse_args())
