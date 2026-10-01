"""Drift between train and test, measured with PSI.

Every training customer is observed at the same time (March 2018), so there is no out-of-time split inside the
training data. The test set comes from a later period, so train against test is an out-of-time comparison of the
inputs, which is what a deployed model has to watch. PSI is computed per feature and for the score:

    PSI = sum over bins of (test% - train%) * ln(test% / train%)

Under 0.10 is stable, 0.10 to 0.25 a moderate shift, over 0.25 a significant one. Writes reports/figures/score_drift.png
and reports/drift_psi.csv.

    python drift.py --sample 80000
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config


def psi(expected, actual, bins=10) -> float:
    """PSI of actual against an expected reference, with bins taken from the expected quantiles."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    eps = 1e-6
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _sample(path, n, cols=None, seed=config.SEED):
    df = pd.read_parquet(path, columns=cols)
    if len(df) > n:
        df = df.sample(n, random_state=seed)
    return df


def main(args) -> None:
    feats = [c for c in pq.read_schema(config.TRAIN_FEATURES).names
             if c != config.ID_COL]  # cheap: schema only, no data read
    print(f"Sampling {args.sample:,} train / test rows over {len(feats)} features ...")
    tr = _sample(config.TRAIN_FEATURES, args.sample)
    te = _sample(config.TEST_FEATURES, args.sample)

    # feature PSI
    rows = []
    for c in feats:
        rows.append((c, psi(tr[c].to_numpy(float), te[c].to_numpy(float))))
    psi_df = pd.DataFrame(rows, columns=["feature", "psi"]).dropna() \
        .sort_values("psi", ascending=False)
    psi_df.to_csv(config.REPORTS_DIR / "drift_psi.csv", index=False)

    stable = (psi_df["psi"] < 0.10).mean()
    moderate = ((psi_df["psi"] >= 0.10) & (psi_df["psi"] < 0.25)).mean()
    shifted = (psi_df["psi"] >= 0.25).mean()
    print("\n================ FEATURE DRIFT (train -> test) ================")
    print(f"features stable  (PSI<0.10): {stable:5.1%}")
    print(f"features moderate(0.10-0.25): {moderate:5.1%}")
    print(f"features shifted (PSI>0.25): {shifted:5.1%}")
    print("\nMost-drifted features:")
    print(psi_df.head(10).to_string(index=False))

    # score PSI: train OOF preds vs test preds
    train_scores = pd.read_parquet(config.PROCESSED_DIR / "oof_predictions.parquet")["oof_pred"].to_numpy()
    sub = config.SUBMISSION_DIR / "submission_lgbm_baseline.csv"
    score_psi = np.nan
    if sub.exists():
        test_scores = pd.read_csv(sub)["prediction"].to_numpy()
        score_psi = psi(train_scores, test_scores)
        print(f"\nModel SCORE PSI (train OOF vs test preds): {score_psi:.4f} "
              f"({'stable' if score_psi < 0.1 else 'moderate' if score_psi < 0.25 else 'SHIFTED'})")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(train_scores, bins=50, density=True, alpha=0.6,
                color="#4C72B0", label="train (OOF)")
        ax.hist(test_scores, bins=50, density=True, alpha=0.6,
                color="#C44E52", label="test")
        ax.set_title(f"Predicted-default-probability distribution (score PSI={score_psi:.3f})")
        ax.set_xlabel("predicted default probability"); ax.legend()
        fig.savefig(config.FIGURE_DIR / "score_drift.png", bbox_inches="tight")
        plt.close(fig)

    verdict = ("stable, the random-CV estimate should transfer to the test period"
               if shifted < 0.05 and (np.isnan(score_psi) or score_psi < 0.1)
               else "some drift, monitor or consider recalibration")
    print(f"\nVerdict: population is {verdict}.")
    print("Saved: drift_psi.csv, score_drift.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=80000)
    main(ap.parse_args())
