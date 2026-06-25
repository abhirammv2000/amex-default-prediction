"""Quick data profiling on a sample of the raw training CSV.

Reads only the first ``--rows`` rows so it runs in seconds and a few hundred MB
of RAM. Used to understand dtypes, missingness, cardinality and the number of
monthly statements per customer before committing to the full pipeline.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config
from metric import amex_metric, amex_metric_np


def profile(rows: int) -> None:
    print(f"Reading first {rows:,} rows of {config.TRAIN_CSV.name} ...")
    df = pd.read_csv(config.TRAIN_CSV, nrows=rows)
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])

    feat_cols = [c for c in df.columns if c not in config.NON_FEATURE_COLS]
    print(f"\nShape: {df.shape}  |  feature columns: {len(feat_cols)}")
    print(f"Unique customers in sample: {df[config.ID_COL].nunique():,}")

    # statements per customer
    spc = df.groupby(config.ID_COL).size()
    print("\nStatements per customer (sample):")
    print(spc.describe().to_string())

    # date range
    print(f"\nDate range: {df[config.DATE_COL].min()}  ->  {df[config.DATE_COL].max()}")

    # dtype breakdown
    print("\nDtype counts:")
    print(df[feat_cols].dtypes.value_counts().to_string())

    # categorical columns present + cardinality
    print("\nDeclared categorical features (cardinality in sample):")
    for c in config.CATEGORICAL_FEATURES:
        if c in df.columns:
            print(f"  {c:7s} dtype={str(df[c].dtype):8s} nunique={df[c].nunique()} "
                  f"missing={df[c].isna().mean():.1%}")

    # missingness summary across all features
    miss = df[feat_cols].isna().mean().sort_values(ascending=False)
    n_high = (miss > 0.5).sum()
    print(f"\nFeatures with >50% missing: {n_high}")
    print("Top 10 most-missing features:")
    print((miss.head(10) * 100).round(1).to_string())

    # column-name prefix groups (D_ delinquency, S_ spend, P_ payment, B_ balance, R_ risk)
    prefixes = {}
    for c in feat_cols:
        p = c.split("_")[0]
        prefixes[p] = prefixes.get(p, 0) + 1
    print("\nFeature groups by prefix:")
    for p, n in sorted(prefixes.items()):
        print(f"  {p}_: {n}")


def test_metric() -> None:
    """Sanity-check the two metric implementations against each other."""
    rng = np.random.default_rng(config.SEED)
    y = rng.integers(0, 2, size=20_000)
    # Continuous, tie-free predictions correlated with the truth (a sigmoid keeps
    # every value unique so both metric variants sort rows identically).
    logits = y * 1.5 + rng.normal(0, 1.0, size=y.shape)
    p = 1.0 / (1.0 + np.exp(-logits))
    m_df = amex_metric(y, p)
    m_np = amex_metric_np(y, p)
    print(f"\n[metric self-test] pandas={m_df:.6f}  numpy={m_np:.6f}  "
          f"diff={abs(m_df - m_np):.2e}")
    assert abs(m_df - m_np) < 1e-6, "metric implementations disagree!"
    # Perfect *ranking* (every positive scored above every negative) -> normalized
    # Gini = 1.0. The full metric is 0.5*(1 + D), and D < 1 whenever defaults make
    # up more than 4% of the population, so the metric itself need not reach 1.0.
    from metric import _normalized_weighted_gini
    p_perfect = y + rng.uniform(0, 0.4, size=y.shape)  # positives strictly above negatives
    g_perfect = _normalized_weighted_gini(
        pd.DataFrame({"target": y, "prediction": p_perfect})
    )
    print(f"[metric self-test] normalized Gini for perfect ranking = {g_perfect:.6f}")
    assert abs(g_perfect - 1.0) < 1e-9
    print("[metric self-test] PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_000)
    args = ap.parse_args()
    test_metric()
    profile(args.rows)
