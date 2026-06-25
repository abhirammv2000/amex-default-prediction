"""Exploratory data analysis -> saves figures to outputs/figures/.

Runs on a customer-level sample so it is fast and memory-light. The target
distribution uses the full (30 MB) labels file. Figures produced:

  1. target_distribution.png   - class balance
  2. statements_per_customer.png
  3. missingness.png           - share of features by missing-rate bucket
  4. feature_groups.png        - feature counts by prefix (B/D/P/R/S)
  5. p2_by_target.png          - P_2 (a key payment feature) last value vs target
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def main(sample_rows: int = 600_000) -> None:
    fig_dir = config.FIGURE_DIR

    # ---- target distribution (full labels) ---------------------------------
    labels = pd.read_csv(config.TRAIN_LABELS_CSV)
    rate = labels["target"].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = labels["target"].value_counts().sort_index()
    ax.bar(["No default (0)", "Default (1)"], counts.values,
           color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n{v/len(labels):.1%}", ha="center", va="bottom")
    ax.set_title(f"Target distribution  (default rate = {rate:.1%})")
    ax.set_ylabel("customers")
    fig.savefig(fig_dir / "target_distribution.png")
    plt.close(fig)
    print(f"target: {len(labels):,} customers, default rate {rate:.4f}")

    # ---- sample of statements ----------------------------------------------
    print(f"reading first {sample_rows:,} rows for EDA sample ...")
    # read numeric columns as float32 to roughly halve the sample's footprint
    header = pd.read_csv(config.TRAIN_CSV, nrows=0).columns.tolist()
    str_cols = {config.ID_COL, config.DATE_COL, "D_63", "D_64"}
    f32 = {c: "float32" for c in header if c not in str_cols}
    df = pd.read_csv(config.TRAIN_CSV, nrows=sample_rows, dtype=f32)
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    feat_cols = [c for c in df.columns if c not in config.NON_FEATURE_COLS]

    # ---- statements per customer -------------------------------------------
    spc = df.groupby(config.ID_COL).size()
    fig, ax = plt.subplots(figsize=(6, 4))
    spc.value_counts().sort_index().plot(kind="bar", color="#55A868", ax=ax)
    ax.set_title("Statements per customer (sample)")
    ax.set_xlabel("number of monthly statements")
    ax.set_ylabel("customers")
    fig.savefig(fig_dir / "statements_per_customer.png")
    plt.close(fig)

    # ---- missingness --------------------------------------------------------
    miss = df[feat_cols].isna().mean()
    buckets = pd.cut(miss, [-0.01, 0.0, 0.1, 0.3, 0.5, 0.9, 1.01],
                     labels=["0%", "0-10%", "10-30%", "30-50%", "50-90%", ">90%"])
    fig, ax = plt.subplots(figsize=(6, 4))
    buckets.value_counts().sort_index().plot(kind="bar", color="#8172B3", ax=ax)
    ax.set_title("Features by missing-rate bucket")
    ax.set_xlabel("missing rate")
    ax.set_ylabel("number of features")
    fig.savefig(fig_dir / "missingness.png")
    plt.close(fig)

    # ---- feature groups -----------------------------------------------------
    prefixes = pd.Series([c.split("_")[0] for c in feat_cols]).value_counts()
    mapping = {"B": "Balance", "D": "Delinquency", "P": "Payment",
               "R": "Risk", "S": "Spend"}
    fig, ax = plt.subplots(figsize=(6, 4))
    prefixes.rename(index=lambda p: f"{p}_ ({mapping.get(p, p)})").plot(
        kind="bar", color="#CCB974", ax=ax)
    ax.set_title("Feature count by group")
    ax.set_ylabel("number of features")
    fig.savefig(fig_dir / "feature_groups.png")
    plt.close(fig)

    # ---- P_2 (last) vs target ----------------------------------------------
    last = df.sort_values(config.DATE_COL).groupby(config.ID_COL)["P_2"].last()
    merged = last.to_frame("P_2_last").merge(
        labels.set_index(config.ID_COL), left_index=True, right_index=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for t, c in [(0, "#4C72B0"), (1, "#C44E52")]:
        ax.hist(merged.loc[merged["target"] == t, "P_2_last"].dropna(),
                bins=50, alpha=0.6, density=True, color=c,
                label=f"target={t}")
    ax.set_title("Last P_2 value by target (sample)")
    ax.set_xlabel("P_2 (last statement)")
    ax.legend()
    fig.savefig(fig_dir / "p2_by_target.png")
    plt.close(fig)

    print(f"Saved 5 figures to {fig_dir}")


if __name__ == "__main__":
    main()
