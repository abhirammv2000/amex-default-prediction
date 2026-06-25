"""Per-customer feature engineering from the Parquet statement tables.

Each customer has up to 13 monthly statements (a short multivariate time series).
A GBDT needs one row per customer, so we collapse the series into summary
statistics:

  * numeric features (177)  -> mean, std, min, max, last
  * categorical features (11) -> last, nunique, count
  * plus the statement count and the time span of the customer's history

To stay within 16 GB of RAM the columns are read from Parquet in small batches
(``--col-batch`` columns at a time) rather than loading the whole 5.5M-row table
at once. ``customer_ID`` is factorised to integer codes a single time and reused
for every batch, so the expensive string column is only read once.

Output: ``train_features.parquet`` / ``test_features.parquet`` — one row per
customer, ready for modelling.
"""
from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config

NUM_AGGS = ["mean", "std", "min", "max", "last"]
CAT_AGGS = ["last", "nunique", "count"]


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _flatten(agg: pd.DataFrame) -> pd.DataFrame:
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg


def build_features(parquet_path, out_path, col_batch: int) -> None:
    pf = pq.ParquetFile(parquet_path)
    all_cols = [c for c in pf.schema.names]
    feature_cols = [c for c in all_cols if c not in config.NON_FEATURE_COLS]
    cat_cols = [c for c in config.CATEGORICAL_FEATURES if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    print(f"\n=== features: {parquet_path.name} ===")
    print(f"{len(num_cols)} numeric | {len(cat_cols)} categorical")

    t0 = time.time()
    # --- factorise customer_ID once (data is grouped by customer already) ----
    cid = pq.read_table(parquet_path, columns=[config.ID_COL]).column(0).to_pandas()
    codes, uniques = pd.factorize(cid)
    codes = codes.astype(np.int32)
    n_cust = len(uniques)
    print(f"{len(cid):,} statements across {n_cust:,} customers "
          f"({time.time() - t0:.0f}s)")
    del cid
    gc.collect()

    parts: list[pd.DataFrame] = []

    # --- numeric aggregations in column batches ------------------------------
    for b, cols in enumerate(_batched(num_cols, col_batch), 1):
        tbl = pq.read_table(parquet_path, columns=cols).to_pandas()
        tbl["_cid"] = codes
        agg = _flatten(tbl.groupby("_cid")[cols].agg(NUM_AGGS).astype(np.float32))
        parts.append(agg)
        print(f"  numeric batch {b}: {len(cols)} cols -> {agg.shape[1]} feats "
              f"({time.time() - t0:.0f}s)", flush=True)
        del tbl, agg
        gc.collect()

    # --- categorical aggregations -------------------------------------------
    tbl = pq.read_table(parquet_path, columns=cat_cols).to_pandas()
    tbl["_cid"] = codes
    cat_agg = _flatten(tbl.groupby("_cid")[cat_cols].agg(CAT_AGGS))
    # 'last' of categoricals may be string/float codes -> label-encode to int16
    for c in cat_cols:
        col = f"{c}_last"
        cat_agg[col] = cat_agg[col].astype("category").cat.codes.astype(np.int16)
    num_like = [c for c in cat_agg.columns if not c.endswith("_last")]
    cat_agg[num_like] = cat_agg[num_like].astype(np.float32)
    parts.append(cat_agg)
    print(f"  categorical: {len(cat_cols)} cols -> {cat_agg.shape[1]} feats "
          f"({time.time() - t0:.0f}s)")
    del tbl, cat_agg
    gc.collect()

    # --- date-derived: statement count + history span in days ----------------
    s2 = pq.read_table(parquet_path, columns=[config.DATE_COL]).column(0).to_pandas()
    date_df = pd.DataFrame({"_cid": codes, config.DATE_COL: s2})
    span = date_df.groupby("_cid")[config.DATE_COL].agg(["count", "min", "max"])
    span["history_days"] = (span["max"] - span["min"]).dt.days.astype(np.float32)
    span = span.rename(columns={"count": "statement_count"})[
        ["statement_count", "history_days"]
    ].astype(np.float32)
    parts.append(span)
    del s2, date_df
    gc.collect()

    # --- assemble ------------------------------------------------------------
    features = pd.concat(parts, axis=1)
    features.insert(0, config.ID_COL, uniques)
    features = features.reset_index(drop=True)

    # categorical-'last' column names recorded for the model
    cat_feature_names = [f"{c}_last" for c in cat_cols]
    print(f"Final feature matrix: {features.shape[0]:,} customers x "
          f"{features.shape[1] - 1} features")

    features.to_parquet(out_path, index=False)
    # Persist the categorical column list alongside (same names both splits).
    (config.PROCESSED_DIR / "categorical_features.txt").write_text(
        "\n".join(cat_feature_names)
    )
    size_mb = out_path.stat().st_size / 1024**2
    print(f"DONE: {out_path.name} ({size_mb:,.0f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["train", "test", "both"], default="both")
    ap.add_argument("--col-batch", type=int, default=40)
    args = ap.parse_args()

    if args.which in ("train", "both"):
        build_features(config.TRAIN_PARQUET, config.TRAIN_FEATURES, args.col_batch)
    if args.which in ("test", "both"):
        build_features(config.TEST_PARQUET, config.TEST_FEATURES, args.col_batch)
