"""Turn the statement tables into one row of features per customer.

A customer has up to 13 monthly statements and the GBDT needs one row, so each series is summarised: mean, std,
min, max and last for the 177 numeric columns, last, nunique and count for the 11 categorical ones, plus the
number of statements and the span of the history. To fit in 16 GB the columns are read from Parquet in small
batches (--col-batch at a time) instead of loading all 5.5M rows, and customer_ID is turned into integer codes
once and reused. Writes train_features.parquet and test_features.parquet.
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config

NUM_AGGS = ["mean", "std", "min", "max", "first", "last"]
CAT_AGGS = ["last", "nunique", "count"]


def _add_numeric_diffs(agg: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Trend features from the aggregates we already have, since how an account is changing says more than its level:
    last_mean_diff (latest vs the customer's usual), last_first_diff (net movement over the history) and range
    (max minus min)."""
    new = {}
    for c in cols:
        last, mean = agg[f"{c}_last"], agg[f"{c}_mean"]
        new[f"{c}_last_mean_diff"] = (last - mean).astype(np.float32)
        new[f"{c}_last_first_diff"] = (last - agg[f"{c}_first"]).astype(np.float32)
        new[f"{c}_range"] = (agg[f"{c}_max"] - agg[f"{c}_min"]).astype(np.float32)
    return pd.concat([agg, pd.DataFrame(new, index=agg.index)], axis=1)


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _flatten(agg: pd.DataFrame) -> pd.DataFrame:
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg


def build_features(parquet_path, out_path, col_batch: int, is_train: bool) -> None:
    pf = pq.ParquetFile(parquet_path)
    all_cols = [c for c in pf.schema.names]
    feature_cols = [c for c in all_cols if c not in config.NON_FEATURE_COLS]
    cat_cols = [c for c in config.CATEGORICAL_FEATURES if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    print(f"\n=== features: {parquet_path.name} ===")
    print(f"{len(num_cols)} numeric | {len(cat_cols)} categorical")

    t0 = time.time()
    # customer_ID to integer codes, once (the rows are already grouped by customer)
    cid = pq.read_table(parquet_path, columns=[config.ID_COL]).column(0).to_pandas()
    codes, uniques = pd.factorize(cid)
    codes = codes.astype(np.int32)
    n_cust = len(uniques)
    print(f"{len(cid):,} statements across {n_cust:,} customers "
          f"({time.time() - t0:.0f}s)")
    del cid
    gc.collect()

    parts: list[pd.DataFrame] = []

    # numeric columns, a batch at a time
    for b, cols in enumerate(_batched(num_cols, col_batch), 1):
        tbl = pq.read_table(parquet_path, columns=cols).to_pandas()
        tbl["_cid"] = codes
        agg = _flatten(tbl.groupby("_cid")[cols].agg(NUM_AGGS).astype(np.float32))
        agg = _add_numeric_diffs(agg, cols)
        parts.append(agg)
        print(f"  numeric batch {b}: {len(cols)} cols -> {agg.shape[1]} feats "
              f"({time.time() - t0:.0f}s)", flush=True)
        del tbl, agg
        gc.collect()

    # categorical columns
    tbl = pq.read_table(parquet_path, columns=cat_cols).to_pandas()
    tbl["_cid"] = codes
    cat_agg = _flatten(tbl.groupby("_cid")[cat_cols].agg(CAT_AGGS))
    # the last value of a categorical can be a string or a float code, so label-encode it to int16. The
    # mapping is fit on train and saved, then reused on test. Encoding each split separately gave different
    # codes for the same category, which the PSI check in drift.py caught
    map_path = config.PROCESSED_DIR / "categorical_maps.json"
    if is_train:
        maps = {}
        for c in cat_cols:
            col = f"{c}_last"
            cats = pd.Categorical(cat_agg[col])
            maps[c] = [None if pd.isna(v) else str(v) for v in cats.categories]
            cat_agg[col] = cats.codes.astype(np.int16)
        map_path.write_text(json.dumps(maps))
    else:
        maps = json.loads(map_path.read_text())
        for c in cat_cols:
            col = f"{c}_last"
            lookup = {v: i for i, v in enumerate(maps[c]) if v is not None}
            cat_agg[col] = (cat_agg[col].astype("object")
                            .map(lambda v: lookup.get(None if pd.isna(v) else str(v), -1))
                            .astype(np.int16))
    num_like = [c for c in cat_agg.columns if not c.endswith("_last")]
    cat_agg[num_like] = cat_agg[num_like].astype(np.float32)
    parts.append(cat_agg)
    print(f"  categorical: {len(cat_cols)} cols -> {cat_agg.shape[1]} feats "
          f"({time.time() - t0:.0f}s)")
    del tbl, cat_agg
    gc.collect()

    # statement count and days of history
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

    # build the wide table column by column in Arrow, not pd.concat. On the 924K customer test set pandas
    # doubles its memory while consolidating (about 12 GB) and runs out on 16 GB. Converting each part to Arrow and
    # freeing the frame keeps the peak near 6 GB
    import pyarrow as pa

    n_feats = sum(p.shape[1] for p in parts)
    arrays = {config.ID_COL: pa.array(np.asarray(uniques))}
    for i in range(len(parts)):
        tbl = pa.Table.from_pandas(parts[i], preserve_index=False)
        for name in tbl.column_names:
            arrays[name] = tbl.column(name)
        parts[i] = None          # release the pandas part
        del tbl
        gc.collect()
    del parts
    gc.collect()

    # categorical-'last' column names recorded for the model
    cat_feature_names = [f"{c}_last" for c in cat_cols]
    print(f"Final feature matrix: {n_cust:,} customers x {n_feats} features")

    pq.write_table(pa.table(arrays), out_path)
    del arrays
    gc.collect()
    # save the categorical column list too
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

    # train goes first, it fits the categorical maps that test reuses
    if args.which in ("train", "both"):
        build_features(config.TRAIN_PARQUET, config.TRAIN_FEATURES, args.col_batch,
                       is_train=True)
    if args.which in ("test", "both"):
        build_features(config.TEST_PARQUET, config.TEST_FEATURES, args.col_batch,
                       is_train=False)
