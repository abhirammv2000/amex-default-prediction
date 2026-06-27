"""Build per-customer sequence tensors for the GRU (no aggregation).

The GBDT path collapses each customer's statements into summary stats; the
sequence model instead keeps the raw monthly series. Each customer becomes a
left-aligned, zero-padded ``[13, F]`` tensor (oldest statement first) plus its
true length, so a packed GRU can ignore the padding.

Numeric features are standardised with statistics **fit on train** (saved and
reused for test); the two string categoricals (D_63, D_64) are integer-encoded
with a train-fit map. Output: an ``.npz`` with sequences (float16), lengths,
labels (train only) and customer_ids.

Usage:
    python build_sequences.py --which train
    python build_sequences.py --which test
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config

MAXLEN = 13
STR_CATS = ["D_63", "D_64"]
STATS_PATH = config.PROCESSED_DIR / "seq_norm_stats.json"


def _feature_cols(parquet_path):
    names = pq.read_schema(parquet_path).names
    return [c for c in names if c not in config.NON_FEATURE_COLS]


def build(which: str) -> None:
    t0 = time.time()
    is_train = which == "train"
    pq_path = config.TRAIN_PARQUET if is_train else config.TEST_PARQUET
    feat_cols = _feature_cols(pq_path)
    print(f"[{which}] {len(feat_cols)} features from {pq_path.name}")

    # --- customer codes (1 column; statements already in customer/date order) -
    cid = pq.read_table(pq_path, columns=[config.ID_COL]).column(0).to_pandas()
    codes, uniques = pd.factorize(cid)                    # contiguous blocks
    codes = codes.astype(np.int64)
    n_cust, n_rows = len(uniques), len(cid)
    print(f"[{which}] {n_rows:,} statements -> {n_cust:,} customers "
          f"({time.time()-t0:.0f}s)")
    del cid

    # --- categorical maps: fit on train (read only the 2 string cols), else load
    if is_train:
        cat_maps = {}
        for c in STR_CATS:
            col = pq.read_table(pq_path, columns=[c]).column(0).to_pandas()
            cats = pd.Categorical(col)
            cat_maps[c] = [None if pd.isna(v) else str(v) for v in cats.categories]
        (config.PROCESSED_DIR / "seq_cat_maps.json").write_text(json.dumps(cat_maps))
    else:
        cat_maps = json.loads((config.PROCESSED_DIR / "seq_cat_maps.json").read_text())
    cat_lut = {c: {v: i for i, v in enumerate(cat_maps[c]) if v is not None}
               for c in STR_CATS}

    def _encode_batch(bdf):
        """In-place encode the string categoricals of a batch DataFrame."""
        for c in STR_CATS:
            lut = cat_lut[c]
            bdf[c] = (bdf[c].astype("object")
                      .map(lambda v: lut.get(None if pd.isna(v) else str(v), -1))
                      .astype(np.float32))
        return bdf

    F = len(feat_cols)
    # --- standardisation stats: compute streaming on train, else load ---------
    if is_train:
        cnt = np.zeros(F); s1 = np.zeros(F); s2 = np.zeros(F)
        for batch in pq.ParquetFile(pq_path).iter_batches(batch_size=1_000_000,
                                                          columns=feat_cols):
            arr = _encode_batch(batch.to_pandas())[feat_cols].to_numpy(np.float32)
            m = ~np.isnan(arr)
            cnt += m.sum(0); s1 += np.nansum(arr, 0); s2 += np.nansum(arr * arr, 0)
        mean = s1 / cnt
        std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 1e-12))
        std[std == 0] = 1.0
        STATS_PATH.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist(),
                                          "features": feat_cols}))
    else:
        st = json.loads(STATS_PATH.read_text())
        mean, std = np.array(st["mean"]), np.array(st["std"])

    # --- fill the standardised float16 feature matrix in row batches ----------
    feats = np.empty((n_rows, F), dtype=np.float16)
    off = 0
    for batch in pq.ParquetFile(pq_path).iter_batches(batch_size=1_000_000,
                                                      columns=feat_cols):
        arr = _encode_batch(batch.to_pandas())[feat_cols].to_numpy(np.float32)
        arr = (arr - mean) / std
        np.nan_to_num(arr, copy=False, nan=0.0)
        np.clip(arr, -10, 10, out=arr)
        feats[off:off + len(arr)] = arr.astype(np.float16)
        off += len(arr)

    # --- vectorised scatter into [N, 13, F] (keep the last 13 statements) -----
    counts = np.bincount(codes, minlength=n_cust)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    within = np.arange(len(codes)) - starts            # 0-based pos within customer
    drop = np.repeat(np.maximum(counts - MAXLEN, 0), counts)
    pos = within - drop                                # left-aligned kept position
    valid = pos >= 0
    seq = np.zeros((n_cust, MAXLEN, F), dtype=np.float16)
    seq[codes[valid], pos[valid], :] = feats[valid]
    lengths = np.minimum(counts, MAXLEN).astype(np.int16)
    print(f"[{which}] tensor {seq.shape} ({seq.nbytes/1e9:.1f} GB) "
          f"({time.time()-t0:.0f}s)")

    out = config.PROCESSED_DIR / f"seq_{which}.npz"
    payload = dict(seq=seq, lengths=lengths,
                   customer_ids=uniques.astype(str))
    if is_train:
        labels = pd.read_csv(config.TRAIN_LABELS_CSV).set_index(config.ID_COL)
        payload["labels"] = labels.loc[uniques, "target"].to_numpy(np.int8)
    np.savez(out, **payload)
    print(f"[{which}] wrote {out} ({out.stat().st_size/1e9:.2f} GB) "
          f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["train", "test"], default="train")
    build(ap.parse_args().which)
