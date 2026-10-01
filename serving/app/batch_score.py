"""Batch scoring of a whole portfolio.

Behavioural default models are usually scored in batch, since the statements update once a cycle and the decisions
they feed (credit line reviews, pricing, collections, provisioning) are periodic runs. This reads raw statements,
scores every customer with the same model and feature code as the API (app.model and app.pipeline, so no skew) and
writes customer_ID, probability_of_default and risk_band. It streams the input and scores in chunks of whole
customers, so any size fits in memory. Reason codes are skipped here, the API makes them on demand. It's meant to run
as a Cloud Run Job and reads and writes local or gs:// paths.

    python -m app.batch_score --input portfolio.parquet --output scores.parquet
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import pyarrow.parquet as pq

from app.model import CreditModel
from app.pipeline import ID_COL

OUT_COLS = ["customer_id", "probability_of_default", "risk_band"]


def run(input_path: str, output_path: str, chunk_customers: int = 25_000,
        row_batch: int = 1_000_000) -> dict:
    model = CreditModel()
    pf = pq.ParquetFile(input_path)

    carry = None                  # rows of the last (possibly incomplete) customer
    pending: list[pd.DataFrame] = []
    pending_custs = 0
    out_frames: list[pd.DataFrame] = []
    t0 = time.time()
    n_scored = 0

    def flush():
        nonlocal pending, pending_custs, n_scored
        if not pending:
            return
        chunk = pd.concat(pending, ignore_index=True)
        res = model.score(chunk, with_reasons=False)
        out_frames.append(pd.DataFrame(res)[OUT_COLS])
        n_scored += len(res)
        pending, pending_custs = [], 0

    for batch in pf.iter_batches(batch_size=row_batch):
        df = batch.to_pandas()
        if carry is not None:
            df = pd.concat([carry, df], ignore_index=True)
        last = df[ID_COL].iloc[-1]            # may continue into the next batch
        carry = df[df[ID_COL] == last]
        complete = df[df[ID_COL] != last]
        if len(complete):
            pending.append(complete)
            pending_custs += complete[ID_COL].nunique()
            if pending_custs >= chunk_customers:
                flush()
    if carry is not None and len(carry):       # final customer is complete
        pending.append(carry)
        pending_custs += 1
    flush()

    out = pd.concat(out_frames, ignore_index=True)
    out.to_parquet(output_path, index=False)
    dt = time.time() - t0
    stats = {"customers": int(n_scored), "seconds": round(dt, 1),
             "throughput_per_s": round(n_scored / dt, 1) if dt else None,
             "default_rate": round(float((out["probability_of_default"]
                                          >= 0.5).mean()), 4)}
    print(f"scored {stats['customers']:,} customers in {stats['seconds']}s "
          f"({stats['throughput_per_s']:,}/s) -> {output_path}")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="portfolio statements parquet")
    ap.add_argument("--output", required=True, help="output predictions parquet")
    ap.add_argument("--chunk-customers", type=int, default=25_000)
    args = ap.parse_args()
    run(args.input, args.output, args.chunk_customers)
