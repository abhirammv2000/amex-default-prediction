"""Convert the raw Kaggle CSVs to compact Parquet files.

The raw ``train_data.csv`` (15.6 GB) and ``test_data.csv`` (32.3 GB) cannot be
loaded into 16 GB of RAM. This script streams them in row chunks, downcasts the
185 numeric columns from float64 to float32 (halving memory and disk), keeps the
two string categoricals (D_63, D_64) and the customer_ID as strings, parses S_2
as a timestamp, and appends each chunk to a Parquet file via a single
``ParquetWriter`` so peak memory stays at ~one chunk.

Usage
-----
    python convert_to_parquet.py --which train
    python convert_to_parquet.py --which test
    python convert_to_parquet.py --which both
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config

# Columns that are NOT float: kept as strings / timestamp.
STRING_COLS = {config.ID_COL, "D_63", "D_64"}
DATE_COLS = {config.DATE_COL}


def _build_schema(columns: list[str]) -> pa.Schema:
    fields = []
    for c in columns:
        if c in STRING_COLS:
            fields.append(pa.field(c, pa.string()))
        elif c in DATE_COLS:
            fields.append(pa.field(c, pa.timestamp("s")))
        else:
            fields.append(pa.field(c, pa.float32()))
    return pa.schema(fields)


def convert(csv_path, parquet_path, chunk_size: int) -> None:
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    float_cols = [c for c in columns if c not in STRING_COLS and c not in DATE_COLS]
    # Force float32 on read so we never materialise a float64 copy.
    read_dtypes = {c: "float32" for c in float_cols}
    schema = _build_schema(columns)

    print(f"\n=== {csv_path.name} -> {parquet_path.name} ===")
    print(f"{len(columns)} columns | {len(float_cols)} float32 | chunk={chunk_size:,}")

    writer = pq.ParquetWriter(parquet_path, schema, compression="snappy")
    t0 = time.time()
    total_rows = 0
    try:
        reader = pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            dtype=read_dtypes,
            parse_dates=[config.DATE_COL],
        )
        for i, chunk in enumerate(reader, 1):
            # Reorder to the schema's column order and cast via the fixed schema.
            table = pa.Table.from_pandas(
                chunk[columns], schema=schema, preserve_index=False
            )
            writer.write_table(table)
            total_rows += len(chunk)
            print(f"  chunk {i:>3}: {total_rows:>12,} rows  "
                  f"({time.time() - t0:6.1f}s)", flush=True)
    finally:
        writer.close()

    size_mb = parquet_path.stat().st_size / 1024**2
    print(f"DONE: {total_rows:,} rows -> {size_mb:,.0f} MB "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["train", "test", "both"], default="both")
    ap.add_argument("--chunk-size", type=int, default=config.CSV_CHUNK_SIZE)
    args = ap.parse_args()

    if args.which in ("train", "both"):
        convert(config.TRAIN_CSV, config.TRAIN_PARQUET, args.chunk_size)
    if args.which in ("test", "both"):
        convert(config.TEST_CSV, config.TEST_PARQUET, args.chunk_size)
