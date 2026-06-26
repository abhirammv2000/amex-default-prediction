"""Extract each customer's last statement date from the train Parquet.

Needed for out-of-time validation: we split customers by *when they were last
observed* (their latest S_2), training on earlier customers and testing on later
ones — the realistic way to validate a credit model (will it hold up on future
applicants?). Output: data/processed/customer_dates.parquet.
"""
from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

import config


def main() -> None:
    tbl = pq.read_table(config.TRAIN_PARQUET, columns=[config.ID_COL, config.DATE_COL])
    df = tbl.to_pandas()
    last = df.groupby(config.ID_COL)[config.DATE_COL].max().reset_index()
    last = last.rename(columns={config.DATE_COL: "last_statement"})
    out = config.PROCESSED_DIR / "customer_dates.parquet"
    last.to_parquet(out, index=False)
    print(f"{len(last):,} customers | last_statement "
          f"{last['last_statement'].min()} -> {last['last_statement'].max()}")
    print(last["last_statement"].dt.to_period("M").value_counts().sort_index().to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
