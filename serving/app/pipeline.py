"""Feature engineering shared by training and serving.

This is the single source of truth that turns a customer's raw monthly
statements into the exact feature vector the model was trained on. Reusing the
*same* aggregation calls as the offline training pipeline
(`src/feature_engineering.py`) is what prevents **training/serving skew** — the
silent failure mode where an online model is fed subtly different features than
it saw in training. The accompanying test (`tests/test_pipeline.py`) asserts the
output matches the offline feature table row-for-row.

The function operates on a batch of customers (groupby) so a single request and
a bulk batch share one code path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID_COL = "customer_ID"
DATE_COL = "S_2"
CATEGORICAL_FEATURES = [
    "B_30", "B_38", "D_114", "D_116", "D_117", "D_120",
    "D_126", "D_63", "D_64", "D_66", "D_68",
]
NUM_AGGS = ["mean", "std", "min", "max", "first", "last"]
CAT_AGGS = ["last", "nunique", "count"]


def _flatten(agg: pd.DataFrame) -> pd.DataFrame:
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg


def engineer_features(statements: pd.DataFrame, cat_maps: dict,
                      feature_order: list[str]) -> pd.DataFrame:
    """Aggregate raw statements -> one feature row per customer.

    Parameters
    ----------
    statements : raw statement rows (customer_ID, S_2, and the 188 features).
    cat_maps   : {categorical -> [category strings in code order]} fit on train.
    feature_order : canonical training column order; output is reindexed to it.
    """
    df = statements.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    # Chronological order within each customer so first/last match training.
    df = df.sort_values([ID_COL, DATE_COL])

    feat_cols = [c for c in df.columns if c not in (ID_COL, DATE_COL)]
    # Only aggregate columns actually present in the request; any column the model
    # expects but the caller omitted is filled with NaN by the final reindex
    # (LightGBM handles NaN natively), so partial requests are tolerated.
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feat_cols]
    num_cols = [c for c in feat_cols if c not in cat_cols]
    g = df.groupby(ID_COL, sort=True)
    parts = []

    # --- numeric aggregations + trend/deviation diffs (identical to training) -
    if num_cols:
        num = _flatten(g[num_cols].agg(NUM_AGGS).astype(np.float32))
        diffs = {}
        for c in num_cols:
            last, mean = num[f"{c}_last"], num[f"{c}_mean"]
            diffs[f"{c}_last_mean_diff"] = (last - mean).astype(np.float32)
            diffs[f"{c}_last_first_diff"] = (last - num[f"{c}_first"]).astype(np.float32)
            diffs[f"{c}_range"] = (num[f"{c}_max"] - num[f"{c}_min"]).astype(np.float32)
        parts.append(pd.concat([num, pd.DataFrame(diffs, index=num.index)], axis=1))

    # --- categorical aggregations; encode 'last' with the train-fit maps ------
    if cat_cols:
        cat = _flatten(g[cat_cols].agg(CAT_AGGS))
        for c in cat_cols:
            col = f"{c}_last"
            lut = {v: i for i, v in enumerate(cat_maps[c]) if v is not None}
            cat[col] = (cat[col].astype("object")
                        .map(lambda v: lut.get(None if pd.isna(v) else str(v), -1))
                        .astype(np.int16))
        num_like = [c for c in cat.columns if not c.endswith("_last")]
        cat[num_like] = cat[num_like].astype(np.float32)
        parts.append(cat)

    # --- date-derived: statement count + history span in days ----------------
    span = g[DATE_COL].agg(["count", "min", "max"])
    span["history_days"] = (span["max"] - span["min"]).dt.days.astype(np.float32)
    span = span.rename(columns={"count": "statement_count"})[
        ["statement_count", "history_days"]].astype(np.float32)
    parts.append(span)

    features = pd.concat(parts, axis=1)
    # Reindex to the exact training column order (and fill any column the model
    # expects but this slice didn't produce — defensive, should not happen).
    return features.reindex(columns=feature_order)
