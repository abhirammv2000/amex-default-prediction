"""Feature engineering shared by training and serving.

It turns a customer's raw monthly statements into the feature vector the model was trained on, using the same
aggregation calls as src/feature_engineering.py so the online model never sees different features than it trained
on (training/serving skew). tests/test_pipeline.py checks the output matches the offline table row for row. It
works on a batch of customers, so one request and a bulk batch use the same code.
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
    """One row of features per customer from the raw statement rows (customer_ID, S_2 and the 188 columns).

    cat_maps is {categorical: [category strings in code order]} fit on train, and the output is put in
    feature_order, the training column order.
    """
    df = statements.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    # Chronological order within each customer so first/last match training.
    df = df.sort_values([ID_COL, DATE_COL])

    feat_cols = [c for c in df.columns if c not in (ID_COL, DATE_COL)]
    # only aggregate the columns in the request. Anything the model expects but is missing becomes NaN in
    # the final reindex, which LightGBM handles, so partial requests work
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feat_cols]
    num_cols = [c for c in feat_cols if c not in cat_cols]
    g = df.groupby(ID_COL, sort=True)
    parts = []

    # numeric aggregates and trend features, same as training
    if num_cols:
        num = _flatten(g[num_cols].agg(NUM_AGGS).astype(np.float32))
        diffs = {}
        for c in num_cols:
            last, mean = num[f"{c}_last"], num[f"{c}_mean"]
            diffs[f"{c}_last_mean_diff"] = (last - mean).astype(np.float32)
            diffs[f"{c}_last_first_diff"] = (last - num[f"{c}_first"]).astype(np.float32)
            diffs[f"{c}_range"] = (num[f"{c}_max"] - num[f"{c}_min"]).astype(np.float32)
        parts.append(pd.concat([num, pd.DataFrame(diffs, index=num.index)], axis=1))

    # categorical aggregates, encoding the last value with the maps fit on train
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

    # statement count and days of history
    span = g[DATE_COL].agg(["count", "min", "max"])
    span["history_days"] = (span["max"] - span["min"]).dt.days.astype(np.float32)
    span = span.rename(columns={"count": "statement_count"})[
        ["statement_count", "history_days"]].astype(np.float32)
    parts.append(span)

    features = pd.concat(parts, axis=1)
    # put the columns in training order (fills any the model expects but we didn't make, which shouldn't happen)
    return features.reindex(columns=feature_order)
