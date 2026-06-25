"""Official American Express default-prediction competition metric.

M = 0.5 * (G + D)

  * G = normalized Gini coefficient
  * D = default rate captured at 4% (recall within the top-ranked 4% of
        predictions)

For both sub-metrics the **negative** class is given a weight of 20 to undo
the 5% negative down-sampling applied to the public dataset. Maximum score
is 1.0.

This is a faithful re-implementation of the reference metric Kaggle published
for the competition (see Rohan Rao's "AMEX competition metric" notebook).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _top_four_percent_captured(df: pd.DataFrame) -> float:
    """Fraction of positives captured in the highest-ranked 4% by weight."""
    df = df.sort_values("prediction", ascending=False)
    df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
    four_pct_cutoff = int(0.04 * df["weight"].sum())
    df["weight_cumsum"] = df["weight"].cumsum()
    df_cutoff = df.loc[df["weight_cumsum"] <= four_pct_cutoff]
    return (df_cutoff["target"] == 1).sum() / (df["target"] == 1).sum()


def _weighted_gini(df: pd.DataFrame) -> float:
    df = df.sort_values("prediction", ascending=False)
    df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
    df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
    total_pos = (df["target"] * df["weight"]).sum()
    df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
    df["lorentz"] = df["cum_pos_found"] / total_pos
    df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
    return df["gini"].sum()


def _normalized_weighted_gini(df: pd.DataFrame) -> float:
    pos = (df["target"] == 1).sum()
    neg = (df["target"] == 0).sum()
    df_perfect = pd.DataFrame(
        {"target": [1] * pos + [0] * neg, "prediction": [1] * pos + [0] * neg}
    )
    return _weighted_gini(df) / _weighted_gini(df_perfect)


def amex_metric(y_true, y_pred) -> float:
    """Compute the competition metric.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth binary labels and predicted default probabilities.
    """
    df = pd.DataFrame(
        {"target": np.asarray(y_true).ravel(), "prediction": np.asarray(y_pred).ravel()}
    )
    g = _normalized_weighted_gini(df)
    d = _top_four_percent_captured(df)
    return 0.5 * (g + d)


def amex_metric_np(y_true, y_pred) -> float:
    """Fast pure-NumPy implementation (identical result, much quicker).

    Preferred inside training loops where the metric is evaluated often.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # --- default rate captured at 4% ---------------------------------------
    order = np.argsort(-y_pred)
    t = y_true[order]
    weight = np.where(t == 0, 20.0, 1.0)
    cum_weight = np.cumsum(weight)
    four_pct = 0.04 * weight.sum()
    mask = cum_weight <= four_pct
    d = t[mask].sum() / t.sum()

    # --- normalized weighted Gini ------------------------------------------
    def _gini(sort_key):
        o = np.argsort(-sort_key)
        tt = y_true[o]
        w = np.where(tt == 0, 20.0, 1.0)
        rand = np.cumsum(w / w.sum())
        total_pos = (tt * w).sum()
        lorentz = np.cumsum(tt * w) / total_pos
        return ((lorentz - rand) * w).sum()

    g = _gini(y_pred) / _gini(y_true)
    return 0.5 * (g + d)


def lgb_amex_metric(y_pred, dtrain):
    """LightGBM custom eval: returns (name, value, is_higher_better)."""
    y_true = dtrain.get_label()
    return "amex", amex_metric_np(y_true, y_pred), True
