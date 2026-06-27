"""Model loading, calibrated scoring, and SHAP adverse-action reason codes."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from app.pipeline import CATEGORICAL_FEATURES, engineer_features

ART = Path(__file__).resolve().parents[1] / "artifacts"

# Anonymised features carry an informative prefix; turn a feature name into a
# plain-English driver for adverse-action explanations.
_PREFIX = {"P": "payment", "D": "delinquency", "B": "balance",
           "R": "risk", "S": "spend"}
_AGG = {
    "last": "most recent", "first": "earliest", "mean": "average",
    "min": "lowest", "max": "highest", "std": "volatility of",
    "range": "range of", "last_mean_diff": "recent deviation in",
    "last_first_diff": "trend in", "nunique": "number of distinct",
    "count": "number of", "statement_count": "statement count",
    "history_days": "length of history",
}


def _describe(feature: str) -> str:
    if feature in ("statement_count", "history_days"):
        return _AGG[feature]
    base, agg = feature.rsplit("_", 1)
    for multi in ("last_mean_diff", "last_first_diff"):
        if feature.endswith(multi):
            base, agg = feature[: -len(multi) - 1], multi
            break
    grp = _PREFIX.get(base.split("_")[0], "account")
    return f"{_AGG.get(agg, agg)} {grp} ({base})"


def _band(pd_value: float) -> str:
    for thr, name in [(0.05, "very low"), (0.20, "low"), (0.50, "medium"),
                      (0.80, "high")]:
        if pd_value < thr:
            return name
    return "very high"


class CreditModel:
    def __init__(self, art_dir: Path = ART):
        self.booster = lgb.Booster(model_file=str(art_dir / "model.txt"))
        self.feature_order = json.loads((art_dir / "feature_names.json").read_text())
        self.cat_maps = json.loads((art_dir / "categorical_maps.json").read_text())
        cal = json.loads((art_dir / "calibrator.json").read_text())
        self._cal_x, self._cal_y = np.array(cal["x"]), np.array(cal["y"])
        self.meta = json.loads((art_dir / "model_metadata.json").read_text())
        self.explainer = shap.TreeExplainer(self.booster)
        self.version = self.meta.get("best_iteration", 0)

    def _calibrate(self, p: np.ndarray) -> np.ndarray:
        return np.interp(p, self._cal_x, self._cal_y)

    def score(self, statements: pd.DataFrame, n_reasons: int = 4) -> list[dict]:
        """Score a batch of customers; returns PD, risk band and reason codes."""
        feats = engineer_features(statements, self.cat_maps, self.feature_order)
        ids = feats.index.tolist()
        raw = self.booster.predict(feats)
        cal = self._calibrate(np.asarray(raw))

        sv = self.explainer.shap_values(feats)
        if isinstance(sv, list):           # some shap versions: [class0, class1]
            sv = sv[1]
        sv = np.asarray(sv)

        def _finite(x: float, default: float = 0.0) -> float:
            x = float(x)
            return x if np.isfinite(x) else default

        out = []
        for i, cid in enumerate(ids):
            order = np.argsort(sv[i])[::-1]      # features pushing PD up first
            reasons = [{"feature": self.feature_order[j],
                        "description": _describe(self.feature_order[j]),
                        "contribution": round(float(sv[i][j]), 4)}
                       for j in order[:n_reasons]
                       if np.isfinite(sv[i][j]) and sv[i][j] > 0]
            pd_val = _finite(cal[i])
            out.append({
                "customer_id": str(cid),
                "probability_of_default": round(pd_val, 6),
                "raw_score": round(_finite(raw[i]), 6),
                "risk_band": _band(pd_val),
                "top_reason_codes": reasons,
            })
        return out
