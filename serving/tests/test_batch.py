"""Batch scorer test — runs the job on a tiny synthetic portfolio (no data
files needed) and checks the output contract."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "serving"))

ART = ROOT / "serving" / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "model.txt").exists(),
                                reason="serving model artifact not built")


def _portfolio() -> pd.DataFrame:
    rows = []
    for cust in ("cust_a", "cust_b", "cust_c"):       # customer-contiguous
        for date, p2, d39 in [("2018-01-31", 0.6, 0.0),
                              ("2018-02-28", 0.4, 0.5),
                              ("2018-03-31", 0.1, 2.0)]:
            rows.append({"customer_ID": cust, "S_2": date, "P_2": p2, "D_39": d39})
    return pd.DataFrame(rows)


def test_batch_score(tmp_path):
    from app.batch_score import run
    inp, out = tmp_path / "portfolio.parquet", tmp_path / "scores.parquet"
    _portfolio().to_parquet(inp, index=False)

    stats = run(str(inp), str(out), chunk_customers=2)
    assert stats["customers"] == 3

    res = pd.read_parquet(out)
    assert set(res.columns) == {"customer_id", "probability_of_default", "risk_band"}
    assert len(res) == 3
    assert res["probability_of_default"].between(0, 1).all()
    assert set(res["customer_id"]) == {"cust_a", "cust_b", "cust_c"}
