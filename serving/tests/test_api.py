"""API contract tests using FastAPI's TestClient.

Builds a request from real raw statements (a couple of customers) and checks the
health and scoring endpoints return a well-formed, sane response.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "serving"))
import config  # noqa: E402

ART = ROOT / "serving" / "artifacts"
pytestmark = pytest.mark.skipif(
    not (ART / "model.txt").exists(),
    reason="serving model artifact not built")


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def _sample_request(n_customers: int = 2) -> dict:
    tbl = pq.read_table(config.TRAIN_PARQUET).slice(0, 60000).to_pandas()
    ids = tbl[config.ID_COL].unique()[:n_customers]
    rows = tbl[tbl[config.ID_COL].isin(ids)].copy()
    rows[config.DATE_COL] = rows[config.DATE_COL].astype(str)
    # Cast to object first so NaN -> None actually sticks (on float columns a
    # plain .where recasts None back to NaN), giving valid JSON nulls.
    rows = rows.astype(object).where(pd.notna(rows), None)
    return {"statements": rows.to_dict(orient="records")}


def _synthetic_request() -> dict:
    """A customer with 3 statements and a few raw features — no data files needed
    (missing features become NaN, which the model handles). Runs in CI."""
    base = {"customer_ID": "synthetic_0001"}
    stmts = []
    for i, (date, p2, b1) in enumerate([("2018-01-31", 0.55, 0.02),
                                        ("2018-02-28", 0.40, 0.05),
                                        ("2018-03-31", 0.18, 0.20)]):
        stmts.append({**base, "S_2": date, "P_2": p2, "B_1": b1})
    return {"statements": stmts}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True and body["n_features"] == 1628


def test_score_synthetic(client):
    """End-to-end scoring with no dependency on the raw data (CI-friendly)."""
    r = client.post("/score", json=_synthetic_request())
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert 0.0 <= res["probability_of_default"] <= 1.0
    assert res["risk_band"] in {"very low", "low", "medium", "high", "very high"}


@pytest.mark.skipif(not config.TRAIN_PARQUET.exists(),
                    reason="raw parquet not present")
def test_score(client):
    r = client.post("/score", json=_sample_request(2))
    assert r.status_code == 200
    body = r.json()
    assert body["n_customers"] == 2
    for res in body["results"]:
        assert 0.0 <= res["probability_of_default"] <= 1.0
        assert res["risk_band"] in {"very low", "low", "medium", "high", "very high"}
        assert len(res["top_reason_codes"]) >= 1
        assert all("description" in rc for rc in res["top_reason_codes"])


def test_validation_rejects_empty(client):
    assert client.post("/score", json={"statements": []}).status_code == 422
