"""AMEX default-prediction inference service.

A FastAPI app that scores customers from their raw monthly statements and
returns a calibrated probability of default, a risk band, and SHAP-based
adverse-action reason codes. The feature engineering is the *same* code used in
training (app.pipeline) — no training/serving skew.
"""
from __future__ import annotations

import math
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest


def _clean(obj):
    """Recursively replace non-finite floats with None so the response is always
    valid JSON, whatever the inputs."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return super().render(_clean(content))

from app import monitoring as mon
from app.model import CreditModel
from app.schemas import (HealthResponse, ScoreRequest, ScoreResponse)

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["model"] = CreditModel()           # load artifacts once at startup
    mon.logger.info("startup", {"event": "model_loaded",
                                "n_features": STATE["model"].meta["n_features"]})
    yield
    STATE.clear()


app = FastAPI(
    title="AMEX Default-Prediction API",
    description="Calibrated probability of default + adverse-action reason codes.",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=SafeJSONResponse,
)


def _score(req: ScoreRequest, endpoint: str) -> ScoreResponse:
    model: CreditModel = STATE["model"]
    mon.REQUESTS.labels(endpoint=endpoint).inc()
    t0 = time.perf_counter()
    rows = [s.model_dump() for s in req.statements]
    df = pd.DataFrame(rows)
    if "customer_ID" not in df or df.empty:
        raise HTTPException(422, "each statement needs a customer_ID and S_2")
    try:
        results = model.score(df)
    except Exception as exc:                 # surface feature/scoring errors cleanly
        mon.logger.info("error", {"endpoint": endpoint, "detail": str(exc)})
        raise HTTPException(400, f"scoring failed: {exc}") from exc
    mon.LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - t0)
    mon.log_predictions(results)
    return ScoreResponse(model_version=model.version, n_customers=len(results),
                         results=results)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    m = STATE.get("model")
    return HealthResponse(status="ok" if m else "loading", model_loaded=bool(m),
                          n_features=m.meta["n_features"] if m else 0,
                          model_version=m.version if m else 0)


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    """Score one (or a few) customers in real time."""
    return _score(req, "score")


@app.post("/score/batch", response_model=ScoreResponse)
def score_batch(req: ScoreRequest) -> ScoreResponse:
    """Same contract, intended for bulk portfolio scoring."""
    return _score(req, "score_batch")


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root() -> dict:
    return {"service": "amex-default-prediction", "docs": "/docs",
            "endpoints": ["/score", "/score/batch", "/health", "/metrics"]}
