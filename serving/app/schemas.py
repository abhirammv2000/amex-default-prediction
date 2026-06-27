"""Typed request/response contract (Pydantic v2)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Statement(BaseModel):
    """One monthly statement. customer_ID and S_2 (date) are required; the ~188
    anonymised feature columns are accepted as extra fields."""
    model_config = ConfigDict(extra="allow")
    customer_ID: str
    S_2: str


class ScoreRequest(BaseModel):
    statements: list[Statement] = Field(
        ..., min_length=1,
        description="Raw monthly statements for one or more customers "
                    "(up to 13 per customer).")


class ReasonCode(BaseModel):
    feature: str
    description: str
    contribution: float


class CustomerScore(BaseModel):
    customer_id: str
    probability_of_default: float
    raw_score: float
    risk_band: str
    top_reason_codes: list[ReasonCode]


class ScoreResponse(BaseModel):
    model_version: int
    n_customers: int
    results: list[CustomerScore]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_features: int
    model_version: int
