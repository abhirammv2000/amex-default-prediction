"""Observability: structured JSON logging + Prometheus metrics + a drift hook.

Production credit models must be monitored. We emit one structured log line per
scored customer (the raw material for the offline PSI drift job in
src/drift.py), and expose Prometheus metrics so request rate, latency and the
live predicted-PD distribution can be scraped/alerted on.
"""
from __future__ import annotations

import json
import logging
import sys
import time

from prometheus_client import Counter, Histogram

# --- structured JSON logging ------------------------------------------------
_handler = logging.StreamHandler(sys.stdout)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": time.time(), "level": record.levelname,
                   "msg": record.getMessage()}
        if isinstance(record.args, dict):
            payload.update(record.args)
        return json.dumps(payload)


_handler.setFormatter(_JsonFormatter())
logger = logging.getLogger("amex_serving")
logger.setLevel(logging.INFO)
logger.handlers = [_handler]
logger.propagate = False

# --- Prometheus metrics -----------------------------------------------------
REQUESTS = Counter("amex_score_requests_total", "Score requests", ["endpoint"])
CUSTOMERS = Counter("amex_scored_customers_total", "Customers scored")
LATENCY = Histogram("amex_score_latency_seconds", "Scoring latency", ["endpoint"])
PD_HIST = Histogram("amex_predicted_pd", "Predicted probability of default",
                    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])


def log_predictions(results: list[dict]) -> None:
    """Emit one structured line per prediction + update the PD histogram."""
    for r in results:
        pd_val = r["probability_of_default"]
        PD_HIST.observe(pd_val)
        logger.info("prediction", {"customer_id": r["customer_id"],
                                   "pd": pd_val, "risk_band": r["risk_band"]})
    CUSTOMERS.inc(len(results))
