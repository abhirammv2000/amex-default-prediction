# AMEX Default-Prediction — Inference Service

A production-style inference service for the credit-default model: it turns a
customer's **raw monthly statements** into a **calibrated probability of
default**, a **risk band**, and **SHAP adverse-action reason codes** — served
over a typed REST API, containerized, and deployable to Cloud Run.

## Why this design

| Decision | Reasoning |
|----------|-----------|
| **Serve the calibrated LightGBM**, not the 3-way blend | Single model: fast, well-calibrated PDs, and *explainable*. Regulated lending needs reason codes; a research blend doesn't pass model-risk review. The blend stays an offline benchmark. |
| **One feature-engineering codebase** for train **and** serve (`app/pipeline.py`) | Eliminates **training/serving skew** — the silent failure where the online model sees subtly different features. Proven by `tests/test_pipeline.py`, which asserts the API's features match the offline training table row-for-row. |
| **Final model trained on all data** (`train_serving_model.py`) | CV was for model *selection*; production uses one model on 100% of the data, with an isotonic calibrator fit on a holdout. |
| **Reason codes in the response** | SHAP per-prediction attributions → adverse-action explanations (ECOA). |
| **Prometheus `/metrics` + structured logs** | Real observability; prediction logs feed the offline PSI drift job (`src/drift.py`). |

## Architecture

```
            raw statements (JSON)
                    │
              FastAPI /score
                    │
   app/pipeline.py  │  ← SAME code as training (no skew)
   engineer_features│
                    ▼
        1,628-feature vector
                    │
   app/model.py     │  LightGBM → isotonic calibration → SHAP
                    ▼
   { probability_of_default, risk_band, top_reason_codes }
                    │
   app/monitoring.py│  structured log + Prometheus PD histogram
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/score` | score one / a few customers in real time |
| `POST` | `/score/batch` | same contract, bulk portfolio scoring |
| `GET`  | `/health` | liveness + model status |
| `GET`  | `/metrics` | Prometheus metrics |
| `GET`  | `/docs` | OpenAPI / Swagger UI |

**Example**

```bash
curl -s localhost:8080/score -H 'content-type: application/json' -d '{
  "statements": [
    {"customer_ID":"c1","S_2":"2018-01-31","P_2":0.55,"B_1":0.02},
    {"customer_ID":"c1","S_2":"2018-02-28","P_2":0.40,"B_1":0.05},
    {"customer_ID":"c1","S_2":"2018-03-31","P_2":0.18,"B_1":0.20}
  ]}'
```
```json
{
  "model_version": 612,
  "n_customers": 1,
  "results": [{
    "customer_id": "c1",
    "probability_of_default": 0.83,
    "risk_band": "very high",
    "top_reason_codes": [
      {"feature":"P_2_last","description":"most recent payment (P_2)","contribution":0.84},
      {"feature":"D_39_last_mean_diff","description":"recent deviation in delinquency (D_39)","contribution":0.48}
    ]
  }]
}
```

## Run locally

```bash
# 1. build the model artifact (one-time; needs the engineered training data)
python serving/prepare_artifacts.py
python serving/train_serving_model.py

# 2a. run with uvicorn
uvicorn app.main:app --app-dir serving --port 8080
# 2b. or with Docker
docker build -f serving/Dockerfile -t amex-default-api .
docker run -p 8080:8080 amex-default-api

# 3. test
pytest serving/tests -q
```

## Deploy to Cloud Run

```bash
bash serving/deploy_cloudrun.sh    # Cloud Build → Artifact Registry → Cloud Run
```
Serverless, scales to zero (no idle cost). CI/CD in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) lints, tests and builds
on every push, and deploys to Cloud Run on `main` once `GCP_SA_KEY` /
`GCP_PROJECT` secrets are set.

## Production notes (honest limitations)

* **Batch is the natural pattern** for credit risk (monthly statements); the
  real-time API here is the demonstrable surface, with `/score/batch` for bulk.
* **Fairness:** the features are anonymized, so protected-attribute bias testing
  isn't possible on this dataset — it would be required before real deployment.
* **Model registry:** the artifact is versioned by its training iteration; a real
  deployment would push to a registry (Vertex AI Model Registry) with lineage.
