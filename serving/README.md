# AMEX Default-Prediction — Inference Service

Turns a customer's **raw monthly statements** into a **calibrated probability of
default**, a **risk band**, and **SHAP adverse-action reason codes**, in the two
modes a real card issuer actually uses:

* **Batch portfolio scoring (primary)** — [`app/batch_score.py`](app/batch_score.py).
  Behavioural default models are scored in **batch**: the inputs (monthly
  statements) only change once per cycle, and the decisions they feed
  (credit-line reviews, risk-based pricing, collections, IFRS 9 / CECL
  provisioning) are **periodic portfolio runs**, not point-of-event decisions.
  Scores the **entire 924,621-customer test portfolio in ~6 min (~2,500
  customers/s)** on one machine; memory-safe by streaming customer-contiguous
  chunks, so any portfolio size fits. Runs as a scheduled **Cloud Run Job**.
* **Real-time API (on-demand)** — [`app/main.py`](app/main.py). A **FastAPI**
  service for single-customer lookups (a risk analyst or servicing agent pulling
  one account's current PD + reason codes), deployed on **Cloud Run**.

Both modes call the **same model and the same feature code** — verified
identical to the bit (`tests/test_batch.py` checks batch == API on the same
customers) — so there is no train/serve **or** batch/online skew.

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

# 2c. batch-score a portfolio parquet (primary deployment)
python -m app.batch_score --input portfolio.parquet --output scores.parquet \
  --chunk-customers 25000   # run from the serving/ dir

# 3. test
pytest serving/tests -q
```

## Deploy

```bash
# Real-time API -> Cloud Run service (Cloud Build → Artifact Registry → Cloud Run)
bash serving/deploy_cloudrun.sh

# Batch scoring -> Cloud Run Job (same image, different entrypoint), schedulable
# monthly via Cloud Scheduler:
gcloud run jobs deploy amex-batch-score --image="$IMAGE" --region="$REGION" \
  --command python --args=-m,app.batch_score,\
--input,gs://BUCKET/portfolio.parquet,--output,gs://BUCKET/scores.parquet \
  --memory=4Gi --cpu=2 --task-timeout=3600
gcloud run jobs execute amex-batch-score --region="$REGION"
```
The service is serverless and scales to zero (no idle cost). CI/CD in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) tests and builds on
every push, and deploys to Cloud Run on `main` once `GCP_SA_KEY` / `GCP_PROJECT`
secrets are set.

## Production notes (honest limitations)

* **Fairness:** the features are anonymized, so protected-attribute bias testing
  isn't possible on this dataset — it would be required before real deployment.
* **Model registry:** the artifact is versioned by its training iteration; a real
  deployment would push to a registry (Vertex AI Model Registry) with lineage.
* **Auth:** the demo API is public (`--allow-unauthenticated`) so it can be
  curled; a real deployment would put it behind IAM / an API gateway.
