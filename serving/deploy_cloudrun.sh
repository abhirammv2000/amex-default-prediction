#!/usr/bin/env bash
# Build the container with Cloud Build and deploy it to Cloud Run.
set -euo pipefail
PROJECT="$(gcloud config get-value project 2>/dev/null)"
REGION="${REGION:-us-central1}"
SERVICE="amex-default-api"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/amex/${SERVICE}:latest"
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # repo root (build context)

gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com --quiet
gcloud artifacts repositories describe amex --location="$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create amex --repository-format=docker --location="$REGION"

CFG="$(mktemp --suffix=.yaml)"
cat > "$CFG" <<CONFIG
steps:
  - name: gcr.io/cloud-builders/docker
    args: ["build","-f","serving/Dockerfile","-t","${IMAGE}","."]
images: ["${IMAGE}"]
CONFIG
gcloud builds submit "$HERE" --config="$CFG"
rm -f "$CFG"

gcloud run deploy "$SERVICE" --image="$IMAGE" --region="$REGION" \
  --platform=managed --allow-unauthenticated \
  --memory=2Gi --cpu=2 --concurrency=20 --min-instances=0 --max-instances=3 \
  --port=8080
echo "Deployed. URL:"; gcloud run services describe "$SERVICE" --region="$REGION" \
  --format="value(status.url)"
