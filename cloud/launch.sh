#!/usr/bin/env bash
# Provision a high-memory GCP spot VM, ship code + features to GCS, and kick off
# unattended training via the startup-script (cloud/bootstrap_vm.sh).
#
# Prereqs: `gcloud auth login` done; billing enabled on the project.
# Usage:   bash cloud/launch.sh
set -euo pipefail

# ---- config ---------------------------------------------------------------
PROJECT="$(gcloud config get-value project 2>/dev/null)"
REGION="us-central1"
ZONE="us-central1-a"
VM="amex-train"
MACHINE="n2-highmem-8"          # 8 vCPU, 64 GB RAM
BUCKET="gs://amex-train-${PROJECT}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # repo root

echo "project=$PROJECT zone=$ZONE machine=$MACHINE bucket=$BUCKET"

# ---- enable APIs ----------------------------------------------------------
gcloud services enable compute.googleapis.com storage.googleapis.com --quiet

# ---- bucket + upload (gcloud storage; gsutil is broken on this host) -------
gcloud storage ls "$BUCKET" >/dev/null 2>&1 || \
  gcloud storage buckets create "$BUCKET" --location="$REGION"
gcloud storage cp "$HERE/src/"*.py                                "$BUCKET/src/"
gcloud storage cp "$HERE/data/processed/train_features.parquet"   "$BUCKET/data/processed/"
gcloud storage cp "$HERE/data/processed/categorical_features.txt" "$BUCKET/data/processed/"
gcloud storage cp "$HERE/amex-default-prediction/train_labels.csv" "$BUCKET/data/"
# clear any stale status marker from a previous run
gcloud storage rm "$BUCKET/results/_STATUS" 2>/dev/null || true

# ---- create the spot VM with the training startup-script ------------------
gcloud compute instances create "$VM" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=60GB --boot-disk-type=pd-balanced \
  --scopes=cloud-platform \
  --metadata=bucket="$BUCKET",shutdown=1 \
  --metadata-from-file=startup-script="$HERE/cloud/bootstrap_vm.sh"

echo "VM created. Poll for completion with:"
echo "  gsutil cat $BUCKET/results/_STATUS"
echo "When SUCCESS, pull results with:"
echo "  gsutil -m cp -r $BUCKET/results ./outputs/cloud_results"
echo "Then delete the VM:  gcloud compute instances delete $VM --zone=$ZONE -q"
