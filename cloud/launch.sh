#!/usr/bin/env bash
# Provision a high-memory GCP spot VM and kick off an unattended job (training,
# tuning, ...) via the startup-script (cloud/bootstrap_vm.sh). Results land in
# gs://<bucket>/results/<job>/ with a _STATUS marker; the VM powers off when done.
#
# Prereqs: `gcloud auth login` done; billing enabled on the project.
# Usage:
#   bash cloud/launch.sh                       # default: full 5-fold training
#   JOB=tune RUNCMD="python3 -u tune.py --n-trials 50" bash cloud/launch.sh
#   FORCE_UPLOAD=1 ... bash cloud/launch.sh    # re-upload the feature table
set -euo pipefail

# ---- config (override via env) --------------------------------------------
PROJECT="$(gcloud config get-value project 2>/dev/null)"
REGION="us-central1"
ZONE="${ZONE:-us-central1-a}"
MACHINE="${MACHINE:-n2-highmem-8}"           # default 8 vCPU, 64 GB RAM
PROVISIONING="${PROVISIONING:-SPOT}"         # SPOT (cheap) or STANDARD (no preemption)
JOB="${JOB:-train}"
RUNCMD="${RUNCMD:-python3 -u train_baseline.py}"
VM="amex-${JOB}"
BUCKET="gs://amex-train-${PROJECT}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"     # repo root

echo "project=$PROJECT zone=$ZONE job=$JOB vm=$VM bucket=$BUCKET"
echo "runcmd='$RUNCMD'"

# ---- enable APIs ----------------------------------------------------------
gcloud services enable compute.googleapis.com storage.googleapis.com --quiet

# ---- bucket + upload (gcloud storage; gsutil is broken on this host) -------
gcloud storage ls "$BUCKET" >/dev/null 2>&1 || \
  gcloud storage buckets create "$BUCKET" --location="$REGION"
# Code is always refreshed (cheap); the 3 GB feature table only when missing
# or FORCE_UPLOAD=1 (it does not change between model iterations).
gcloud storage cp "$HERE/src/"*.py                                "$BUCKET/src/"
gcloud storage cp "$HERE/data/processed/categorical_features.txt" "$BUCKET/data/processed/"
gcloud storage cp "$HERE/amex-default-prediction/train_labels.csv" "$BUCKET/data/"
if [[ "${FORCE_UPLOAD:-0}" == "1" ]] || \
   ! gcloud storage ls "$BUCKET/data/processed/train_features.parquet" >/dev/null 2>&1; then
  echo "uploading feature table (~3 GB) ..."
  gcloud storage cp "$HERE/data/processed/train_features.parquet" "$BUCKET/data/processed/"
else
  echo "feature table already in bucket - skipping upload"
fi
# clear any stale status marker for this job
gcloud storage rm "$BUCKET/results/$JOB/_STATUS" 2>/dev/null || true

# ---- create the spot VM with the job startup-script -----------------------
# Remove any same-named VM left over from a previous run (e.g. a spot preemption
# that skipped the self-delete hook) so the create below doesn't name-clash.
gcloud compute instances delete "$VM" --zone="$ZONE" -q 2>/dev/null || true
# Spot is ~4x cheaper but can be preempted (bad for long jobs); STANDARD won't be.
if [[ "$PROVISIONING" == "SPOT" ]]; then
  PROV_FLAGS=(--provisioning-model=SPOT --instance-termination-action=STOP)
else
  PROV_FLAGS=(--provisioning-model=STANDARD)
fi
gcloud compute instances create "$VM" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  "${PROV_FLAGS[@]}" \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=60GB --boot-disk-type=pd-balanced \
  --scopes=cloud-platform \
  --metadata=bucket="$BUCKET",shutdown=1,jobname="$JOB",runcmd="$RUNCMD" \
  --metadata-from-file=startup-script="$HERE/cloud/bootstrap_vm.sh"

echo "VM created. Poll:    gcloud storage cat $BUCKET/results/$JOB/_STATUS"
echo "Pull results:        gcloud storage cp -r $BUCKET/results/$JOB ./outputs/cloud_$JOB"
echo "Delete VM:           gcloud compute instances delete $VM --zone=$ZONE -q"
