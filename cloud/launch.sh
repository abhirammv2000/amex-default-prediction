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
# Image + optional GPU. For GPU jobs pass ACCELERATOR=type=nvidia-tesla-t4,count=1
# and a Deep-Learning image (PyTorch+CUDA preinstalled).
IMAGE_FAMILY="${IMAGE_FAMILY:-debian-12}"
IMAGE_PROJECT="${IMAGE_PROJECT:-debian-cloud}"
ACCELERATOR="${ACCELERATOR:-}"
BOOT_DISK="${BOOT_DISK:-80}"                  # GB; Deep-Learning GPU images need >=100
# Per-job overrides for the generalized bootstrap.
PYTHON="${PYTHON:-python3}"
PIPDEPS="${PIPDEPS-numpy pandas pyarrow scikit-learn lightgbm xgboost optuna}"
INPUTS="${INPUTS-data/processed/train_features.parquet data/processed/categorical_features.txt data/train_labels.csv}"
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
# Code is always refreshed (cheap). Each declared INPUT is uploaded only when
# missing in the bucket (or FORCE_UPLOAD=1) — large tensors/tables don't change
# between runs, so we don't re-ship them.
gcloud storage cp "$HERE/src/"*.py "$BUCKET/src/"
for rel in $INPUTS; do
  case "$rel" in
    data/*.csv) local="$HERE/amex-default-prediction/$(basename "$rel")" ;;
    *)          local="$HERE/$rel" ;;
  esac
  if [[ "${FORCE_UPLOAD:-0}" == "1" ]] || ! gcloud storage ls "$BUCKET/$rel" >/dev/null 2>&1; then
    echo "uploading $rel ..."
    gcloud storage cp "$local" "$BUCKET/$rel"
  else
    echo "$rel already in bucket - skipping"
  fi
done
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
# GPU: attach the accelerator, force the host to TERMINATE on maintenance (GPUs
# can't live-migrate) and have the Deep-Learning image auto-install the driver.
GPU_FLAGS=()
GPU_META=""
# GPU mode for an explicit accelerator (N1+T4) OR a GPU-native machine (g2=L4,
# a2=A100, which embed the GPU and reject --accelerator).
if [[ -n "$ACCELERATOR" || "$MACHINE" == g2-* || "$MACHINE" == a2-* ]]; then
  [[ -n "$ACCELERATOR" ]] && GPU_FLAGS+=(--accelerator="$ACCELERATOR")
  GPU_FLAGS+=(--maintenance-policy=TERMINATE)
  GPU_META=",install-nvidia-driver=True"
fi
# Values may contain spaces (runcmd, pipdeps, inputs) but never commas, so a
# single comma-joined --metadata string is safe.
META="bucket=$BUCKET,shutdown=1,jobname=$JOB,runcmd=$RUNCMD,python=$PYTHON,pipdeps=$PIPDEPS,inputs=$INPUTS$GPU_META"
gcloud compute instances create "$VM" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  "${PROV_FLAGS[@]}" "${GPU_FLAGS[@]}" \
  --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="${BOOT_DISK}GB" --boot-disk-type=pd-balanced \
  --scopes=cloud-platform \
  --metadata="$META" \
  --metadata-from-file=startup-script="$HERE/cloud/bootstrap_vm.sh"

echo "VM created. Poll:    gcloud storage cat $BUCKET/results/$JOB/_STATUS"
echo "Pull results:        gcloud storage cp -r $BUCKET/results/$JOB ./outputs/cloud_$JOB"
echo "Delete VM:           gcloud compute instances delete $VM --zone=$ZONE -q"
