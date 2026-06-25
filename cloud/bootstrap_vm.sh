#!/usr/bin/env bash
# GCE startup-script: trains the AMEX LightGBM model on a high-memory VM.
#
# Designed to run unattended as the VM's startup-script. It reads the GCS
# bucket from instance metadata, pulls code + features, trains, pushes results
# back to GCS (always, even on failure), and powers the VM off so billing stops.
#
# Metadata attributes expected:
#   bucket   -> gs://<bucket>      (required)
#   shutdown -> "1" to power off when done (default "1")
set -uo pipefail

LOG=/var/log/amex_bootstrap.log
exec > >(tee -a "$LOG") 2>&1
echo "=== bootstrap start $(date -u) ==="

meta() { curl -s -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }

BUCKET="$(meta bucket)"
SHUTDOWN="$(meta shutdown)"; SHUTDOWN="${SHUTDOWN:-1}"
RUNCMD="$(meta runcmd)"; RUNCMD="${RUNCMD:-python3 -u train_baseline.py}"
JOB="$(meta jobname)"; JOB="${JOB:-train}"
WORK="/opt/amex"
RESULTS="$BUCKET/results/$JOB"

if [[ -z "$BUCKET" ]]; then echo "FATAL: no bucket metadata"; exit 1; fi
echo "bucket=$BUCKET work=$WORK shutdown=$SHUTDOWN job=$JOB runcmd='$RUNCMD'"

# Always try to ship logs + a status marker back, whatever happens.
finish() {
  local status="$1"
  echo "=== bootstrap finish: $status $(date -u) ==="
  gcloud storage cp "$LOG" "$RESULTS/bootstrap.log" || true
  echo "$status" | gcloud storage cp - "$RESULTS/_STATUS" || true
  if [[ "$SHUTDOWN" == "1" ]]; then sudo poweroff; fi
}
trap 'finish FAILED' ERR

# --- system deps -----------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv libgomp1
# Debian 12 marks the system Python as externally managed (PEP 668); this is a
# disposable VM, so install straight into it with --break-system-packages.
PIP="python3 -m pip install --quiet --break-system-packages"
$PIP --upgrade pip
$PIP numpy pandas pyarrow scikit-learn lightgbm xgboost optuna

# --- layout matching config.py (ROOT/src, ROOT/data/processed, ROOT/amex-default-prediction) ---
sudo mkdir -p "$WORK" && sudo chown -R "$(whoami)" "$WORK"
mkdir -p "$WORK/src" "$WORK/data/processed" "$WORK/amex-default-prediction" \
         "$WORK/outputs/models"

# --- pull code + data from GCS --------------------------------------------
gcloud storage cp "$BUCKET/src/*.py" "$WORK/src/"
gcloud storage cp "$BUCKET/data/processed/train_features.parquet" "$WORK/data/processed/"
gcloud storage cp "$BUCKET/data/processed/categorical_features.txt" "$WORK/data/processed/"
gcloud storage cp "$BUCKET/data/train_labels.csv" "$WORK/amex-default-prediction/"

# --- run the job -----------------------------------------------------------
cd "$WORK/src"
echo "=== job '$JOB' start $(date -u): $RUNCMD ==="
eval "$RUNCMD"
echo "=== job done $(date -u) ==="

# --- push results (everything under outputs/ + OOF) ------------------------
gcloud storage cp -r "$WORK/outputs/"* "$RESULTS/" || true
gcloud storage cp "$WORK/data/processed/oof_predictions.parquet" "$RESULTS/" || true

finish SUCCESS
