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
meta_root() { curl -s -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }

BUCKET="$(meta bucket)"
SHUTDOWN="$(meta shutdown)"; SHUTDOWN="${SHUTDOWN:-1}"
RUNCMD="$(meta runcmd)"; RUNCMD="${RUNCMD:-python3 -u train_baseline.py}"
JOB="$(meta jobname)"; JOB="${JOB:-train}"
# Configurable so one image serves CPU (debian+pip) and GPU (Deep-Learning VM)
# jobs: PY=interpreter, PIPDEPS=packages to install (empty to skip), INPUTS=GCS
# sub-paths (relative to bucket) to mirror locally.
PY="$(meta python)"; PY="${PY:-python3}"
PIPDEPS="$(meta pipdeps)"
PIPDEPS="${PIPDEPS-numpy pandas pyarrow scikit-learn lightgbm xgboost optuna}"
INPUTS="$(meta inputs)"
INPUTS="${INPUTS:-data/processed/train_features.parquet data/processed/categorical_features.txt data/train_labels.csv}"
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
  if [[ "$SHUTDOWN" == "1" ]]; then
    # Self-delete so no stopped VM or orphaned boot disk lingers (and the name is
    # free to reuse); fall back to power-off if the delete call fails.
    local name zone
    name="$(meta_root name)"
    zone="$(meta_root zone | awk -F/ '{print $NF}')"
    gcloud compute instances delete "$name" --zone="$zone" -q || sudo poweroff
  fi
}
trap 'finish FAILED' ERR

# --- resolve a torch-capable interpreter on GPU images (path varies by image) --
if [[ "$(meta install-nvidia-driver)" == "True" ]]; then
  for cand in /opt/conda/bin/python /opt/conda/envs/*/bin/python /opt/venv/bin/python \
              $(ls /opt/*/bin/python3 2>/dev/null) /usr/bin/python3; do
    if [[ -x "$cand" ]] && "$cand" -c "import torch" >/dev/null 2>&1; then PY="$cand"; break; fi
  done
  # Expose the resolved interpreter as both `python` and `python3` on PATH, so the
  # job's runcmd works regardless of how that image names it.
  mkdir -p /tmp/pybin && ln -sf "$PY" /tmp/pybin/python && ln -sf "$PY" /tmp/pybin/python3
  export PATH="/tmp/pybin:$PATH"
  echo "resolved torch interpreter: $PY"
fi

# --- python deps (skipped when PIPDEPS empty, e.g. GPU image already has them) --
if [[ -n "${PIPDEPS// }" ]]; then
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-pip libgomp1 || true
  # Newer Debian marks system Python as externally managed (PEP 668) and needs
  # --break-system-packages; older pip (e.g. the GPU image) doesn't support the
  # flag, so only add it when this pip understands it.
  PIPFLAGS=""
  "$PY" -m pip install --help 2>/dev/null | grep -q -- "--break-system-packages" \
    && PIPFLAGS="--break-system-packages"
  "$PY" -m pip install --quiet $PIPFLAGS $PIPDEPS
fi

# --- layout matching config.py (ROOT/src, data/processed, amex-default-prediction) --
sudo mkdir -p "$WORK" && sudo chown -R "$(whoami)" "$WORK"
mkdir -p "$WORK/src" "$WORK/data/processed" "$WORK/amex-default-prediction" \
         "$WORK/outputs/models"

# --- pull code + the requested inputs from GCS ----------------------------
gcloud storage cp "$BUCKET/src/*.py" "$WORK/src/"
for rel in $INPUTS; do
  case "$rel" in
    data/*.csv) dest="$WORK/amex-default-prediction/$(basename "$rel")" ;;
    *)          dest="$WORK/$rel" ;;
  esac
  mkdir -p "$(dirname "$dest")"
  gcloud storage cp "$BUCKET/$rel" "$dest"
done

# --- wait for the GPU driver (Deep-Learning image installs it at boot) ------
if [[ "$(meta install-nvidia-driver)" == "True" ]]; then
  echo "waiting for NVIDIA driver ..."
  for _ in $(seq 1 90); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
  nvidia-smi || echo "WARN: GPU not ready after wait"
fi

# --- run the job -----------------------------------------------------------
cd "$WORK/src"
echo "=== job '$JOB' start $(date -u): $RUNCMD ==="
eval "$RUNCMD"
echo "=== job done $(date -u) ==="

# --- push results (everything under outputs/ + OOF) ------------------------
gcloud storage cp -r "$WORK/outputs/"* "$RESULTS/" || true
# OOF parquet(s) live under data/processed, not outputs (oof_predictions.parquet,
# oof_xgb.parquet, ...). Push them all so blending has every model's OOF.
gcloud storage cp "$WORK/data/processed/"oof*.parquet "$RESULTS/" || true

finish SUCCESS
