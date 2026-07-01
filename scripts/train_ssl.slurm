#!/bin/bash
#SBATCH --job-name=adf-ssl-train
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#
# Train the Keras back-end on cached SSL features (TensorFlow container).
# Run AFTER extract_ssl.slurm has produced the manifests.
#   FEATS=/models/$USER/ssl_xlsr OUT=reports/ssl_run1 sbatch scripts/train_ssl.slurm
set -euo pipefail
CONTAINER="${CONTAINER:-/opt/containers/tensorflow-25.02.sif}"
FEATS="${FEATS:-/models/$USER/ssl_xlsr}"
OUT="${OUT:-reports/ssl_run_${SLURM_JOB_ID:-manual}}"

apptainer exec --nv "$CONTAINER" python -m src.detection.train_ssl \
    --train "$FEATS/train/manifest.jsonl" --dev "$FEATS/dev/manifest.jsonl" \
    --epochs 40 --batch-size 64 --out "$OUT"
