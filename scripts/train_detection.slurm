#!/bin/bash
# SUPERSEDED by full_baseline.slurm (which passes --model and --asv-scores,
# so it emits min_tdcf). Kept for provenance.
#SBATCH --job-name=adf-detect
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#
# Full detection-baseline training on the L4 via Slurm.
#   DATA=/datasets/ASVspoof2019/LA OUT=reports/run1 sbatch scripts/train_detection.slurm
# Adjust DATA to wherever ASVspoof 2019 LA actually lives (see: ls /datasets).

set -euo pipefail
CONTAINER="${CONTAINER:-/opt/containers/tensorflow-25.02.sif}"
DATA="${DATA:-/datasets/ASVspoof2019/LA}"
OUT="${OUT:-reports/run_${SLURM_JOB_ID:-manual}}"
FEAT="${FEAT:-lfcc}"

echo "container=$CONTAINER data=$DATA out=$OUT feat=$FEAT"
apptainer exec --nv "$CONTAINER" python -m src.detection.train \
    --data-root "$DATA" --feat "$FEAT" --epochs 30 --batch-size 32 --out "$OUT"
