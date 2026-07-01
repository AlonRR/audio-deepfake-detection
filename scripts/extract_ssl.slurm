#!/bin/bash
#SBATCH --job-name=adf-ssl-extract
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
#
# Cache frozen SSL features for all three ASVspoof subsets (PyTorch container).
#   DATA=/datasets/ASVspoof2019/LA OUT=/models/$USER/ssl_xlsr sbatch scripts/extract_ssl.slurm
set -euo pipefail
CONTAINER="${CONTAINER:-/opt/containers/pytorch-25.04.sif}"
DATA="${DATA:-/datasets/ASVspoof2019/LA}"
OUT="${OUT:-/models/$USER/ssl_xlsr}"
FRONTEND="${FRONTEND:-facebook/wav2vec2-xls-r-300m}"

for SUBSET in train dev eval; do
  echo ">> extracting $SUBSET"
  apptainer exec --nv "$CONTAINER" python -m src.detection.features_ssl \
      --data-root "$DATA" --subset "$SUBSET" --out "$OUT/$SUBSET" --frontend "$FRONTEND"
done
