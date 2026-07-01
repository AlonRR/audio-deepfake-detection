#!/bin/bash
#SBATCH --job-name=adf-ssl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -e
cd "$HOME/adf"
SUBSET=${SUBSET:-train}
CT=/opt/containers/pytorch-25.04.sif
apptainer exec --nv --env SUBSET="$SUBSET" "$CT" bash -c '
set -e
cd "$HOME/adf"
source scripts/lab_env_pt.sh
export HF_HOME="$HOME/.cache/huggingface"
OUT="$HOME/adf/features/ssl_xlsr/$SUBSET"
python3 -m src.detection.features_ssl --data-root "$HOME/data/LA" --subset "$SUBSET" --out "$OUT" --frontend facebook/wav2vec2-xls-r-300m --layer avg --max-frames 200
'
echo SSL-EXTRACT-DONE-$SUBSET
