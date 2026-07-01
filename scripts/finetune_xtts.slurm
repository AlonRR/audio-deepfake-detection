#!/bin/bash
#SBATCH --job-name=adf-xtts-ft
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#
# Fine-tune XTTS-v2 on the target speaker (PyTorch container).
# Run AFTER prepare.py has built data/raw/{metadata.csv,wavs}.
#   DATA=data/raw OUT=models/xtts_ft sbatch scripts/finetune_xtts.slurm
set -euo pipefail
CONTAINER="${CONTAINER:-/opt/containers/pytorch-25.04.sif}"
DATA="${DATA:-data/raw}"
OUT="${OUT:-models/xtts_ft}"

apptainer exec --nv "$CONTAINER" python -m src.creation.xtts_finetune.finetune \
    --data-root "$DATA" --out "$OUT" --epochs 10 --batch-size 3
