#!/bin/bash
#SBATCH --job-name=adf-xtts-ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#
# Fine-tune XTTS-v2 on the target speaker (PyTorch container).
#
# Reads data/raw/metadata.csv = the TRAIN split. The 4 held-out evaluation clips
# live in metadata_eval.csv and are excluded by construction.
#
# Requires the isolated XTTS user-site: sbatch scripts/setup_env_xtts.slurm (once).
#
#   DATA=data/raw OUT=models/xtts_ft sbatch scripts/finetune_xtts.slurm
#
# --time is 02:00:00 because the bsc QOS caps MaxWall at 2h; the original
# 08:00:00 was rejected at submit. 30 clips at batch 3 is ~10 steps/epoch.
set -e
cd "$HOME/adf"
DATA=${DATA:-data/raw}
OUT=${OUT:-models/xtts_ft}
EPOCHS=${EPOCHS:-10}
BS=${BS:-3}
CT=/opt/containers/pytorch-25.04.sif
apptainer exec --nv --env DATA="$DATA" --env OUT="$OUT" --env EPOCHS="$EPOCHS" --env BS="$BS" "$CT" bash -c '
set -e
cd "$HOME/adf"
source scripts/lab_env_xtts.sh
python3 -m src.creation.xtts_finetune.finetune \
    --data-root "$DATA" --out "$OUT" --epochs "$EPOCHS" --batch-size "$BS"
'
echo XTTS-FT-DONE
