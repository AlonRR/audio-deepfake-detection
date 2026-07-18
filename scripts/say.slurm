#!/bin/bash
#SBATCH --job-name=adf-say
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#
# Speak arbitrary text in the cloned voice, using the FINE-TUNED XTTS checkpoint.
#
#   TEXT="whatever you want to say" sbatch scripts/say.slurm
#   TEXT_FILE=lines.txt NAME=demo OUT=data/generated/demo sbatch scripts/say.slurm
#
# Text is passed through the environment rather than built into a command string -
# an earlier version used eval and broke on the first quote in the sentence.
set -e
cd "$HOME/adf"
OUT=${OUT:-data/generated/on_demand}
NAME=${NAME:-say}
TEXT=${TEXT:-}
TEXT_FILE=${TEXT_FILE:-}
[ -z "$TEXT" ] && [ -z "$TEXT_FILE" ] && { echo "set TEXT=... or TEXT_FILE=..."; exit 1; }
mkdir -p "$OUT"
CT=/opt/containers/pytorch-25.04.sif
apptainer exec --nv \
  --env OUT="$OUT" --env NAME="$NAME" --env TEXT="$TEXT" --env TEXT_FILE="$TEXT_FILE" \
  "$CT" bash -c '
set -e
cd "$HOME/adf"
source scripts/lab_env_xtts.sh
if [ -n "$TEXT_FILE" ]; then
  python3 -m src.creation.xtts_finetune.synthesize --mode finetuned \
      --ckpt-dir models/xtts_ft/infer --out "$OUT" --name "$NAME" --text-file "$TEXT_FILE"
else
  python3 -m src.creation.xtts_finetune.synthesize --mode finetuned \
      --ckpt-dir models/xtts_ft/infer --out "$OUT" --name "$NAME" --text "$TEXT"
fi
'
echo "wrote $(ls "$OUT"/*.wav 2>/dev/null | wc -l) clip(s) -> $OUT"
echo SAY-DONE
