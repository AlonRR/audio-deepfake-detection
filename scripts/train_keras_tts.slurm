#!/bin/bash
#SBATCH --job-name=adf-a1-tts
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#
# A1: from-scratch Keras Tacotron2-lite on the target voice (TensorFlow container).
# This baseline is EXPECTED to underperform - ~4 min of speech is far too little
# for a seq2seq TTS trained from zero, and Griffin-Lim adds its own artifacts.
# The A1-vs-A2 gap is the deliverable, not a bug to fix.
#
#   EPOCHS=300 sbatch scripts/train_keras_tts.slurm
set -e
cd "$HOME/adf"
EPOCHS=${EPOCHS:-300}
OUT=${OUT:-reports/tts_baseline}
CT=/opt/containers/tensorflow-25.02.sif
apptainer exec --nv --env EPOCHS="$EPOCHS" --env OUT="$OUT" "$CT" bash -c '
set -e
cd "$HOME/adf"
source scripts/lab_env.sh
python3 -m src.creation.keras_tts.train --data-root data/raw --epochs "$EPOCHS" --out "$OUT"
W=$(ls "$OUT"/*.weights.h5 2>/dev/null | head -1)
echo "weights: $W"
python3 -m src.creation.keras_tts.synthesize --weights "$W" --out data/generated/keras_tts
'
echo "a1 clips: $(ls data/generated/keras_tts/*.wav 2>/dev/null | wc -l)"
echo A1-DONE
