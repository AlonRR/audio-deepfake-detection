#!/bin/bash
#SBATCH --job-name=adf-a1-par
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
set -e
cd "$HOME/adf"
OUT=${OUT:-reports/tts_baseline}
CT=/opt/containers/tensorflow-25.02.sif
apptainer exec --nv --env OUT="$OUT" "$CT" bash -c '
set -e
cd "$HOME/adf"
source scripts/lab_env.sh
W=$(ls "$OUT"/*.weights.h5 2>/dev/null | head -1)
echo "weights: $W"
python3 -m src.creation.keras_tts.synthesize --weights "$W" \
    --texts-csv data/raw/metadata_eval.csv --out data/generated/keras_tts_parallel
'
echo "a1_par: $(ls data/generated/keras_tts_parallel/*.wav 2>/dev/null | wc -l)"
echo A1-PAR-DONE
