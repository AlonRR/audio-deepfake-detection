#!/bin/bash
#SBATCH --job-name=adf-cnneval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
set -e
cd "$HOME/adf"
CT=/opt/containers/tensorflow-25.02.sif
apptainer exec --nv "$CT" bash -c '
set -e
cd "$HOME/adf"
source scripts/lab_env.sh
ROOT="$HOME/data/LA"
ASV="$ROOT/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
python3 -m src.detection.score --mode cnn --model reports/base_cnn_lfcc_full/model.keras \
  --data-root "$ROOT" --subset eval --feat lfcc \
  --asv-scores "$ASV" --out reports/base_cnn_lfcc_eval
'
echo CNN-EVAL-DONE
