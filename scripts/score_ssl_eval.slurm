#!/bin/bash
#SBATCH --job-name=adf-ssleval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
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
python3 -m src.detection.score --mode ssl --model reports/ssl_xlsr_run1/model.keras \
  --ssl-manifest "$HOME/adf/features/ssl_xlsr/eval/manifest.jsonl" \
  --asv-scores "$ASV" --out reports/ssl_xlsr_eval
'
echo SSL-EVAL-DONE
