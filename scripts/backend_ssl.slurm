#!/bin/bash
#SBATCH --job-name=adf-sslback
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -e
cd "$HOME/adf"
CT=/opt/containers/tensorflow-25.02.sif
apptainer exec --nv "$CT" bash -c '
set -e
cd "$HOME/adf"
source scripts/lab_env.sh
ROOT="$HOME/data/LA"
ASV="$ROOT/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
python3 -m src.detection.train_ssl --train "$HOME/adf/features/ssl_xlsr/train/manifest.jsonl" --dev "$HOME/adf/features/ssl_xlsr/dev/manifest.jsonl" --epochs 40 --batch-size 64 --out reports/ssl_xlsr_run1 --asv-scores "$ASV"
'
echo SSL-BACKEND-DONE
