#!/bin/bash
#SBATCH --job-name=adf-rawnet2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
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
ASV="$ROOT/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
python3 -m src.detection.train --data-root "$ROOT" --model rawnet2 --epochs 25 --batch-size 32 \
  --out reports/base_rawnet2_full --asv-scores "$ASV"
'
echo RAWNET2-DONE
