#!/bin/bash
#SBATCH --job-name=adf-base
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
EXTRA=""
[ -f "$ASV" ] && EXTRA="--asv-scores $ASV"
echo "data-root=$ROOT  asv=$ASV  extra=$EXTRA"
python3 -m src.detection.train --data-root "$ROOT" --model cnn --feat lfcc --limit 4000 --epochs 12 --batch-size 32 --out reports/base_cnn_lfcc_sub $EXTRA
'
echo BASE-DONE
