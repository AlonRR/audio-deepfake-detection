#!/bin/bash
# SSL back-end hyper-parameter search on cached XLS-R features (TF container).
# Sweeps learning rate (+ two architecture variants) and includes a DELIBERATE failed
# run (lr=1e-1) for the "show the process / failed run" rubric item. Each config trains
# fast on cached features, so the whole sweep fits one 2h job.
#SBATCH --job-name=adf-sslhp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
set -u
cd "$HOME/adf"
CT=/opt/containers/tensorflow-25.02.sif
apptainer exec --nv "$CT" bash -c '
set -u
cd "$HOME/adf"
source scripts/lab_env.sh
ROOT="$HOME/data/LA"
ASV="$ROOT/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
TR="$HOME/adf/features/ssl_xlsr/train/manifest.jsonl"
DV="$HOME/adf/features/ssl_xlsr/dev/manifest.jsonl"
# tag  lr  proj  dropout   (lr1e1_FAIL is the intentional failure)
CONFIGS="
lr1e4 0.0001 256 0.3
lr3e4 0.0003 256 0.3
lr1e3 0.001 256 0.3
lr3e3 0.003 256 0.3
lr1e2 0.01 256 0.3
lr1e1_FAIL 0.1 256 0.3
proj128 0.001 128 0.3
do0p5 0.001 256 0.5
"
echo "$CONFIGS" | while read tag lr proj dropout; do
  [ -z "$tag" ] && continue
  OUT="reports/ssl_hp_$tag"
  if [ -f "$OUT/result.json" ]; then echo "skip $tag (already done)"; continue; fi
  # A deliberate-failure config never writes result.json; its history.csv IS the
  # deliverable. Skip it on resume so a re-run cannot overwrite the divergence evidence.
  case "$tag" in *FAIL*)
    if [ -f "$OUT/history.csv" ]; then echo "skip $tag (failure evidence captured)"; continue; fi;;
  esac
  echo "===== HP $tag  lr=$lr proj=$proj dropout=$dropout ====="
  python3 -m src.detection.train_ssl --train "$TR" --dev "$DV" \
    --epochs 25 --batch-size 64 --lr "$lr" --proj "$proj" --dropout "$dropout" \
    --out "$OUT" --asv-scores "$ASV" || echo "HP-$tag-FAILED"
done
'
echo SSL-HP-DONE
