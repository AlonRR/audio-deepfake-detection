#!/bin/bash
#SBATCH --job-name=adf-cross
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#
# Cross-generator test: does a detector trained on ASVspoof 2019 LA (attacks from
# 2019-era TTS/VC) still catch clips from a 2025 synthesizer it has never seen?
# Scored on OUR OWN clips: held-out real speech vs A1/A2 fakes.
set -e
cd "$HOME/adf"

# Materialise the 4 held-out real clips as a folder (bona fide reference).
HELD=data/raw/heldout
mkdir -p "$HELD"
cut -d"|" -f1 data/raw/metadata_eval.csv | while read -r cid; do
  [ -f "data/raw/wavs/$cid.wav" ] && cp -f "data/raw/wavs/$cid.wav" "$HELD/"
done
echo "held-out real clips: $(ls $HELD/*.wav 2>/dev/null | wc -l)"
for d in data/generated/keras_tts data/generated/xtts_zeroshot data/generated/xtts_ft; do
  echo "  $d: $(ls $d/*.wav 2>/dev/null | wc -l) clips"
done

PT=/opt/containers/pytorch-25.04.sif
TF=/opt/containers/tensorflow-25.02.sif

# --- stage 1: cache SSL features for our own clips (PyTorch container) --------
# Frontend/layer/max-frames MUST match the trained back-end or features are not comparable.
apptainer exec --nv "$PT" bash -c '
set -e
cd "$HOME/adf"
source scripts/lab_env_pt.sh
export HF_HOME="$HOME/.cache/huggingface"
python3 -m src.detection.features_ssl \
    --wav-dirs 1:data/raw/heldout 0:data/generated/keras_tts \
               0:data/generated/xtts_zeroshot 0:data/generated/xtts_ft \
    --out features/ssl_own --frontend facebook/wav2vec2-xls-r-300m \
    --layer avg --max-frames 200
'

# --- stage 2: score with both detectors (TensorFlow container) ---------------
apptainer exec --nv "$TF" bash -c '
set -e
cd "$HOME/adf"
source scripts/lab_env.sh
echo "=== SSL detector (main system) vs own fakes ==="
python3 -m src.detection.cross_test --model reports/ssl_xlsr_run1/model.keras \
    --model-type ssl --ssl-manifest features/ssl_own/manifest.jsonl \
    --out reports/cross_ssl
echo "=== CNN-LFCC baseline vs own fakes ==="
python3 -m src.detection.cross_test --model reports/base_cnn_lfcc_full/model.keras \
    --model-type cnn --feat lfcc \
    --fakes data/generated/xtts_ft --real data/raw/heldout \
    --out reports/cross_cnn
'
echo CROSS-DONE
