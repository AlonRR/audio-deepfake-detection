#!/bin/bash
#SBATCH --job-name=adf-detect1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#
# Score ONE wav with both detectors.  FILE=/path/to.wav sbatch scripts/detect_one.slurm
# Writes JSON to $OUT/detect_one.json (default reports/detect_one).
#
# Two containers, because the SSL frontend is torch and the back-end is Keras:
# stage 1 caches features in the PyTorch container, stage 2 scores in TensorFlow.
set -e
cd "$HOME/adf"
FILE=${FILE:?set FILE=/abs/path/to.wav}
OUT=${OUT:-reports/detect_one}
mkdir -p "$OUT" "$OUT/in"
cp -f "$FILE" "$OUT/in/probe.wav"
PT=/opt/containers/pytorch-25.04.sif
TF=/opt/containers/tensorflow-25.02.sif

apptainer exec --nv --env OUT="$OUT" "$PT" bash -c '
set -e; cd "$HOME/adf"; source scripts/lab_env_pt.sh
export HF_HOME="$HOME/.cache/huggingface"
rm -rf "$OUT/feat"
python3 -m src.detection.features_ssl --wav-dirs 0:"$OUT/in" \
    --out "$OUT/feat" --frontend facebook/wav2vec2-xls-r-300m --layer avg --max-frames 200
' >/dev/null 2>&1

apptainer exec --nv --env OUT="$OUT" "$TF" bash -c '
set -e; cd "$HOME/adf"; source scripts/lab_env.sh
python3 - <<PY
import json, numpy as np, keras, os
out = os.environ["OUT"]
res = {}
# SSL back-end (headline system)
from src.detection import data_ssl as DS
e = DS.read_manifest(f"{out}/feat/manifest.jsonl")
x = np.asarray([np.load(i["npy"]).astype("float32") for i in e])
m = keras.models.load_model("reports/ssl_xlsr_run1/model.keras")
res["ssl_bonafide_prob"] = float(m(x, training=False).numpy()[0, 1])
# CNN-LFCC baseline
from src.common.audio import FeatureConfig
from src.detection.cross_test import _cnn_scores
cnn = keras.models.load_model("reports/base_cnn_lfcc_full/model.keras")
res["cnn_bonafide_prob"] = float(_cnn_scores(cnn, [f"{out}/in/probe.wav"], "lfcc", FeatureConfig())[0])
json.dump(res, open(f"{out}/detect_one.json", "w"), indent=2)
print(json.dumps(res))
PY
' 2>/dev/null | tail -5
echo DETECT-DONE
