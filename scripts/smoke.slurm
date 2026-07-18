#!/bin/bash
#SBATCH --job-name=adf-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=%x-%j.out
set -e
cd "$HOME/adf"
CT=/opt/containers/tensorflow-25.02.sif
apptainer exec --nv "$CT" bash -c '
set -e
cd "$HOME/adf"
PYLIBS="$HOME/adf/.pylibs"
export PYTHONPATH="$PYLIBS:$PWD"
echo "=== deps (first run compiles; cached after) ==="
python3 -m pip install -q --target="$PYLIBS" librosa soundfile scipy matplotlib 2>&1 | tail -3
python3 -c "import librosa, soundfile, scipy, matplotlib" && echo deps-ok || { echo DEPS-FAIL; exit 1; }
echo "=== generate smoke data ==="
python3 scripts/make_smoke_data.py "$HOME/adf/data/smoke"
set +e
echo "=== CNN / lfcc ==="
python3 -m src.detection.train --data-root "$HOME/adf/data/smoke" --model cnn --feat lfcc   --limit 40 --epochs 2 --batch-size 8 --out reports/smoke_cnn_lfcc   && echo "OK cnn lfcc"   || echo "FAILED cnn lfcc"
echo "=== CNN / logmel ==="
python3 -m src.detection.train --data-root "$HOME/adf/data/smoke" --model cnn --feat logmel --limit 40 --epochs 2 --batch-size 8 --out reports/smoke_cnn_logmel && echo "OK cnn logmel" || echo "FAILED cnn logmel"
echo "=== RawNet2 / raw ==="
python3 -m src.detection.train --data-root "$HOME/adf/data/smoke" --model rawnet2          --limit 40 --epochs 2 --batch-size 8 --out reports/smoke_rawnet2     && echo "OK rawnet2"    || echo "FAILED rawnet2"
'
echo SMOKE-DONE
