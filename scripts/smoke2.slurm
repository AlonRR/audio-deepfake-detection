#!/bin/bash
#SBATCH --job-name=adf-smoke2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
set -e
cd "$HOME/adf"
CT=/opt/containers/tensorflow-25.02.sif
apptainer exec --nv "$CT" bash -c '
set -e
cd "$HOME/adf"
export PYTHONUSERBASE="$HOME/adf/.pyuser"
US="$HOME/adf/.pyuser/lib/python3.12/site-packages"
export PYTHONPATH="$US:$PWD"
echo "=== install missing deps (pinned numpy, --user respects container numpy) ==="
python3 -m pip install --user -q -c constraints.txt librosa soundfile matplotlib audioread soxr 2>&1 | tail -3
python3 -c "import numpy, librosa, soundfile, matplotlib, scipy, numba" && echo deps-ok || { echo DEPS-FAIL; exit 1; }
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
echo SMOKE2-DONE
