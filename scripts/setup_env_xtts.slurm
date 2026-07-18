#!/bin/bash
#SBATCH --job-name=adf-xttsenv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
set -e
cd "$HOME/adf"
CT=/opt/containers/pytorch-25.04.sif
apptainer exec --nv "$CT" bash -c '
set -e
cd "$HOME/adf"
source scripts/lab_env_xtts.sh
echo "=== building isolated XTTS user-site: $PYTHONUSERBASE ==="
python3 -m pip install --user -q "transformers>=4.57,<5" coqui-tts==0.27.5 2>&1 \
  | grep -viE "^DEPRECATION|egg is deprecated|already satisfied" | tail -12
# CPU build: NVIDIA torch is CUDA 12.9 and no cu129 torchaudio wheel exists for 2.7.0.
# torchaudio is only used for audio I/O here; the model still runs on GPU via torch.
python3 -m pip install --user --no-deps -q \
    --index-url https://download.pytorch.org/whl/cpu torchaudio==2.7.0 2>&1 | tail -3
echo
echo "=== verify ==="
python3 - <<PY
import torch, numpy, transformers, torchaudio
print("torch       ", torch.__version__, "| cuda", torch.cuda.is_available())
assert "nv" in torch.__version__, "NVIDIA torch build was replaced!"
print("numpy       ", numpy.__version__)
print("transformers", transformers.__version__)
print("torchaudio  ", torchaudio.__version__)
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
print("XTTS IMPORT OK")
x = torch.randn(512, 512, device="cuda"); print("gpu matmul ok:", bool((x@x).sum().isfinite()))
PY
'
echo XTTSENV-DONE
