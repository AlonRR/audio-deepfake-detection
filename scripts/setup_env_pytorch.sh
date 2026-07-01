#!/bin/bash
# Install the PyTorch-side deps (SSL frontend, XTTS, ECAPA, Whisper) into ~/.local, visible
# inside the PyTorch apptainer container. torch/torchaudio are already in the container.
#
# Usage:  bash scripts/setup_env_pytorch.sh [/opt/containers/pytorch-25.04.sif]
set -euo pipefail
CONTAINER="${1:-/opt/containers/pytorch-25.04.sif}"

echo ">> using container: $CONTAINER"
apptainer exec --nv "$CONTAINER" python -m pip install --user --no-warn-script-location \
    "numpy>=1.26" "scipy>=1.11" "librosa>=0.10" "soundfile>=0.12" "matplotlib>=3.8" \
    "transformers>=4.44" "speechbrain>=1.0" "coqui-tts>=0.24" "faster-whisper>=1.0"

echo ">> sanity check:"
apptainer exec --nv "$CONTAINER" python - <<'PY'
import torch, transformers  # noqa
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers:", transformers.__version__)
PY
echo ">> ok — PyTorch-side deps in ~/.local"
