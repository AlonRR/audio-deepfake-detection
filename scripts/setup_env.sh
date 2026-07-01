#!/bin/bash
# Install the extra Python deps the detector needs into ~/.local (visible inside the
# apptainer container, since $HOME is bind-mounted). TensorFlow/Keras are ALREADY in the
# TensorFlow container, so we only add audio + plotting libs.
#
# Usage:  bash scripts/setup_env.sh [/opt/containers/tensorflow-25.02.sif]
set -euo pipefail
CONTAINER="${1:-/opt/containers/tensorflow-25.02.sif}"

echo ">> using container: $CONTAINER"
apptainer exec --nv "$CONTAINER" python -m pip install --user --no-warn-script-location \
    "numpy>=1.26" "scipy>=1.11" "librosa>=0.10" "soundfile>=0.12" "matplotlib>=3.8" "tqdm>=4.66"

echo ">> sanity check (GPU + imports):"
apptainer exec --nv "$CONTAINER" python - <<'PY'
import tensorflow as tf, librosa, soundfile  # noqa
print("TF:", tf.__version__, "GPUs:", tf.config.list_physical_devices("GPU"))
print("librosa:", librosa.__version__)
PY
echo ">> ok — deps in ~/.local"
