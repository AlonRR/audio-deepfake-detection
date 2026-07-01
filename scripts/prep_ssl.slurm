#!/bin/bash
#SBATCH --job-name=adf-sslprep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
set -e
cd "$HOME/adf"
apptainer exec --nv /opt/containers/pytorch-25.04.sif bash -c '
set -e
cd "$HOME/adf"
source scripts/lab_env_pt.sh
export HF_HOME="$HOME/.cache/huggingface"
python3 -c "import transformers, torch; print(\"transformers\", transformers.__version__, \"torch\", torch.__version__)"
python3 -c "from transformers import AutoModel; m=AutoModel.from_pretrained(\"facebook/wav2vec2-xls-r-300m\"); print(\"XLS-R hidden_size\", m.config.hidden_size)"
'
echo SSL-PREP-DONE
