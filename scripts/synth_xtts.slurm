#!/bin/bash
#SBATCH --job-name=adf-xtts-syn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#
# Synthesize the >=10 demo clips with XTTS-v2, zero-shot AND fine-tuned.
# Both in one job: each mode loads a ~2GB model, so sharing a queue slot is cheaper.
set -e
cd "$HOME/adf"
RUN=${RUN:?set RUN=models/xtts_ft/<run-subdir>}
CT=/opt/containers/pytorch-25.04.sif

# Coqui load_checkpoint expects model.pth + vocab.json in ONE dir; the trainer run
# dir has best_model.pth and no vocab. Assemble an inference dir from both.
INFER=models/xtts_ft/infer
mkdir -p "$INFER"
cp -f "$RUN/best_model.pth"            "$INFER/model.pth"
cp -f "$RUN/config.json"               "$INFER/config.json"
cp -f models/xtts_ft/xtts_base/vocab.json "$INFER/vocab.json"
echo "assembled $INFER:"; ls -la "$INFER"

apptainer exec --nv --env INFER="$INFER" "$CT" bash -c '
set -e
cd "$HOME/adf"
source scripts/lab_env_xtts.sh
echo "=== A2 zero-shot ==="
python3 -m src.creation.xtts_finetune.synthesize --mode zero_shot \
    --out data/generated/xtts_zeroshot
echo "=== A2 fine-tuned ==="
python3 -m src.creation.xtts_finetune.synthesize --mode finetuned \
    --ckpt-dir "$INFER" --out data/generated/xtts_ft
'
echo "zeroshot: $(ls data/generated/xtts_zeroshot/*.wav 2>/dev/null | wc -l) clips"
echo "finetuned: $(ls data/generated/xtts_ft/*.wav 2>/dev/null | wc -l) clips"
echo XTTS-SYN-DONE
