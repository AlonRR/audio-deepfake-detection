#!/bin/bash
#SBATCH --job-name=adf-synth-par
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#
# Parallel-text synthesis: speak the HELD-OUT transcripts with every system so
# MCD / log-mel SSIM compare identical content against the real held-out audio.
# Non-parallel MCD (synth "hello." vs real "Somewhere beyond the ridge...") is
# meaningless, which is why this second pass exists alongside the PROMPTS clips.
set -e
cd "$HOME/adf"
EVAL=data/raw/metadata_eval.csv
INFER=models/xtts_ft/infer
CT=/opt/containers/pytorch-25.04.sif
apptainer exec --nv --env EVAL="$EVAL" --env INFER="$INFER" "$CT" bash -c '
set -e
cd "$HOME/adf"
source scripts/lab_env_xtts.sh
echo "=== zero-shot (parallel) ==="
python3 -m src.creation.xtts_finetune.synthesize --mode zero_shot \
    --texts-csv "$EVAL" --out data/generated/xtts_zeroshot_parallel
echo "=== fine-tuned (parallel) ==="
python3 -m src.creation.xtts_finetune.synthesize --mode finetuned \
    --ckpt-dir "$INFER" --texts-csv "$EVAL" --out data/generated/xtts_ft_parallel
'
echo "zs_par: $(ls data/generated/xtts_zeroshot_parallel/*.wav 2>/dev/null | wc -l)  ft_par: $(ls data/generated/xtts_ft_parallel/*.wav 2>/dev/null | wc -l)"
echo SYNTH-PAR-DONE
