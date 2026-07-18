#!/bin/bash
#SBATCH --job-name=adf-xtts-ft-syn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#
# Synthesize with the FINE-TUNED XTTS. Split out from synth_xtts.slurm because
# job 511 was OOM-killed here: the trainer checkpoint (best_model.pth) is 5.6GB
# since it carries optimizer state, and loading it exceeded the job memory.
# We strip it to model weights only (~1.9GB) and ask for more RAM.
set -e
cd "$HOME/adf"
RUN=${RUN:?set RUN=models/xtts_ft/<run-subdir>}
INFER=models/xtts_ft/infer
CT=/opt/containers/pytorch-25.04.sif
apptainer exec --nv --env RUN="$RUN" --env INFER="$INFER" "$CT" bash -c '
set -e
cd "$HOME/adf"
source scripts/lab_env_xtts.sh
echo "=== strip optimizer state from checkpoint ==="
python3 - <<PY
import torch, os
run, infer = os.environ["RUN"], os.environ["INFER"]
src = os.path.join(run, "best_model.pth")
ck = torch.load(src, map_location="cpu", weights_only=False)
print("checkpoint keys:", list(ck.keys())[:8])
slim = {"model": ck["model"]} if "model" in ck else ck
dst = os.path.join(infer, "model.pth")
torch.save(slim, dst)
print(f"{src} {os.path.getsize(src)/2**30:.2f} GiB -> {dst} {os.path.getsize(dst)/2**30:.2f} GiB")
PY
echo "=== A2 fine-tuned synthesis ==="
python3 -m src.creation.xtts_finetune.synthesize --mode finetuned \
    --ckpt-dir "$INFER" --out data/generated/xtts_ft
'
echo "finetuned clips: $(ls data/generated/xtts_ft/*.wav 2>/dev/null | wc -l)"
echo XTTS-FT-SYN-DONE
