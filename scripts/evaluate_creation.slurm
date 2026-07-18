#!/bin/bash
#SBATCH --job-name=adf-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#
# Creation-evaluation harness: MCD / log-mel SSIM on the parallel clips, ECAPA
# speaker cosine on the prompt clips, both scored against the 4 HELD-OUT real
# clips. Also builds the blind MOS test (never scores it - humans do that).
#
# Job 522 died silently here: the home quota was full, so pip could not write,
# and `set -e` aborted the chain before anything printed. ECAPA is now explicitly
# best-effort - if it cannot install, the harness still produces MCD/SSIM/MOS and
# records the speaker metric as an error rather than omitting or faking it.
set -e
cd "$HOME/adf"
CT=/opt/containers/pytorch-25.04.sif
apptainer exec --nv "$CT" bash -c '
cd "$HOME/adf"
source scripts/lab_env_xtts.sh
echo "=== ECAPA / speechbrain (best-effort) ==="
set +e
python3 -c "import speechbrain, sys; print(\"present:\", speechbrain.__version__)"
if [ $? -ne 0 ]; then
  python3 -m pip install --user speechbrain 2>&1 | tail -12
  python3 -c "import speechbrain; print(\"installed:\", speechbrain.__version__)" \
    || echo "ECAPA UNAVAILABLE - speaker cosine reported as error, not faked"
fi
set -e
echo
echo "=== held-out reference clips ==="
mkdir -p data/raw/heldout
cut -d"|" -f1 data/raw/metadata_eval.csv | while read -r cid; do
  [ -f "data/raw/wavs/$cid.wav" ] && cp -f "data/raw/wavs/$cid.wav" data/raw/heldout/
done
ls data/raw/heldout/
echo
echo "=== harness ==="
python3 -m src.evaluation.run_all --out reports/evaluation --build-mos
echo
echo "=== results table ==="
cat reports/evaluation/creation_results.md
'
echo EVAL-DONE
