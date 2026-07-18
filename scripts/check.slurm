#!/bin/bash
#SBATCH --job-name=adf-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out
set -e
cd "$HOME/adf"
echo "node=$(hostname)"
command -v apptainer || { echo NO-APPTAINER; exit 1; }
apptainer exec --nv /opt/containers/tensorflow-25.02.sif bash -c 'cd "$HOME/adf" && export PYTHONPATH=. && python3 scripts/_check.py && python3 tests/test_metrics.py'
echo CHECK-DONE
