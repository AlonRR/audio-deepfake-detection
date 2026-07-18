#!/bin/bash
#SBATCH --job-name=adf-deps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out
cd "$HOME/adf"
rm -rf "$HOME/adf/.pylibs"
apptainer exec /opt/containers/tensorflow-25.02.sif bash -c 'cd "$HOME/adf" && python3 -m site | head -8 && echo ---- && python3 scripts/check_deps.py'
echo DEPS-CHECK-DONE
