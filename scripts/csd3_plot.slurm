#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -p cclake
#SBATCH -J fvm_plot
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd ~/rds/hpc-work/fvm_solver_dataset
mkdir -p logs
source .venv/bin/activate

python -m ml.plot_predictions \
  --grid datasets/grid_family_v1 \
  --ckpt checkpoints/family_v1_gpu/best_model.pt \
  --out figures/predictions/family_v1_gpu \
  --device cpu \
  --num-examples 4
