#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -p cclake
#SBATCH -J fvm_plot_id200_factor
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd ~/rds/hpc-work/fvm_solver_dataset
mkdir -p logs figures/predictions
source .venv/bin/activate

python -m ml.plot_predictions \
  --grid datasets/grid_fixed_id200_csd3 \
  --ckpt checkpoints/fixed_id200_factorized/best_model.pt \
  --out figures/predictions/fixed_id200_factorized \
  --context 4 \
  --horizon 1 \
  --stride 1 \
  --device cpu \
  --num-examples 4
