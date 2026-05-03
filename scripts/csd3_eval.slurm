#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -p cclake
#SBATCH -J fvm_eval
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd ~/rds/hpc-work/fvm_solver_dataset
mkdir -p logs
source .venv/bin/activate

python -m ml.evaluate \
  --grid datasets/grid_family_v1 \
  --ckpt checkpoints/family_v1_gpu/best_model.pt \
  --out eval/family_v1_gpu \
  --device cpu \
  --batch 4
