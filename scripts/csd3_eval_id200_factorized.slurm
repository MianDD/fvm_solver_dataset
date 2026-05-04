#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -p cclake
#SBATCH -J fvm_eval_id200_factor
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd ~/rds/hpc-work/fvm_solver_dataset
mkdir -p logs eval
source .venv/bin/activate

python -m ml.evaluate \
  --grid datasets/grid_fixed_id200_csd3 \
  --ckpt checkpoints/fixed_id200_factorized/best_model.pt \
  --out eval/fixed_id200_factorized \
  --context 4 \
  --horizon 1 \
  --device cpu \
  --batch 4 \
  --num-workers 4 \
  --rollout-steps 4,8,16
