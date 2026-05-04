#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -p cclake
#SBATCH -J fvm_id20_factor
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd ~/rds/hpc-work/fvm_solver_dataset
mkdir -p logs checkpoints
source .venv/bin/activate

python -m ml.train \
  --grid datasets/grid_fixed_id20_csd3 \
  --out checkpoints/fixed_id20_factorized_cmp \
  --epochs 5 \
  --batch 4 \
  --context 4 \
  --horizon 1 \
  --device cpu \
  --num-workers 4 \
  --d-model 128 \
  --heads 4 \
  --layers 4 \
  --patch 8 \
  --prediction-mode derivative \
  --integrator euler \
  --use-derivatives \
  --derivative-mode central \
  --strides 1,2,4 \
  --attention-type factorized
