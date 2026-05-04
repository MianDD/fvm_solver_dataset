#!/bin/bash
# Optional experiment: factorized attention with a deeper encoder.
# Run only after the 4-layer ID200 baseline has completed successfully.
#SBATCH -A YOUR_PROJECT
#SBATCH -p cclake
#SBATCH -J fvm_id200_factor_l6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd ~/rds/hpc-work/fvm_solver_dataset
mkdir -p logs checkpoints
source .venv/bin/activate

python -m ml.train \
  --grid datasets/grid_fixed_id200_csd3 \
  --out checkpoints/fixed_id200_factorized_layers6_optional \
  --epochs 10 \
  --batch 4 \
  --context 4 \
  --horizon 1 \
  --device cpu \
  --num-workers 4 \
  --d-model 128 \
  --heads 4 \
  --layers 6 \
  --patch 8 \
  --prediction-mode derivative \
  --integrator euler \
  --use-derivatives \
  --derivative-mode central \
  --strides 1,2,4 \
  --attention-type factorized
