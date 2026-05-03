#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -p cclake
#SBATCH -J fvm_fixed_tiny
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd ~/rds/hpc-work/fvm_solver_dataset
mkdir -p logs
source .venv/bin/activate

python -m sweep.sweep_fvm \
  --out datasets/raw_fixed_tiny_csd3 \
  --n 2 \
  --family id \
  --geometry-mode fixed_ellipse \
  --device cpu \
  --n-iter 20 \
  --save-t 0.001 \
  --dt 5e-4 \
  --min-A 0.2 \
  --max-A 0.4 \
  --lnscale 3 \
  --max-mesh-retries 2 \
  --mesh-attempt-timeout-s 30 \
  --timeout-s 300 \
  --min-snapshots 1

python -m sweep.summarize_dataset --sweep datasets/raw_fixed_tiny_csd3 --write-json
