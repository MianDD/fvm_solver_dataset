#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -p cclake
#SBATCH -J fvm_fixed_ood_mild
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd ~/rds/hpc-work/fvm_solver_dataset
mkdir -p logs
source .venv/bin/activate

python -m sweep.sweep_fvm \
  --out datasets/raw_fixed_ood_mild_csd3 \
  --n 20 \
  --family ood_mild \
  --geometry-mode fixed_ellipse \
  --device cpu \
  --n-iter 500 \
  --save-t 0.01 \
  --dt 5e-4 \
  --min-A 0.2 \
  --max-A 0.4 \
  --lnscale 3 \
  --max-mesh-retries 2 \
  --mesh-attempt-timeout-s 30 \
  --timeout-s 900 \
  --min-snapshots 2

python -m sweep.summarize_dataset --sweep datasets/raw_fixed_ood_mild_csd3 --write-json
