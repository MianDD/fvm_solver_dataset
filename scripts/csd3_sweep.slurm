#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -p cclake
#SBATCH -J fvm_sweep
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd ~/rds/hpc-work/fvm_solver_dataset
mkdir -p logs
source .venv/bin/activate

python -m sweep.sweep_fvm \
  --out datasets/raw_family_v1 \
  --n 64 \
  --family id \
  --device cpu \
  --n-iter 2500 \
  --save-t 0.05 \
  --dt 5e-4 \
  --min-A 0.02 \
  --max-A 0.04 \
  --lnscale 4 \
  --max-mesh-retries 3 \
  --mesh-attempt-timeout-s 60 \
  --timeout-s 3600 \
  --min-snapshots 8

python -m sweep.summarize_dataset --sweep datasets/raw_family_v1 --write-json
