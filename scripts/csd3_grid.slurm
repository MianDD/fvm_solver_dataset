#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -p cclake
#SBATCH -J fvm_grid
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

python -m sweep.summarize_dataset --sweep datasets/raw_family_v1 --write-json
python -m ml.grid_adapter --sweep datasets/raw_family_v1 --out datasets/grid_family_v1 --H 64 --W 96
