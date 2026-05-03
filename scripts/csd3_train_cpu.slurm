#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -p cclake
#SBATCH -J fvm_train_cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd ~/rds/hpc-work/fvm_solver_dataset
mkdir -p logs
source .venv/bin/activate

python -m ml.train \
  --grid datasets/grid_family_v1 \
  --out checkpoints/family_v1_cpu \
  --epochs 50 \
  --batch 4 \
  --context 4 \
  --horizon 1 \
  --device cpu \
  --num-workers 4 \
  --d-model 128 \
  --heads 4 \
  --layers 4 \
  --patch 8

python -m ml.evaluate \
  --grid datasets/grid_family_v1 \
  --ckpt checkpoints/family_v1_cpu/best_model.pt \
  --out eval/family_v1_cpu \
  --device cpu \
  --batch 4
