#!/bin/bash
#SBATCH -A YOUR_PROJECT
#SBATCH -p ampere
#SBATCH -J fvm_train_gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd ~/rds/hpc-work/fvm_solver_dataset
mkdir -p logs
source .venv/bin/activate

python -m ml.train \
  --grid datasets/grid_family_v1 \
  --out checkpoints/family_v1_gpu \
  --epochs 100 \
  --batch 8 \
  --context 4 \
  --horizon 1 \
  --device cuda \
  --num-workers 4 \
  --d-model 192 \
  --heads 6 \
  --layers 6 \
  --patch 8

python -m ml.evaluate \
  --grid datasets/grid_family_v1 \
  --ckpt checkpoints/family_v1_gpu/best_model.pt \
  --out eval/family_v1_gpu \
  --device cuda \
  --batch 8
