#!/usr/bin/env bash
# End-to-end: from a clean clone, produce everything needed for Report 1.
#
# Run this with:
#     bash scripts/make_report1_artifacts.sh
#
# It is idempotent: re-running it skips steps that have already produced
# their outputs.

set -e
cd "$(dirname "$0")/.."

# --- 1. Patch the supervisor's solver for Python 3.10+ -----------------
echo "[1/5] Applying compatibility patch"
python scripts/patch_fvm_solver.py .

# --- 2. Run a small in-distribution sweep -------------------------------
if [ ! -d "datasets/family_v1" ] || [ -z "$(ls datasets/family_v1/sim_*/t_*.npz 2>/dev/null)" ]; then
    echo "[2/5] Sweeping in-distribution PDE family (N=8 sims, ~5 min on CPU)"
    python -m sweep.sweep_fvm --out datasets/family_v1 --n 8 \
        --n-iter 1500 --end-t 1.0 --save-t 0.05 --dt 5e-4 \
        --lnscale 4 --min-A 2e-3 --max-A 4e-3 --device cpu
else
    echo "[2/5] datasets/family_v1 already exists — skipping"
fi

# --- 3. Run a small OOD sweep ------------------------------------------
if [ ! -d "datasets/ood" ] || [ -z "$(ls datasets/ood/sim_*/t_*.npz 2>/dev/null)" ]; then
    echo "[3/5] Sweeping OOD PDE family (N=4 sims)"
    python -m sweep.sweep_fvm --out datasets/ood --n 4 --ood \
        --n-iter 1500 --end-t 1.0 --save-t 0.05 --dt 5e-4 \
        --lnscale 4 --min-A 2e-3 --max-A 4e-3 --device cpu
else
    echo "[3/5] datasets/ood already exists — skipping"
fi

# --- 4. Convert unstructured snapshots --> regular grids ----------------
echo "[4/5] Packing onto regular grids"
python -m ml.grid_adapter --sweep datasets/family_v1 --out datasets/grid_main --H 64 --W 96
python -m ml.grid_adapter --sweep datasets/ood       --out datasets/grid_ood  --H 64 --W 96

# --- 5. Train the foundation model -------------------------------------
echo "[5/5] Training the foundation model"
python -m ml.train --grid datasets/grid_main --out checkpoints/run0 \
    --epochs 20 --batch 4 --d-model 96 --patch 8 --device cpu

echo
echo "DONE. Inspect outputs in:"
echo "  datasets/family_v1/, datasets/ood/  (raw FVM output)"
echo "  datasets/grid_main/,  datasets/grid_ood/  (regular-grid)"
echo "  checkpoints/run0/                    (trained model + history)"
