# `fvm_solver_dataset` — varying-PDE dataset generation and foundation-model training

This repository extends [`Maxzhu123/fvm_solver`](https://github.com/Maxzhu123/fvm_solver) (the supervisor's compressible-Navier–Stokes finite-volume solver) into a **foundation-model pipeline for fluid simulations**: it sweeps a *family* of PDE configurations through the solver, packs the resulting unstructured snapshots onto regular grids, and trains a patch-transformer that predicts future fluid states from a context history.

The motivation, following the project brief, is that an ML surrogate trained on a *single* fixed PDE cannot generalise to fluids with non-ideal Newtonian behaviour, different gas constants, or different transport coefficients — but a model trained across a *family* of PDEs can use its observed history to identify the dynamics in-context (in the spirit of TabPFN, Poseidon, Walrus and TNT).

## Repository layout

```
.                              
├── base_cfg.py                     paths
├── mesh_gen/                       meshpy + custom triangulation
├── time_fvm/                       FVM equations, integrators, BCs, saver
├── requirements.txt
.                              -- new code added by this project:
├── scripts/
│   └── patch_fvm_solver.py         one-time compatibility fix (Python 3.10+)
├── sweep/
│   ├── sweep_fvm.py                runs the solver many times with sampled physics
│   └── __init__.py
├── ml/
│   ├── grid_adapter.py             unstructured snapshots --> regular grids
│   ├── model.py                    patch transformer w/ spatiotemporal embeddings
│   ├── dataset.py                  PyTorch Dataset over the regular-grid .npz
│   ├── train.py                    teacher-forced autoregressive training
│   └── __init__.py
├── docs/                           theory and design notes (Reports 1 & 2)
└── figures/                        plots produced by training and evaluation
```

## Quickstart

```bash
# 1. Set up environment
git clone https://github.com/MianDD/fvm_solver_dataset.git
cd fvm_solver_dataset
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install scipy                                         # used by ml/grid_adapter.py

# 2. One-off: fix forward-reference bug in supervisor's code (Python 3.10+)
python scripts/patch_fvm_solver.py

# 3. Generate a family of CFD simulations (~30 min on CPU; one process per sim)
python -m sweep.sweep_fvm --out datasets/family_v1 --n 24
python -m sweep.sweep_fvm --out datasets/ood       --n  6 --ood

# 4. Pack onto regular grids for the ML model
python -m ml.grid_adapter --sweep datasets/family_v1 --out datasets/grid_main --H 64 --W 96
python -m ml.grid_adapter --sweep datasets/ood       --out datasets/grid_ood  --H 64 --W 96

# 5. Train (CPU, ~10 min for the small config)
python -m ml.train --grid datasets/grid_main --out checkpoints/run0 \
                   --epochs 25 --batch 4 --d-model 128 --patch 8
```

## Pipeline at a glance

```
ConfigEllipse  --PDE family-->  sweep_fvm.py  --N independent runs-->
   datasets/family_v1/sim_NNNN/{mesh_props.npz, t_*.npz, config.json}
   --grid_adapter--> datasets/grid_main/sim_NNNN.npz  (T, 4, H, W)
   --train.py-->     checkpoints/run0/model.pt
```

Each step is independent — you can re-run the sweep with different parameter ranges, or re-pack at a different resolution, without re-training.

## What the foundation-model formulation looks like

The brief asks us to formalise CFD prediction the way TabPFN formalises tabular classification. Loosely: we draw a PDE θ ~ p(θ) (here, ranges over γ, μ, μ_b, k, C_v, ...), draw an initial state s_0, simulate forward to get a trajectory s_{0:T}, and at inference time the model is given a partial trajectory s_{0:τ} and predicts s_{τ+1:T}. Implicit Bayesian marginalisation over θ, conditioned on the observed history, is exactly what an attention mechanism approximates when it is trained over many trajectories with different θ.

## Reproducibility

All sweep parameters are saved in `sim_NNNN/config.json`; the seed is logged in the run's `MANIFEST.json`. Every training run dumps its full config and per-epoch loss to `checkpoints/run0/`.

## Acknowledgements

Solver and dataset-saving code are by Max Zhu (project supervisor). Patch-transformer architecture follows the design space explored in TNT, Poseidon, Walrus and FLUID-LLM.
