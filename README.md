# fvm_solver_dataset

This repository extends a compressible Navier-Stokes finite-volume solver into
a research pipeline for CFD/FVM dataset generation and Transformer surrogate
modelling.

The project has two deliberately separate pipelines:

- Pipeline A, raw dataset generation: sample physical parameters, generate a
  mesh, run the FVM solver, save unstructured snapshots, validate quality, and
  write metadata.
- Pipeline B, ML surrogate modelling: consume saved datasets, skip failed or
  invalid runs, convert successful simulations to grid tensors, train/evaluate
  a Transformer, and plot predictions.

The FVM equations, fluxes, integrators, and physical model live in `time_fvm/`.
The current ML upgrades do not modify that numerical solver logic.

## Repository Layout

```text
base_cfg.py                Base paths/configuration
mesh_gen/                  Mesh generation utilities
time_fvm/                  Finite-volume solver, physics, BCs, saver
sweep/sweep_fvm.py         Robust raw simulation sweep runner
sweep/summarize_dataset.py Dataset status/parameter summary tool
ml/grid_adapter.py         Unstructured snapshots to regular-grid tensors
ml/dataset.py              Sliding-window dataset with optional derivatives
ml/model.py                Patch Transformer surrogate
ml/train.py                Training entry point
ml/evaluate.py             One-step and rollout evaluation
ml/plot_predictions.py     Prediction/error plots
docs/workflow.md           End-to-end local and CSD3 workflow
scripts/                   CSD3 Slurm templates
```

## Output Layout

Generated artifacts should stay outside source directories:

- raw simulations: `datasets/raw_*`
- gridded tensors: `datasets/grid_*`
- checkpoints: `checkpoints/*`
- evaluation results: `eval/*`
- prediction figures: `figures/predictions/*`

These paths are ignored by Git.

## Quickstart

Set up the environment:

```powershell
cd C:\Users\Lenovo\Desktop\fvm_solver_dataset
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run a small end-to-end CPU smoke test:

```powershell
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw_local_smoke `
  --n 2 --family id --device cpu --n-iter 20 --save-t 0.001 --dt 5e-4 `
  --min-A 0.1 --max-A 0.2 --lnscale 4 --min-snapshots 2

.\.venv\Scripts\python.exe -m sweep.summarize_dataset --sweep datasets\raw_local_smoke --write-json

.\.venv\Scripts\python.exe -m ml.grid_adapter --sweep datasets\raw_local_smoke --out datasets\grid_local_smoke --H 64 --W 96

.\.venv\Scripts\python.exe -m ml.train --grid datasets\grid_local_smoke --out checkpoints\local_delta `
  --epochs 2 --batch 1 --context 4 --horizon 1 --device cpu `
  --d-model 64 --heads 4 --layers 2 --patch 8

.\.venv\Scripts\python.exe -m ml.evaluate --grid datasets\grid_local_smoke `
  --ckpt checkpoints\local_delta\best_model.pt --out eval\local_delta --device cpu --batch 1

.\.venv\Scripts\python.exe -m ml.plot_predictions --grid datasets\grid_local_smoke `
  --ckpt checkpoints\local_delta\best_model.pt --out figures\predictions\local_delta --device cpu
```

## GPhyT-Inspired ML Options

The model can optionally use derivative features as input and predict a
time-derivative update that is integrated by Forward Euler:

```powershell
.\.venv\Scripts\python.exe -m ml.train --grid datasets\grid_local_smoke --out checkpoints\local_derivative `
  --epochs 2 --batch 1 --context 4 --horizon 1 --device cpu `
  --d-model 64 --heads 4 --layers 2 --patch 8 `
  --prediction-mode derivative --integrator euler `
  --use-derivatives --derivative-mode central --strides 1,2
```

Derivative input channels are `[original, dx, dy, dt]` for each physical
channel. Spatial derivatives use physical `dx` and `dy` saved by
`ml.grid_adapter`; old grid files without spacing metadata fall back to index
spacing with a warning. Targets remain `[V_x, V_y, rho, T]`.

`--horizon` is currently fixed to `1`. Multi-horizon training and evaluation
are not implemented yet.

For the complete workflow, including CSD3 templates, see
[`docs/workflow.md`](docs/workflow.md).
