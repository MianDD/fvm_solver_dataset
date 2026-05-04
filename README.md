# fvm_solver_dataset

This repository is a CFD/FVM dataset-generation and Transformer surrogate
modelling pipeline. It starts from a compressible Navier-Stokes finite-volume
solver, adds robust raw simulation sweeps and metadata, converts successful
unstructured snapshots to regular-grid tensors, and trains/evaluates patch-based
Transformer surrogates.

The numerical FVM equations, fluxes, integrators, and physical model live in
`time_fvm/`. The dataset and ML pipeline code is designed around those solver
outputs and does not change the solver mathematics.

## Current Functionality

The project has two deliberately separate pipelines.

Pipeline A, raw CFD/FVM dataset generation:

- sample controlled physical/PDE parameter families: `id`, `ood_mild`,
  `ood_hard`
- generate random ellipse meshes or stable fixed ellipse meshes
- run FVM simulations and save raw unstructured snapshots
- handle mesh-generation failures without hanging a full sweep
- validate saved snapshots for NaN/Inf, positive density/temperature, minimum
  snapshot count, and obvious numerical blow-up
- write `config.json`, `status.json`, `MANIFEST.json`, and optional
  `summary.json`

Pipeline B, ML surrogate modelling:

- read saved raw simulation folders
- skip `failed` or `invalid` simulations
- convert successful raw snapshots to regular-grid `.npz` tensors
- build sliding-window training samples with configurable context and temporal
  stride
- optionally add derivative input features `[original, dx, dy, dt]`
- train a patch Transformer in delta mode or derivative mode
- evaluate one-step and autoregressive rollout errors
- plot prediction, target, and error maps for `rho` and velocity magnitude

## Repository Layout

```text
base_cfg.py                Base paths/configuration
mesh_gen/                  Geometry and mesh generation helpers
time_fvm/                  FVM solver, physics, BCs, integrators, saver
sweep/sweep_fvm.py         Robust raw simulation sweep runner
sweep/summarize_dataset.py Dataset status and parameter summary tool
ml/grid_adapter.py         Raw unstructured snapshots to grid .npz tensors
ml/dataset.py              Sliding-window dataset and derivative features
ml/model.py                Patch Transformer and factorized attention model
ml/train.py                Training entry point
ml/evaluate.py             One-step and rollout evaluation
ml/plot_predictions.py     Prediction/error figure generation
ml/smoke_attention.py      Global/factorized attention smoke test
docs/workflow.md           More detailed local and CSD3 workflow
scripts/                   CSD3 Slurm templates and experiment scripts
```

## Output Layout

Generated artifacts should stay outside source directories:

- raw simulations: `datasets/raw_*`
- gridded tensors: `datasets/grid_*`
- checkpoints: `checkpoints/*`
- evaluation results: `eval/*`
- prediction figures: `figures/predictions/*`
- logs: `logs/*`

These generated paths are ignored by Git. Do not commit generated `.npz`
datasets, checkpoints, evaluation outputs, plots, or logs.

## Install

On Windows PowerShell:

```powershell
cd C:\Users\Lenovo\Desktop\fvm_solver_dataset
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On CSD3/Linux:

```bash
cd ~/rds/hpc-work/fvm_solver_dataset
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## End-To-End Local Smoke Test

This is the compact workflow for checking the full pipeline on CPU.

```powershell
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw_local_smoke `
  --n 2 --family id --geometry-mode fixed_ellipse --device cpu `
  --n-iter 20 --save-t 0.001 --dt 5e-4 `
  --min-A 0.1 --max-A 0.2 --lnscale 4 `
  --max-mesh-retries 2 --mesh-attempt-timeout-s 20 `
  --min-snapshots 2

.\.venv\Scripts\python.exe -m sweep.summarize_dataset --sweep datasets\raw_local_smoke --write-json

.\.venv\Scripts\python.exe -m ml.grid_adapter --sweep datasets\raw_local_smoke --out datasets\grid_local_smoke --H 64 --W 96

.\.venv\Scripts\python.exe -m ml.train --grid datasets\grid_local_smoke --out checkpoints\local_factorized `
  --epochs 2 --batch 1 --context 4 --horizon 1 --device cpu `
  --d-model 64 --heads 4 --layers 2 --patch 8 `
  --attention-type factorized

.\.venv\Scripts\python.exe -m ml.evaluate --grid datasets\grid_local_smoke `
  --ckpt checkpoints\local_factorized\best_model.pt --out eval\local_factorized `
  --device cpu --batch 1 --rollout-steps 4

.\.venv\Scripts\python.exe -m ml.plot_predictions --grid datasets\grid_local_smoke `
  --ckpt checkpoints\local_factorized\best_model.pt --out figures\predictions\local_factorized `
  --device cpu --num-examples 2
```

## Raw Dataset Generation

Main entry point:

```powershell
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw_fixed_id20 `
  --n 20 --family id --geometry-mode fixed_ellipse --device cpu `
  --n-iter 50 --save-t 0.0025 --dt 5e-4 `
  --min-A 0.2 --max-A 0.4 --lnscale 3 `
  --max-mesh-retries 2 --mesh-attempt-timeout-s 60 `
  --min-snapshots 4 --validate-physics
```

Important sweep options:

- `--family {id,ood_mild,ood_hard}` selects controlled physical/PDE ranges
- `--geometry-mode random_ellipse` preserves random geometry generation
- `--geometry-mode fixed_ellipse` uses one deterministic geometry for reliable
  CSD3 dataset generation while still varying physical/PDE parameters
- `--sample-mesh-params` also samples `lnscale`, `min_A`, and `max_A`
- `--max-mesh-retries` and `--mesh-attempt-timeout-s` bound mesh-generation
  failures
- `--validate-physics` marks bad trajectories as `invalid`

Every simulation folder receives a `status.json`. Successful, failed, and
invalid simulations are all recorded in `MANIFEST.json`.

Summarize a raw sweep:

```powershell
.\.venv\Scripts\python.exe -m sweep.summarize_dataset --sweep datasets\raw_fixed_id20 --write-json
```

## Grid Conversion

`ml.grid_adapter` converts only usable simulations. It skips folders with
`status.json` values other than `success`, while still supporting older datasets
that do not have `status.json`.

```powershell
.\.venv\Scripts\python.exe -m ml.grid_adapter --sweep datasets\raw_fixed_id20 --out datasets\grid_fixed_id20 --H 64 --W 96
```

Each grid `.npz` includes:

- `states` and backward-compatible `snapshots`, shaped `(T, C, H, W)`
- `times`, shaped `(T,)`
- `channel_names`, currently `[V_x, V_y, rho, T]`
- `x_coords`, `y_coords`, `dx`, `dy`, and `bbox`
- `metadata_json` and `cfg_json`
- `pde_vec` for diagnostics

## Transformer Surrogate

Training entry point:

```powershell
.\.venv\Scripts\python.exe -m ml.train --grid datasets\grid_fixed_id20 --out checkpoints\fixed_id20_factorized `
  --epochs 10 --batch 4 --context 4 --horizon 1 --device cpu `
  --d-model 128 --heads 4 --layers 4 --patch 8 `
  --prediction-mode derivative --integrator euler `
  --use-derivatives --derivative-mode central --strides 1,2,4 `
  --attention-type factorized
```

Current model options:

- input tensor format is `(T, C, H, W)`
- `--horizon` is currently fixed to `1`
- `--patch-t` is currently fixed to `1`
- `--prediction-mode delta` predicts `X_next - X_current`
- `--prediction-mode derivative --integrator euler` predicts `dX/dt` and uses
  Forward Euler
- `--use-derivatives` adds `dx`, `dy`, and `dt` input features only; targets
  remain `[V_x, V_y, rho, T]`
- `--strides 1,2,4` trains on variable temporal strides when enough frames
  exist
- `--input-noise-std 0.0` keeps deterministic baseline behavior
- small `--input-noise-std` values such as `0.005` or `0.01` add training-only
  Gaussian input noise scaled by the fitted input normalizer std

Checkpoint output includes:

- `best_model.pt`
- `last_model.pt`
- `model.pt`
- `train_config.json`
- `model_config.json`
- `normalizer.json`
- `history.json`
- `metrics.json`

## Attention Modes

`--attention-type global` is the original flattened global Transformer baseline
and remains the default for backward compatibility with old commands and
checkpoints.

`--attention-type factorized` is the main scalable architecture for future
ID20/ID200 experiments. It performs spatial attention over patches within each
context frame and causal temporal attention over context frames for each patch
index. Global is retained as the baseline/ablation.

For context length `T` and patches per frame `N`:

- global attention pair count: `O((T N)^2)`
- factorized attention pair count: `O(T N^2 + N T^2)`

For the current ID200 setting `H=64`, `W=96`, `patch=8`, `context=4`, there are
`N=96` patches, `384` global tokens, `147456` global attention pairs, and
`38400` factorized attention pairs, about a `3.8x` pair-count reduction.

Smoke-test both attention paths:

```powershell
.\.venv\Scripts\python.exe -m ml.smoke_attention
```

## Evaluation And Plotting

Evaluate one-step and rollout metrics:

```powershell
.\.venv\Scripts\python.exe -m ml.evaluate --grid datasets\grid_fixed_id20 `
  --ckpt checkpoints\fixed_id20_factorized\best_model.pt `
  --out eval\fixed_id20_factorized `
  --device cpu --batch 4 --rollout-steps 4,8,16
```

Plot predictions:

```powershell
.\.venv\Scripts\python.exe -m ml.plot_predictions --grid datasets\grid_fixed_id20 `
  --ckpt checkpoints\fixed_id20_factorized\best_model.pt `
  --out figures\predictions\fixed_id20_factorized `
  --device cpu --num-examples 4 --stride 1
```

Evaluation and plotting load model settings from the checkpoint by default,
including derivative features, prediction mode, integrator, and strides.

## CSD3 Scripts

Slurm templates live in `scripts/` and use `#SBATCH -A YOUR_PROJECT` as a
placeholder.

Core workflow templates:

- `scripts/csd3_sweep_fixed_tiny.slurm`
- `scripts/csd3_sweep_fixed_id20.slurm`
- `scripts/csd3_grid.slurm`
- `scripts/csd3_train_cpu.slurm`
- `scripts/csd3_train_gpu.slurm`
- `scripts/csd3_eval.slurm`
- `scripts/csd3_plot.slurm`

Attention comparison scripts:

- `scripts/csd3_train_id20_global_cmp.slurm`
- `scripts/csd3_train_id20_factorized_cmp.slurm`

Main ID200 factorized scripts:

- `scripts/csd3_train_id200_factorized.slurm`
- `scripts/csd3_eval_id200_factorized.slurm`
- `scripts/csd3_plot_id200_factorized.slurm`

Optional ID200 follow-up experiments:

- `scripts/csd3_train_id200_factorized_context8_optional.slurm`
- `scripts/csd3_train_id200_factorized_layers6_optional.slurm`

Typical CSD3 sequence after syncing the repo:

```bash
cd ~/rds/hpc-work/fvm_solver_dataset
source .venv/bin/activate
sbatch scripts/csd3_sweep_fixed_id20.slurm
sbatch scripts/csd3_grid.slurm
sbatch scripts/csd3_train_id20_global_cmp.slurm
sbatch scripts/csd3_train_id20_factorized_cmp.slurm
```

For the ID200 factorized run, generate/convert `datasets/grid_fixed_id200_csd3`
first, then run:

```bash
sbatch scripts/csd3_train_id200_factorized.slurm
sbatch scripts/csd3_eval_id200_factorized.slurm
sbatch scripts/csd3_plot_id200_factorized.slurm
```

## Sanity Checks

Useful checks before a longer run:

```powershell
.\.venv\Scripts\python.exe -m compileall ml sweep
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --help
.\.venv\Scripts\python.exe -m ml.train --help
.\.venv\Scripts\python.exe -m ml.smoke_attention
```

For the complete workflow with more explanation, see
[`docs/workflow.md`](docs/workflow.md).
