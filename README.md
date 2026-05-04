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
ml/pde.py                  PDE vector schema, normalized loss, metrics helpers
ml/train.py                Training entry point
ml/evaluate.py             One-step and rollout evaluation
ml/evaluate_context_scaling.py Context-length scaling evaluation
ml/plot_predictions.py     Prediction/error figure generation
ml/plot_report1_metrics.py Report 1 PDE/rollout metric figures
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
- `--viscosity-law {family,sutherland,constant,power_law}` optionally overrides
  the family-sampled constitutive law
- `--power-law-n` sets the exponent for `power_law` viscosity
- `--eos-type {family,ideal,stiffened_gas}` optionally selects the equation of
  state; `family` uses the selected family distribution, while `id` remains
  ideal-gas only
- `--p-inf` sets the stiffened-gas pressure offset when `--eos-type stiffened_gas`
- `--max-mesh-retries` and `--mesh-attempt-timeout-s` bound mesh-generation
  failures
- `--validate-physics` marks bad trajectories as `invalid`

Every simulation folder receives a `status.json`. Successful, failed, and
invalid simulations are all recorded in `MANIFEST.json`.

Family EOS sampling is conservative: `id` remains ideal-gas only,
`ood_mild` samples both ideal and stiffened-gas cases, and `ood_hard` samples
stiffened-gas cases with larger `p_inf`. CLI overrides still take precedence.

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
- `mask`, shaped `(H, W)`, with `1` for fluid grid points and `0` for
  solid/invalid/outside-obstacle points
- `x_coords`, `y_coords`, `dx`, `dy`, and `bbox`
- `metadata_json` and `cfg_json`
- `pde_vec` and `pde_vec_names` for physical-parameter diagnostics

Newly generated grids use a 16D `pde_vec`:
`[gamma, viscosity, visc_bulk, thermal_cond, C_v, T_0, rho_inf, T_inf,
v_n_inf, viscosity_law_sutherland, viscosity_law_constant,
viscosity_law_power_law, power_law_n, eos_type_ideal,
eos_type_stiffened_gas, p_inf]`. Older 9D and 13D vectors are still inferred
and supported.

Older grid files without `mask` still load: the dataset class creates an
all-fluid mask and prints warnings when mask-dependent options are requested.

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
- `--use-mask-channel` appends the static fluid mask as an input channel
- `--mask-loss` computes the prediction loss over fluid cells only
- `--pde-aux-loss --pde-aux-weight 0.01` adds an optional auxiliary head that
  predicts `pde_vec` from pooled latent context tokens
- `--pde-normalize` is enabled by default with `--pde-aux-loss`; it computes
  `pde_vec` mean/std from the training split only and stores those statistics
  in the checkpoint
- `--pde-log-transport` is also enabled by default and normalizes viscosity,
  bulk viscosity, and thermal conductivity in log-space
- `--pde-cont-weight`, `--pde-law-weight`, and `--pde-eos-weight` weight the
  continuous-parameter regression, viscosity-law classification, and EOS-type
  classification terms inside the auxiliary loss
- `--pde-cont-loss huber` is the default robust continuous PDE loss in
  normalized space; use `--pde-cont-loss mse` only for ablations
- `--pos-encoding learned_absolute` is the default checkpoint-compatible
  position embedding; `--pos-encoding sinusoidal` is parameter-free and less
  tied to learned position-table sizes
- `--strides 1,2,4` trains on variable temporal strides when enough frames
  exist
- `--input-noise-std 0.0` keeps deterministic baseline behavior
- small `--input-noise-std` values such as `0.005` or `0.01` add training-only
  Gaussian input noise scaled by the fitted input normalizer std
- `--pushforward-prob 0.0` keeps the original single-step training loop
- `--pushforward-prob 0.5` enables a training-only rollout-stability
  augmentation: the model predicts a detached one-step state from the previous
  context and uses it to replace only the last-frame physical channels before
  the final supervised prediction
- `--rollout-train-steps` is reserved for future multi-step rollout loss and
  currently must remain `1`

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

Smoke-test the Report 1 optional features on synthetic tensors:

```powershell
.\.venv\Scripts\python.exe -m ml.smoke_report1
```

## Report 1 Extensions And Scope

Boundary masks are included in new gridded datasets so the model can receive
geometry information and the loss can ignore solid or invalid cells. Enable
them during training with `--use-mask-channel --mask-loss`.

The optional PDE auxiliary head is a lightweight way to test whether the
temporal context encodes enough information to recover sampled physical
parameters. It is enabled with `--pde-aux-loss`; if a grid file lacks `pde_vec`,
baseline training still works, while an explicitly requested auxiliary loss
raises a clear error.

Because `pde_vec` mixes values with different scales, the auxiliary loss
normalizes continuous parameters using train-set-only mean/std by default.
Transport coefficients are especially small and positive, so viscosity,
bulk viscosity, and thermal conductivity are log-transformed before computing
their normalization statistics. This avoids raw-scale domination and makes
relative transport-coefficient errors more meaningful. For newly generated 16D
grids, the viscosity-law and EOS one-hot slices are treated as classification
logits and trained with cross entropy; `p_inf` remains a continuous regression
target. Old 13D `pde_vec` files still provide viscosity-law classification, and
old 9D files fall back to normalized continuous regression over all dimensions,
with the same log-space treatment for indices 1, 2, and 3.

The continuous PDE auxiliary loss is deliberately conservative for stability.
Near-constant continuous dimensions are skipped using train-set transformed
standard deviation, `power_law_n` is supervised only on `power_law` samples, and
`p_inf` is supervised only on `stiffened_gas` samples. The default continuous
loss is Huber in normalized PDE space, which avoids thousands-scale losses from
tiny-variance or conditionally meaningless dimensions. Categorical accuracies
are marked/warned as degenerate when the training or evaluation set contains
only one class; ID-only data has trivial viscosity-law and EOS accuracy.

Example Report 1 training command:

```powershell
.\.venv\Scripts\python.exe -m ml.train --grid datasets\grid_fixed_id20 --out checkpoints\fixed_id20_report1 `
  --epochs 10 --batch 4 --context 4 --horizon 1 --device cpu `
  --d-model 128 --heads 4 --layers 4 --patch 8 `
  --attention-type factorized --prediction-mode derivative --integrator euler `
  --use-derivatives --use-mask-channel --mask-loss `
  --pde-aux-loss --pde-normalize --pde-log-transport --pde-aux-weight 0.01 `
  --pde-cont-weight 1.0 --pde-law-weight 1.0 --pde-eos-weight 1.0 `
  --pde-cont-loss huber --pde-huber-beta 1.0 `
  --pushforward-prob 0.5
```

Pushforward training is off by default. When enabled, derivative and mask input
channels are preserved rather than recomputed after the last physical frame is
replaced, so it should be interpreted as a lightweight pushforward-noise
augmentation rather than exact multi-step training.

The solver now supports three shear-viscosity laws: `sutherland` (the original
default), `constant`, and `power_law`. This is a first modular step toward
varying constitutive relations. It is not a full generic PDE registry: the
solver is still primarily the ideal-gas compressible Navier-Stokes family, with
scalar parameter variation and initial constitutive-law variation.

An optional `stiffened_gas` EOS prototype is available behind explicit flags.
The default remains `ideal`, so existing runs are unchanged. The prototype uses
`p = rho R T - p_inf` and `c = sqrt(gamma (p + p_inf) / rho)` with simple
floors for numerical safety. EOS metadata is saved in config/status/manifest and
grid metadata; new grids append `eos_type_ideal`, `eos_type_stiffened_gas`, and
`p_inf` to `pde_vec` so the PDE auxiliary head can identify EOS variation.
EOS identification metrics are meaningful only for datasets that actually
contain both ideal and stiffened-gas examples.

The temporal context window supports implicit PDE identification from observed
dynamics, but it is not identical to TabPFN-style explicit in-context learning.
Future technical work should add non-ideal equations of state, non-Newtonian or
temperature-dependent rheology beyond this first law switch, a cleaner PDE
registry, and neural-operator baselines.

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
If the checkpoint has a PDE auxiliary head and the grid files include
`pde_vec`, `ml.evaluate` also writes `pde_metrics.json` with continuous
parameter MAE/RMSE/R2, viscosity-law accuracy/confusion-matrix metrics, and
EOS-type accuracy/confusion-matrix metrics when the 16D schema is present.
When family labels are available in grid metadata, the PDE metrics also include
`by_family` entries for ID/OOD breakdowns.

Create Report 1 metric figures from an evaluation JSON:

```powershell
.\.venv\Scripts\python.exe -m ml.plot_report1_metrics `
  --metrics eval\fixed_id20_factorized\metrics.json `
  --out figures\report1\fixed_id20_factorized
```

Grouped ID/OOD comparison plots can be generated from separate evaluation
outputs:

```powershell
.\.venv\Scripts\python.exe -m ml.plot_report1_metrics `
  --metrics-list id:eval\id\metrics.json,ood_mild:eval\ood_mild\metrics.json,ood_hard:eval\ood_hard\metrics.json `
  --out figures\report1\grouped
```

Evaluate how performance changes with the observed temporal context length:

```powershell
.\.venv\Scripts\python.exe -m ml.evaluate_context_scaling `
  --grid datasets\grid_fixed_id20 `
  --ckpt checkpoints\fixed_id20_factorized\best_model.pt `
  --out eval\context_scaling_fixed_id20 `
  --contexts 2 4 8 16 --device cpu --batch 4

.\.venv\Scripts\python.exe -m ml.plot_report1_metrics `
  --context-scaling eval\context_scaling_fixed_id20\context_scaling_metrics.json `
  --out figures\report1\context_scaling_fixed_id20
```

`learned_absolute` checkpoints skip contexts larger than their stored
`max_context`; `sinusoidal` checkpoints can be evaluated at other context
lengths when enough frames exist.

## CSD3 Scripts

Slurm templates live in `scripts/` and use `#SBATCH -A YOUR_PROJECT` as a
placeholder.

Core workflow templates:

- `scripts/csd3_sweep_fixed_tiny.slurm`
- `scripts/csd3_sweep_fixed_id20.slurm`
- `scripts/csd3_sweep_fixed_ood_mild.slurm`
- `scripts/csd3_sweep_fixed_ood_hard.slurm`
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
sbatch scripts/csd3_sweep_fixed_ood_mild.slurm
sbatch scripts/csd3_sweep_fixed_ood_hard.slurm
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
.\.venv\Scripts\python.exe -m compileall ml sweep time_fvm
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --help
.\.venv\Scripts\python.exe -m ml.train --help
.\.venv\Scripts\python.exe -m ml.evaluate --help
.\.venv\Scripts\python.exe -m ml.evaluate_context_scaling --help
.\.venv\Scripts\python.exe -m ml.plot_report1_metrics --help
.\.venv\Scripts\python.exe -m ml.smoke_attention
.\.venv\Scripts\python.exe -m ml.smoke_report1
```

For the complete workflow with more explanation, see
[`docs/workflow.md`](docs/workflow.md).
