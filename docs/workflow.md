# Research Workflow

This project has two explicit pipelines.

Pipeline A generates raw CFD/FVM data only:

1. Sample a named parameter family.
2. Generate a mesh.
3. Run the FVM simulation.
4. Save unstructured snapshots.
5. Validate snapshot quality.
6. Write `config.json`, `status.json`, `MANIFEST.json`, and `summary.json`.

Pipeline B consumes saved data only:

1. Read raw simulation folders.
2. Skip failed or invalid simulations.
3. Convert successful unstructured snapshots to regular-grid tensors.
4. Build sliding-window samples.
5. Train a Transformer surrogate.
6. Evaluate one-step and rollout errors.
7. Plot predictions.

The numerical FVM equations, fluxes, integrators, and physical model live in
`time_fvm/` and are not changed by the ML workflow.

## Output Layout

Use separate roots so generated artifacts do not get mixed with source code:

- raw FVM simulations: `datasets/raw_*`
- gridded ML tensors: `datasets/grid_*`
- checkpoints: `checkpoints/*`
- evaluation outputs: `eval/*`
- prediction figures: `figures/predictions/*`

## A. Generate Raw FVM Dataset

```powershell
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw_local_smoke `
  --n 2 --family id --device cpu --n-iter 20 --save-t 0.001 --dt 5e-4 `
  --min-A 0.1 --max-A 0.2 --lnscale 4 `
  --max-mesh-retries 2 --mesh-attempt-timeout-s 20 --timeout-s 180 `
  --min-snapshots 2
```

Use `--family id` for in-distribution training data, `--family ood_mild` for
nearby generalisation tests, and `--family ood_hard` for stress testing. Mesh
sampling is separate from PDE/physics sampling; enable it with
`--sample-mesh-params`.

EOS sampling is also family-controlled: `id` remains ideal-gas only,
`ood_mild` samples both ideal and stiffened-gas examples, and `ood_hard`
samples stiffened-gas examples. Use `--eos-type ideal` or
`--eos-type stiffened_gas` to override the family choice explicitly.

The default shear-viscosity law is the original `sutherland` model. To make a
small constitutive-law variation dataset without changing the FVM fluxes or
integrators, use:

```powershell
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw_powerlaw_smoke `
  --n 2 --family ood_mild --geometry-mode fixed_ellipse --device cpu `
  --viscosity-law power_law --power-law-n 0.75
```

Supported laws are `sutherland`, `constant`, and `power_law`. This is a first
controlled constitutive-relation variation, not a generic PDE-template system.

Ideal gas remains the default equation of state. A minimal stiffened-gas
prototype can be invoked explicitly for Report 1/Report 2 bridge experiments:

```powershell
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw_stiffened_smoke `
  --n 1 --family id --geometry-mode fixed_ellipse --device cpu `
  --eos-type stiffened_gas --p-inf 0.001 `
  --n-iter 20 --save-t 0.001 --dt 5e-4 `
  --min-A 0.2 --max-A 0.4 --lnscale 3
```

The prototype uses `p = rho R T - p_inf` and
`c = sqrt(gamma (p + p_inf) / rho)`. Existing commands remain ideal-gas by
default. New grid files append EOS information to `pde_vec`, while old 9D and
13D grid files remain supported.

For CSD3 runs where random ellipse mesh generation is unstable, use
`--geometry-mode fixed_ellipse`. This keeps one deterministic ellipse geometry
and varies only the physical/PDE family parameters across simulations:

```powershell
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw_fixed_id20_csd3 `
  --n 20 --family id --geometry-mode fixed_ellipse --device cpu `
  --min-A 0.2 --max-A 0.4 --lnscale 3
```

## B. Summarize Raw Dataset

```powershell
.\.venv\Scripts\python.exe -m sweep.summarize_dataset --sweep datasets\raw_local_smoke --write-json
```

This prints success, failed, and invalid counts, parameter summaries, snapshot
statistics, common failure reasons, and whether the dataset is ready for grid
conversion.

## C. Convert Successful Sims To Grid Tensors

```powershell
.\.venv\Scripts\python.exe -m ml.grid_adapter --sweep datasets\raw_local_smoke --out datasets\grid_local_smoke --H 64 --W 96
```

Each output `.npz` contains:

- `states` and backward-compatible `snapshots`, shaped `(T, C, H, W)`
- `times`, shaped `(T,)`
- `channel_names`, currently `[V_x, V_y, rho, T]`
- `mask`, shaped `(H, W)`, where `1` means fluid and `0` means solid/invalid
- `x_coords`, `y_coords`, `dx`, `dy`, and `bbox` for physical derivative features
- `metadata_json` with simulation id, seed, family, physical parameters, mesh
  parameters, time settings, validation status, and source folder
- `cfg_json` for backward-compatible access to the raw simulation config
- `pde_vec` and `pde_vec_names` for physical-parameter diagnostics

Newly generated grids use a 16D `pde_vec`:
`[gamma, viscosity, visc_bulk, thermal_cond, C_v, T_0, rho_inf, T_inf,
v_n_inf, viscosity_law_sutherland, viscosity_law_constant,
viscosity_law_power_law, power_law_n, eos_type_ideal,
eos_type_stiffened_gas, p_inf]`. The loader and schema inference still support
old 9D vectors and the earlier 13D viscosity-law schema.

Old grid files without `mask` still load. The dataset class fills in an
all-fluid mask, so baseline training remains backward-compatible.

## D. Train Transformer

Default mode is backward-compatible: no derivative features, temporal stride 1,
direct delta prediction, and `--horizon 1`. Multi-horizon training is not
implemented yet.

```powershell
.\.venv\Scripts\python.exe -m ml.train --grid datasets\grid_local_smoke --out checkpoints\local_delta `
  --epochs 2 --batch 1 --context 4 --horizon 1 --device cpu `
  --d-model 64 --heads 4 --layers 2 --patch 8 `
  --prediction-mode delta --no-derivatives --strides 1
```

GPhyT-inspired mode adds derivative input features and asks the model to predict
`dX/dt`; Forward Euler integrates the derivative to the next state:

```powershell
.\.venv\Scripts\python.exe -m ml.train --grid datasets\grid_local_smoke --out checkpoints\local_derivative `
  --epochs 2 --batch 1 --context 4 --horizon 1 --device cpu `
  --d-model 64 --heads 4 --layers 2 --patch 8 `
  --prediction-mode derivative --integrator euler `
  --use-derivatives --derivative-mode central --strides 1,2
```

Boundary-mask training appends the static fluid mask as an input channel and
uses the same mask to exclude solid/invalid cells from the prediction loss:

```powershell
.\.venv\Scripts\python.exe -m ml.train --grid datasets\grid_local_smoke --out checkpoints\local_masked `
  --epochs 2 --batch 1 --context 4 --horizon 1 --device cpu `
  --d-model 64 --heads 4 --layers 2 --patch 8 `
  --use-mask-channel --mask-loss
```

The optional PDE-identification auxiliary head predicts the saved `pde_vec`
from pooled latent context tokens. It supports an implicit-PDE-identification
experiment from temporal context without changing the main next-state target:

```powershell
.\.venv\Scripts\python.exe -m ml.train --grid datasets\grid_local_smoke --out checkpoints\local_pde_aux `
  --epochs 2 --batch 1 --context 4 --horizon 1 --device cpu `
  --d-model 64 --heads 4 --layers 2 --patch 8 `
  --pde-aux-loss --pde-normalize --pde-log-transport --pde-aux-weight 0.01 `
  --pde-cont-weight 1.0 --pde-law-weight 1.0 --pde-eos-weight 1.0 `
  --pde-cont-loss huber --pde-huber-beta 1.0
```

If `--pde-aux-loss` is explicitly requested and a grid file lacks `pde_vec`, the
trainer raises a clear error. Baseline training without the flag still works on
old datasets.

The PDE auxiliary loss is normalized by default because `pde_vec` mixes
dimensionless values, tiny transport coefficients, temperature-scale values,
one-hot viscosity-law entries, and `power_law_n`. Training-set-only mean/std are
saved in the checkpoint. Viscosity, bulk viscosity, and thermal conductivity are
positive transport coefficients, so they are log-transformed before computing
normalization statistics. With the new 16D schema, continuous dimensions use a
robust normalized-space loss while the viscosity-law and EOS-type slices use
cross entropy; `p_inf` is a continuous regression target. Old 13D `pde_vec`
files still expose viscosity-law classification, and old 9D files use
normalized continuous regression over all dimensions, with indices 1, 2, and 3
log-normalized by default.

For stability, near-constant continuous dimensions are skipped using train-set
transformed standard deviation. `power_law_n` contributes only for samples whose
viscosity law is `power_law`, and `p_inf` contributes only for `stiffened_gas`
samples. The default is `--pde-cont-loss huber`; `--pde-cont-loss mse` is
available for ablations. Law/EOS accuracy on single-class datasets is
degenerate: ID-only data has only `sutherland` and `ideal`, so those accuracies
should not be reported as PDE-form identification.

The default attention path is the original flattened global Transformer:
`--attention-type global`, so old commands and checkpoints remain compatible.
For future ID20/ID200 experiments, `--attention-type factorized` is the main
scalable architecture to try, while global is retained as the baseline/ablation.
No accuracy claim is made until the matched ID20/ID200 comparison is complete.

Global attention has pair-count complexity `O((T N)^2)` for context length `T`
and patches per frame `N`. Factorized attention uses spatial attention per time
step plus causal temporal attention per patch, `O(T N^2 + N T^2)`. For the
current ID200 shape `H=64`, `W=96`, `P=8`, `T=4`, there are `N=96` patches:
global pairs `147456`, factorized pairs `38400`, about a `3.8x` reduction.

To use factorized attention while keeping the same dataset and training loop:

```powershell
.\.venv\Scripts\python.exe -m ml.train --grid datasets\grid_local_smoke --out checkpoints\local_factorized `
  --epochs 2 --batch 1 --context 4 --horizon 1 --device cpu `
  --d-model 64 --heads 4 --layers 2 --patch 8 `
  --attention-type factorized
```

The default positional embedding is `--pos-encoding learned_absolute`, which is
kept for old checkpoint compatibility. `--pos-encoding sinusoidal` is an
optional parameter-free alternative:

```powershell
.\.venv\Scripts\python.exe -m ml.train --grid datasets\grid_local_smoke --out checkpoints\local_sinusoidal `
  --epochs 1 --batch 1 --context 4 --horizon 1 --device cpu `
  --d-model 64 --heads 4 --layers 2 --patch 8 `
  --pos-encoding sinusoidal
```

Training-only input noise can be used as a small rollout-stability regularizer.
The default `--input-noise-std 0.0` keeps deterministic baseline behavior.
Values such as `0.005` or `0.01` add Gaussian noise to the input context scaled
by the fitted input-channel normalizer std. Targets stay clean, and no noise is
applied during validation, evaluation, or plotting.

```powershell
.\.venv\Scripts\python.exe -m ml.train --grid datasets\grid_local_smoke --out checkpoints\local_factorized_noise `
  --epochs 1 --batch 1 --context 4 --horizon 1 --device cpu `
  --d-model 64 --heads 4 --layers 2 --patch 8 `
  --attention-type factorized --input-noise-std 0.005
```

`--horizon 1` and `--patch-t 1` are currently implemented. Larger temporal
tubelets are the next architecture step because they require changing both
tokenisation and detokenisation; this pass keeps the stable spatial patch
decoder.

Scope note: the current solver remains an ideal-gas compressible
Navier-Stokes-family generator. It now includes scalar parameter variation,
fixed/random geometry choices, and an initial shear-viscosity-law switch, but it
is not yet a fully generic PDE registry. The temporal context window is an
implicit PDE-identification mechanism, not a direct TabPFN-style in-context
learning system.

Training output contains:

- `best_model.pt`
- `last_model.pt`
- `model.pt`
- `train_config.json`
- `model_config.json`
- `normalizer.json`
- `history.json`
- `metrics.json`

## E. Evaluate

```powershell
.\.venv\Scripts\python.exe -m ml.evaluate --grid datasets\grid_local_smoke `
  --ckpt checkpoints\local_delta\best_model.pt --out eval\local_delta `
  --device cpu --batch 1 --rollout-steps 4,8,16
```

Evaluation reports one-step MSE, MAE, relative L2, per-channel errors, metrics
grouped by family and stride, and rollout metrics when enough frames are
available.

If the checkpoint includes a PDE auxiliary head and the grid files include
`pde_vec`, evaluation also writes `pde_metrics.json`:

```powershell
.\.venv\Scripts\python.exe -m ml.evaluate --grid datasets\grid_local_smoke `
  --ckpt checkpoints\local_pde_aux\best_model.pt --out eval\local_pde_aux `
  --device cpu --batch 1 --rollout-steps 4,8
```

The PDE metrics include continuous-parameter MAE/RMSE/R2 and, when categorical
slices are present, viscosity-law and EOS-type accuracy, per-class accuracy,
and confusion matrices. If grid metadata contains family labels such as `id`,
`ood_mild`, or `ood_hard`, `pde_metrics.json` also includes a `by_family`
section for Report 1 tables. EOS identification metrics are only meaningful on
datasets that contain both ideal and stiffened-gas samples.

## F. Plot Predictions

```powershell
.\.venv\Scripts\python.exe -m ml.plot_predictions --grid datasets\grid_local_smoke `
  --ckpt checkpoints\local_delta\best_model.pt --out figures\predictions\local_delta `
  --device cpu --num-examples 2
```

Plots compare context final frame, target, prediction, and absolute error for
`rho` and velocity magnitude. Filenames include simulation id, stride, step, and
channel. Use `--stride N` to choose the plotted temporal stride; otherwise the
first stride saved in the checkpoint is used.

For Report 1 metric figures from `metrics.json` or `pde_metrics.json`:

```powershell
.\.venv\Scripts\python.exe -m ml.plot_report1_metrics `
  --metrics eval\local_pde_aux\metrics.json `
  --out figures\report1\local_pde_aux
```

This produces available plots such as `pde_r2_bar.png`, `pde_mae_bar.png`,
`viscosity_law_confusion.png`, `eos_type_confusion.png`, and
`rollout_error.png`. Missing metric sections are skipped gracefully.

Grouped ID/OOD plots from separate evaluation outputs:

```powershell
.\.venv\Scripts\python.exe -m ml.plot_report1_metrics `
  --metrics-list id:eval\id\metrics.json,ood_mild:eval\ood_mild\metrics.json,ood_hard:eval\ood_hard\metrics.json `
  --out figures\report1\grouped
```

When a single `pde_metrics.json` contains `by_family`, the same plotting script
also creates grouped law-accuracy and mean-R2 plots from that file.

## G. Context-Size Scaling

Context-size scaling evaluates the same checkpoint with different numbers of
observed frames. This is useful for Report 1 because temporal context is the
mechanism used for implicit PDE identification.

```powershell
.\.venv\Scripts\python.exe -m ml.evaluate_context_scaling `
  --grid datasets\grid_local_smoke `
  --ckpt checkpoints\local_pde_aux\best_model.pt `
  --out eval\context_scaling_local_pde_aux `
  --contexts 2 4 8 16 --device cpu --batch 1

.\.venv\Scripts\python.exe -m ml.plot_report1_metrics `
  --context-scaling eval\context_scaling_local_pde_aux\context_scaling_metrics.json `
  --out figures\report1\context_scaling_local_pde_aux
```

The output JSON records `mse`, `mae`, optional PDE mean R2, optional
viscosity-law accuracy, optional EOS-type accuracy, and the number of windows
for each requested context.
If a context is too long for the dataset, or if a `learned_absolute` checkpoint
is asked to exceed its stored `max_context`, that context is skipped with a
recorded reason instead of crashing. `sinusoidal` checkpoints can be evaluated
at other context lengths when the trajectory has enough saved frames.

## Local Windows Smoke Test

```powershell
cd C:\Users\Lenovo\Desktop\fvm_solver_dataset

Remove-Item -Recurse -Force datasets\raw_local_smoke,datasets\grid_local_smoke,checkpoints\local_delta,checkpoints\local_derivative,eval\local_delta,eval\local_derivative,figures\predictions\local_delta,figures\predictions\local_derivative -ErrorAction SilentlyContinue

.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw_local_smoke `
  --n 2 --family id --device cpu --n-iter 20 --save-t 0.001 --dt 5e-4 `
  --min-A 0.1 --max-A 0.2 --lnscale 4 `
  --max-mesh-retries 2 --mesh-attempt-timeout-s 20 --timeout-s 180 `
  --min-snapshots 2

.\.venv\Scripts\python.exe -m sweep.summarize_dataset --sweep datasets\raw_local_smoke --write-json

.\.venv\Scripts\python.exe -m ml.grid_adapter --sweep datasets\raw_local_smoke --out datasets\grid_local_smoke --H 64 --W 96

.\.venv\Scripts\python.exe -m ml.train --grid datasets\grid_local_smoke --out checkpoints\local_delta `
  --epochs 2 --batch 1 --context 4 --horizon 1 --device cpu `
  --d-model 64 --heads 4 --layers 2 --patch 8 `
  --prediction-mode delta --no-derivatives --strides 1

.\.venv\Scripts\python.exe -m ml.train --grid datasets\grid_local_smoke --out checkpoints\local_derivative `
  --epochs 2 --batch 1 --context 4 --horizon 1 --device cpu `
  --d-model 64 --heads 4 --layers 2 --patch 8 `
  --prediction-mode derivative --integrator euler `
  --use-derivatives --derivative-mode central --strides 1,2

.\.venv\Scripts\python.exe -m ml.evaluate --grid datasets\grid_local_smoke `
  --ckpt checkpoints\local_delta\best_model.pt --out eval\local_delta --device cpu --batch 1

.\.venv\Scripts\python.exe -m ml.evaluate --grid datasets\grid_local_smoke `
  --ckpt checkpoints\local_derivative\best_model.pt --out eval\local_derivative --device cpu --batch 1

.\.venv\Scripts\python.exe -m ml.plot_predictions --grid datasets\grid_local_smoke `
  --ckpt checkpoints\local_delta\best_model.pt --out figures\predictions\local_delta --device cpu --num-examples 2

.\.venv\Scripts\python.exe -m ml.plot_predictions --grid datasets\grid_local_smoke `
  --ckpt checkpoints\local_derivative\best_model.pt --out figures\predictions\local_derivative --device cpu --num-examples 2
```

## CSD3 Sketch

Keep code and generated data under:

```bash
~/rds/hpc-work/fvm_solver_dataset
```

Create a virtual environment once:

```bash
cd ~/rds/hpc-work/fvm_solver_dataset
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Submit the templates in `scripts/` after replacing `YOUR_PROJECT`:

```bash
sbatch scripts/csd3_sweep.slurm
sbatch scripts/csd3_sweep_fixed_tiny.slurm
sbatch scripts/csd3_sweep_fixed_id20.slurm
sbatch scripts/csd3_sweep_fixed_ood_mild.slurm
sbatch scripts/csd3_sweep_fixed_ood_hard.slurm
sbatch scripts/csd3_grid.slurm
sbatch scripts/csd3_train_cpu.slurm
sbatch scripts/csd3_train_gpu.slurm
sbatch scripts/csd3_eval.slurm
sbatch scripts/csd3_plot.slurm
```

For a controlled ID20 attention comparison after generating/converting
`datasets/grid_fixed_id20_csd3`, run:

```bash
sbatch scripts/csd3_train_id20_global_cmp.slurm
sbatch scripts/csd3_train_id20_factorized_cmp.slurm
```

For the main ID200 factorized CPU run after generating/converting
`datasets/grid_fixed_id200_csd3`, run:

```bash
sbatch scripts/csd3_train_id200_factorized.slurm
sbatch scripts/csd3_eval_id200_factorized.slurm
sbatch scripts/csd3_plot_id200_factorized.slurm
```

Optional follow-up experiments are provided for a longer context and deeper
factorized encoder:

```bash
sbatch scripts/csd3_train_id200_factorized_context8_optional.slurm
sbatch scripts/csd3_train_id200_factorized_layers6_optional.slurm
```

Always summarize before conversion:

```bash
python -m sweep.summarize_dataset --sweep datasets/raw_family_v1 --write-json
```
