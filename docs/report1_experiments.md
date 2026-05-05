# Report 1 Experiment Workflow

This document is the executable workflow for the current Report 1 code state.
It assumes the repository is already patched with:

- corrected stiffened-gas EOS;
- bulk viscosity following `viscosity_law`;
- ID/OOD family sampling with 17D `pde_vec`;
- grid-mask neutralisation of non-fluid pixels;
- mask-aware training normalisation and raw binary mask channel;
- global attention decoding only the final context token.

The goal for Report 1 is to demonstrate a foundation-model-style CFD pipeline
for compressible Navier-Stokes trajectories: generate controlled CFD/FVM
families, convert successful simulations into gridded tensors, train a
temporal-context Transformer surrogate, and report ID/OOD prediction and
PDE-identification metrics. This is not a fully generic PDE registry. The
current physics family is ideal-gas compressible Navier-Stokes with controlled
transport-coefficient, viscosity-law, and minimal stiffened-gas EOS variation.

## Current Dataset Families

Family sampling is implemented in `sweep/sweep_fvm.py`.

`id`:

- EOS: ideal gas only
- `p_inf = 0`
- `p_inf_ratio = 0`
- viscosity law: `sutherland`
- intended use: main in-distribution training and baseline evaluation

`ood_mild`:

- EOS: mixture of ideal and stiffened gas
- stiffened cases use `p_inf_ratio in [0.02, 0.08]`
- viscosity laws: `sutherland`, `constant`, `power_law`
- intended use: nearby OOD generalisation and mixed PDE-form identification

`ood_hard`:

- EOS: stiffened gas
- `p_inf_ratio in [0.08, 0.20]`
- viscosity laws: `constant`, `power_law`
- intended use: stress-test OOD evaluation

For stiffened gas, the sampler computes:

```text
R = C_v * (gamma - 1)
p_inf = p_inf_ratio * R * rho_inf * T_inf / gamma
```

It also checks a representative state before accepting a stiffened-gas sample:

```text
P_ref = R * rho_inf * T_inf - gamma * p_inf
c2_arg_ref = P_ref + p_inf
```

Both must be positive.

## pde_vec Schema

New grid files use a 17D `pde_vec`:

```text
0  gamma
1  viscosity
2  visc_bulk
3  thermal_cond
4  C_v
5  T_0
6  rho_inf
7  T_inf
8  v_n_inf
9  viscosity_law_sutherland
10 viscosity_law_constant
11 viscosity_law_power_law
12 power_law_n
13 eos_type_ideal
14 eos_type_stiffened_gas
15 p_inf
16 p_inf_ratio
```

Old 9D, 13D, and 16D grid files remain loadable, but do not mix different
`pde_vec` lengths in the same training directory. Keep old 16D grids and new
17D grids in separate folders.

## Mask Convention

Grid files contain a binary `mask`:

```text
1 = fluid
0 = solid_or_invalid
```

The gridded state convention is primitive variables:

```text
[V_x, V_y, rho, T]
```

During grid conversion, non-fluid pixels are neutralised before saving:

```text
V_x = 0
V_y = 0
rho = rho_inf, or a safe positive fallback
T   = T_inf, or a safe positive fallback
```

During training, if masks are present:

- physical and derivative channel statistics are computed over `mask == 1`
  cells only;
- the optional mask channel is kept as raw binary `0/1` with mean `0` and std
  `1`;
- `--mask-loss` computes prediction loss over fluid cells only.

For ellipse/obstacle datasets, use both `--use-mask-channel` and `--mask-loss`
unless you are deliberately running an ablation.

## Recommended Execution Order

1. Run smoke tests.
2. Generate tiny local raw data.
3. Summarize raw data.
4. Convert successful raw simulations to grid tensors.
5. Train a tiny model.
6. Evaluate and plot.
7. Generate ID20 debug data.
8. Train/debug on ID20.
9. Scale to ID200, then optional ID1000.
10. Generate OOD-mild/OOD-hard datasets.
11. Evaluate the same checkpoints on ID and OOD grids.
12. Produce Report 1 figures and tables from JSON outputs.

## Smoke Tests

Run these before large jobs:

```powershell
.\.venv\Scripts\python.exe -m compileall ml sweep time_fvm mesh_gen
.\.venv\Scripts\python.exe -m ml.smoke_attention
.\.venv\Scripts\python.exe -m ml.smoke_report1
.\.venv\Scripts\python.exe -m ml.smoke_masked_normalization
.\.venv\Scripts\python.exe -m ml.smoke_grid_mask
.\.venv\Scripts\python.exe -m sweep.smoke_family_sampling
.\.venv\Scripts\python.exe -m time_fvm.smoke_eos
.\.venv\Scripts\python.exe -m time_fvm.smoke_viscosity
```

## Full Local Smoke Pipeline

This is the first command sequence to run locally. It uses fixed geometry and
three simulations because training needs at least one train and one validation
simulation.

```powershell
Remove-Item -Recurse -Force datasets\raw\report1_smoke_id3 -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force datasets\gridded\report1_smoke_id3 -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force checkpoints\report1\smoke_factorized -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force eval\report1\smoke_factorized -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force figures\report1\smoke_factorized -ErrorAction SilentlyContinue

.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw\report1_smoke_id3 `
  --n 3 --family id --geometry-mode fixed_ellipse --device cpu `
  --n-iter 60 --save-t 0.0025 --dt 5e-4 `
  --min-A 0.2 --max-A 0.4 --lnscale 3 `
  --max-mesh-retries 2 --mesh-attempt-timeout-s 30 --timeout-s 300 `
  --min-snapshots 4 --validate-physics

.\.venv\Scripts\python.exe -m sweep.summarize_dataset `
  --sweep datasets\raw\report1_smoke_id3 --write-json

.\.venv\Scripts\python.exe -m ml.grid_adapter `
  --sweep datasets\raw\report1_smoke_id3 `
  --out datasets\gridded\report1_smoke_id3 --H 32 --W 48

.\.venv\Scripts\python.exe -m ml.train `
  --grid datasets\gridded\report1_smoke_id3 `
  --out checkpoints\report1\smoke_factorized `
  --epochs 2 --batch 1 --context 4 --horizon 1 --device cpu `
  --d-model 32 --heads 4 --layers 1 --patch 8 `
  --attention-type factorized --pos-encoding sinusoidal `
  --prediction-mode derivative --integrator euler `
  --use-derivatives --derivative-mode central --strides 1 `
  --use-mask-channel --mask-loss `
  --pde-aux-loss --pde-normalize --pde-log-transport `
  --pde-cont-loss huber --pde-aux-weight 0.01

.\.venv\Scripts\python.exe -m ml.evaluate `
  --grid datasets\gridded\report1_smoke_id3 `
  --ckpt checkpoints\report1\smoke_factorized\best_model.pt `
  --out eval\report1\smoke_factorized `
  --context 4 --horizon 1 --batch 1 --device cpu --rollout-steps 2

.\.venv\Scripts\python.exe -m ml.plot_predictions `
  --grid datasets\gridded\report1_smoke_id3 `
  --ckpt checkpoints\report1\smoke_factorized\best_model.pt `
  --out figures\report1\smoke_factorized\predictions `
  --context 4 --horizon 1 --stride 1 --num-examples 2 --device cpu

.\.venv\Scripts\python.exe -m ml.plot_report1_metrics `
  --metrics eval\report1\smoke_factorized\metrics.json `
  --out figures\report1\smoke_factorized\metrics
```

Expected outputs:

- raw simulation folders in `datasets/raw/report1_smoke_id3`
- `MANIFEST.json` and `summary.json`
- gridded `.npz` files in `datasets/gridded/report1_smoke_id3`
- `best_model.pt`, `last_model.pt`, `history.json`, `metrics.json`,
  `normalizer.json`, and `model_config.json` in the checkpoint folder
- evaluation `metrics.json` and, when a PDE head exists, `pde_metrics.json`
- prediction PNGs and metric PNGs in `figures/report1/smoke_factorized`

## Raw Dataset Generation Commands

All commands below use fixed geometry for CSD3/Windows reliability. They vary
physics/PDE parameters across simulations while keeping geometry fixed.

### ID20 Debug

```powershell
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw\report1_id_20 `
  --n 20 --family id --geometry-mode fixed_ellipse --device cpu `
  --n-iter 500 --save-t 0.01 --dt 5e-4 `
  --min-A 0.2 --max-A 0.4 --lnscale 3 `
  --max-mesh-retries 2 --mesh-attempt-timeout-s 30 --timeout-s 900 `
  --min-snapshots 10 --validate-physics
```

### ID200 Main

```powershell
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw\report1_id_200 `
  --n 200 --family id --geometry-mode fixed_ellipse --device cpu `
  --n-iter 500 --save-t 0.01 --dt 5e-4 `
  --min-A 0.2 --max-A 0.4 --lnscale 3 `
  --max-mesh-retries 2 --mesh-attempt-timeout-s 30 --timeout-s 900 `
  --min-snapshots 10 --validate-physics
```

### ID1000 Optional Large

```powershell
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw\report1_id_1000 `
  --n 1000 --family id --geometry-mode fixed_ellipse --device cpu `
  --n-iter 500 --save-t 0.01 --dt 5e-4 `
  --min-A 0.2 --max-A 0.4 --lnscale 3 `
  --max-mesh-retries 2 --mesh-attempt-timeout-s 30 --timeout-s 900 `
  --min-snapshots 10 --validate-physics
```

### OOD-Mild 100

```powershell
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw\report1_ood_mild_100 `
  --n 100 --family ood_mild --geometry-mode fixed_ellipse --device cpu `
  --n-iter 500 --save-t 0.01 --dt 5e-4 `
  --min-A 0.2 --max-A 0.4 --lnscale 3 `
  --max-mesh-retries 2 --mesh-attempt-timeout-s 30 --timeout-s 900 `
  --min-snapshots 10 --validate-physics
```

Use `--n 200` and output `datasets\raw\report1_ood_mild_200` for a larger OOD
set.

### OOD-Hard 100

```powershell
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw\report1_ood_hard_100 `
  --n 100 --family ood_hard --geometry-mode fixed_ellipse --device cpu `
  --n-iter 500 --save-t 0.01 --dt 5e-4 `
  --min-A 0.2 --max-A 0.4 --lnscale 3 `
  --max-mesh-retries 2 --mesh-attempt-timeout-s 30 --timeout-s 900 `
  --min-snapshots 10 --validate-physics
```

Use `--n 200` and output `datasets\raw\report1_ood_hard_200` for a larger OOD
set.

Summarize each raw dataset:

```powershell
.\.venv\Scripts\python.exe -m sweep.summarize_dataset --sweep datasets\raw\report1_id_20 --write-json
.\.venv\Scripts\python.exe -m sweep.summarize_dataset --sweep datasets\raw\report1_id_200 --write-json
.\.venv\Scripts\python.exe -m sweep.summarize_dataset --sweep datasets\raw\report1_ood_mild_100 --write-json
.\.venv\Scripts\python.exe -m sweep.summarize_dataset --sweep datasets\raw\report1_ood_hard_100 --write-json
```

## Grid Conversion Commands

```powershell
.\.venv\Scripts\python.exe -m ml.grid_adapter --sweep datasets\raw\report1_id_20 `
  --out datasets\gridded\report1_id_20 --H 64 --W 96

.\.venv\Scripts\python.exe -m ml.grid_adapter --sweep datasets\raw\report1_id_200 `
  --out datasets\gridded\report1_id_200 --H 64 --W 96

.\.venv\Scripts\python.exe -m ml.grid_adapter --sweep datasets\raw\report1_id_1000 `
  --out datasets\gridded\report1_id_1000 --H 64 --W 96

.\.venv\Scripts\python.exe -m ml.grid_adapter --sweep datasets\raw\report1_ood_mild_100 `
  --out datasets\gridded\report1_ood_mild_100 --H 64 --W 96

.\.venv\Scripts\python.exe -m ml.grid_adapter --sweep datasets\raw\report1_ood_hard_100 `
  --out datasets\gridded\report1_ood_hard_100 --H 64 --W 96
```

Skip the ID1000 command until that optional raw dataset exists.

## Main Report 1 Training Commands

### ID20 Debug Model

```powershell
.\.venv\Scripts\python.exe -m ml.train `
  --grid datasets\gridded\report1_id_20 `
  --out checkpoints\report1\id20_factorized_derivative_pde `
  --epochs 5 --batch 2 --context 4 --horizon 1 --device cpu `
  --d-model 64 --heads 4 --layers 2 --patch 8 `
  --attention-type factorized --pos-encoding sinusoidal `
  --prediction-mode derivative --integrator euler `
  --use-derivatives --derivative-mode central --strides 1,2,4 `
  --use-mask-channel --mask-loss `
  --pde-aux-loss --pde-normalize --pde-log-transport `
  --pde-cont-loss huber --pde-aux-weight 0.01 `
  --pde-cont-weight 1.0 --pde-law-weight 1.0 --pde-eos-weight 1.0 `
  --input-noise-std 0.005 --pushforward-prob 0.5
```

### ID200 Main Model

```powershell
.\.venv\Scripts\python.exe -m ml.train `
  --grid datasets\gridded\report1_id_200 `
  --out checkpoints\report1\id200_factorized_derivative_pde `
  --epochs 10 --batch 4 --context 4 --horizon 1 --device cpu `
  --d-model 128 --heads 4 --layers 4 --patch 8 `
  --attention-type factorized --pos-encoding sinusoidal `
  --prediction-mode derivative --integrator euler `
  --use-derivatives --derivative-mode central --strides 1,2,4 `
  --use-mask-channel --mask-loss `
  --pde-aux-loss --pde-normalize --pde-log-transport `
  --pde-cont-loss huber --pde-aux-weight 0.01 `
  --pde-cont-weight 1.0 --pde-law-weight 1.0 --pde-eos-weight 1.0 `
  --input-noise-std 0.005 --pushforward-prob 0.5
```

ID-only data has only `sutherland` viscosity and ideal gas. The law/EOS
accuracy values on an ID-only training set are therefore degenerate and should
not be reported as PDE-form identification. For meaningful viscosity-law and
EOS identification metrics, train or fine-tune on a mixed dataset containing
OOD-mild examples, or use OOD-mild as a separate PDE-identification experiment.

## Ablation Training Commands

Use the same grid and hyperparameters as the main run. Change only the listed
factor for controlled ablations.

### No PDE Auxiliary Loss

```powershell
.\.venv\Scripts\python.exe -m ml.train `
  --grid datasets\gridded\report1_id_200 `
  --out checkpoints\report1\id200_ablate_no_pde_aux `
  --epochs 10 --batch 4 --context 4 --horizon 1 --device cpu `
  --d-model 128 --heads 4 --layers 4 --patch 8 `
  --attention-type factorized --pos-encoding sinusoidal `
  --prediction-mode derivative --integrator euler `
  --use-derivatives --derivative-mode central --strides 1,2,4 `
  --use-mask-channel --mask-loss `
  --input-noise-std 0.005 --pushforward-prob 0.5
```

### Delta Prediction Instead Of Derivative

```powershell
.\.venv\Scripts\python.exe -m ml.train `
  --grid datasets\gridded\report1_id_200 `
  --out checkpoints\report1\id200_ablate_delta `
  --epochs 10 --batch 4 --context 4 --horizon 1 --device cpu `
  --d-model 128 --heads 4 --layers 4 --patch 8 `
  --attention-type factorized --pos-encoding sinusoidal `
  --prediction-mode delta --integrator euler `
  --use-derivatives --derivative-mode central --strides 1,2,4 `
  --use-mask-channel --mask-loss `
  --pde-aux-loss --pde-normalize --pde-log-transport `
  --pde-cont-loss huber --pde-aux-weight 0.01
```

### Global Attention Baseline

```powershell
.\.venv\Scripts\python.exe -m ml.train `
  --grid datasets\gridded\report1_id_200 `
  --out checkpoints\report1\id200_ablate_global_attention `
  --epochs 10 --batch 4 --context 4 --horizon 1 --device cpu `
  --d-model 128 --heads 4 --layers 4 --patch 8 `
  --attention-type global --pos-encoding sinusoidal `
  --prediction-mode derivative --integrator euler `
  --use-derivatives --derivative-mode central --strides 1,2,4 `
  --use-mask-channel --mask-loss `
  --pde-aux-loss --pde-normalize --pde-log-transport `
  --pde-cont-loss huber --pde-aux-weight 0.01
```

### No Mask Loss

Omit `--mask-loss` but keep the mask channel:

```powershell
.\.venv\Scripts\python.exe -m ml.train `
  --grid datasets\gridded\report1_id_200 `
  --out checkpoints\report1\id200_ablate_no_mask_loss `
  --epochs 10 --batch 4 --context 4 --horizon 1 --device cpu `
  --d-model 128 --heads 4 --layers 4 --patch 8 `
  --attention-type factorized --pos-encoding sinusoidal `
  --prediction-mode derivative --integrator euler `
  --use-derivatives --derivative-mode central --strides 1,2,4 `
  --use-mask-channel `
  --pde-aux-loss --pde-normalize --pde-log-transport `
  --pde-cont-loss huber --pde-aux-weight 0.01
```

### No Derivative Features

```powershell
.\.venv\Scripts\python.exe -m ml.train `
  --grid datasets\gridded\report1_id_200 `
  --out checkpoints\report1\id200_ablate_no_derivatives `
  --epochs 10 --batch 4 --context 4 --horizon 1 --device cpu `
  --d-model 128 --heads 4 --layers 4 --patch 8 `
  --attention-type factorized --pos-encoding sinusoidal `
  --prediction-mode derivative --integrator euler `
  --no-derivatives --strides 1,2,4 `
  --use-mask-channel --mask-loss `
  --pde-aux-loss --pde-normalize --pde-log-transport `
  --pde-cont-loss huber --pde-aux-weight 0.01
```

## Evaluation Commands

The evaluator loads most settings from the checkpoint. Pass `--context`,
`--horizon`, `--batch`, and `--rollout-steps` explicitly for clarity.

### Main Model On ID

```powershell
.\.venv\Scripts\python.exe -m ml.evaluate `
  --grid datasets\gridded\report1_id_200 `
  --ckpt checkpoints\report1\id200_factorized_derivative_pde\best_model.pt `
  --out eval\report1\id200_on_id `
  --context 4 --horizon 1 --batch 4 --device cpu --rollout-steps 4,8,16
```

### Main Model On OOD-Mild

```powershell
.\.venv\Scripts\python.exe -m ml.evaluate `
  --grid datasets\gridded\report1_ood_mild_100 `
  --ckpt checkpoints\report1\id200_factorized_derivative_pde\best_model.pt `
  --out eval\report1\id200_on_ood_mild `
  --context 4 --horizon 1 --batch 4 --device cpu --rollout-steps 4,8,16
```

### Main Model On OOD-Hard

```powershell
.\.venv\Scripts\python.exe -m ml.evaluate `
  --grid datasets\gridded\report1_ood_hard_100 `
  --ckpt checkpoints\report1\id200_factorized_derivative_pde\best_model.pt `
  --out eval\report1\id200_on_ood_hard `
  --context 4 --horizon 1 --batch 4 --device cpu --rollout-steps 4,8,16
```

### Ablations On ID

```powershell
.\.venv\Scripts\python.exe -m ml.evaluate `
  --grid datasets\gridded\report1_id_200 `
  --ckpt checkpoints\report1\id200_ablate_global_attention\best_model.pt `
  --out eval\report1\ablate_global_on_id `
  --context 4 --horizon 1 --batch 4 --device cpu --rollout-steps 4,8,16
```

Repeat with the relevant checkpoint path and output directory for each
ablation.

Expected evaluation outputs:

- `metrics.json`: one-step MSE, MAE, relative L2, per-channel metrics,
  grouped-by-stride and grouped-by-family metrics, plus rollout metrics when
  enough frames exist.
- `pde_metrics.json`: PDE-identification metrics when the checkpoint has a PDE
  auxiliary head and the grid files have `pde_vec`.

## Plotting And Report Tables

### Prediction vs Ground Truth Fields

```powershell
.\.venv\Scripts\python.exe -m ml.plot_predictions `
  --grid datasets\gridded\report1_id_200 `
  --ckpt checkpoints\report1\id200_factorized_derivative_pde\best_model.pt `
  --out figures\report1\id200_predictions `
  --context 4 --horizon 1 --stride 1 --num-examples 4 --device cpu
```

This produces PNG comparisons for fields such as `rho`, velocity magnitude,
prediction, target, and absolute error maps.

### PDE Metrics, Confusion Matrices, And Rollout Figures

```powershell
.\.venv\Scripts\python.exe -m ml.plot_report1_metrics `
  --metrics eval\report1\id200_on_id\metrics.json `
  --out figures\report1\id200_on_id_metrics
```

For grouped ID/OOD comparisons:

```powershell
.\.venv\Scripts\python.exe -m ml.plot_report1_metrics `
  --metrics-list id:eval\report1\id200_on_id\metrics.json,ood_mild:eval\report1\id200_on_ood_mild\metrics.json,ood_hard:eval\report1\id200_on_ood_hard\metrics.json `
  --out figures\report1\id_vs_ood_grouped
```

Supported outputs include PDE R2/MAE bars, viscosity-law confusion, EOS
confusion, grouped law/EOS accuracy, grouped mean R2, and rollout MSE when the
corresponding metrics are present.

### Context Scaling

Use this for the Report 1 question of whether more temporal context improves
next-state prediction or PDE identification:

```powershell
.\.venv\Scripts\python.exe -m ml.evaluate_context_scaling `
  --grid datasets\gridded\report1_id_200 `
  --ckpt checkpoints\report1\id200_factorized_derivative_pde\best_model.pt `
  --out eval\report1\context_scaling_id200 `
  --contexts 2 4 8 16 --device cpu --batch 4 --rollout-steps 4,8

.\.venv\Scripts\python.exe -m ml.plot_report1_metrics `
  --context-scaling eval\report1\context_scaling_id200\context_scaling_metrics.json `
  --out figures\report1\context_scaling_id200
```

If the checkpoint uses `learned_absolute` positional encoding and a requested
context exceeds the trained maximum, the evaluator skips that context. Use
`--pos-encoding sinusoidal` during training for variable-context experiments.

### Training And Validation Loss

Training writes:

```text
checkpoints/report1/<run>/history.json
checkpoints/report1/<run>/metrics.json
```

There is currently no dedicated CLI for training-loss curves. Use
`history.json` directly for Report 1 tables or plot it in a notebook/script.
Do not invent a new plotting command in the report without adding and testing
the script first.

## Optional Slurm/HPC Notes

Existing CSD3 templates live in `scripts/`, including:

- `scripts/csd3_sweep_fixed_id20.slurm`
- `scripts/csd3_sweep_fixed_ood_mild.slurm`
- `scripts/csd3_sweep_fixed_ood_hard.slurm`
- `scripts/csd3_train_id200_factorized.slurm`
- `scripts/csd3_eval_id200_factorized.slurm`
- `scripts/csd3_plot_id200_factorized.slurm`

They use `#SBATCH -A YOUR_PROJECT` placeholders and assume:

```bash
cd ~/rds/hpc-work/fvm_solver_dataset
source .venv/bin/activate
```

Some existing templates still use older folder names such as
`datasets/raw_fixed_id20_csd3` and `datasets/grid_fixed_id200_csd3`. For Report
1 consistency, either edit copied scripts to use the folder layout in this
document or keep the CSD3 outputs clearly separate and record the mapping.

## Final Checklist

- Run all smoke tests before a large sweep.
- Start with `report1_id_20` before `report1_id_200` or `report1_id_1000`.
- Keep datasets generated before the EOS correction separate from new Report 1
  datasets.
- Do not mix old 16D and new 17D gridded datasets in one training directory.
- Always use `--geometry-mode fixed_ellipse` for stable CSD3 dataset generation
  unless you intentionally want random-geometry stress tests.
- Always use `--use-mask-channel --mask-loss` for ellipse/obstacle datasets
  unless running a mask ablation.
- Use `--attention-type factorized` for the main scalable model and
  `--attention-type global` only as a baseline/ablation.
- ID-only law/EOS classification accuracy is degenerate. Use mixed/OOD data for
  meaningful PDE-form identification claims.
- Generated data, checkpoints, eval outputs, and figures must not be committed.
