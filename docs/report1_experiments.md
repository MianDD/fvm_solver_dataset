# Report 1 Experiment Workflow

This document is the executable workflow for the current Report 1 code state.
It assumes the repository is already patched with:

- corrected stiffened-gas EOS;
- bulk viscosity following `viscosity_law`;
- ID/OOD family sampling with 17D `pde_vec`;
- grid-mask neutralisation of non-fluid pixels;
- mask-aware training normalisation and raw binary mask channel;
- optional six-channel boundary-condition masks for ML inputs;
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

`id_visc_only`:

- EOS: ideal gas only
- `p_inf = 0`, `p_inf_ratio = 0`
- viscosity law: `constant`
- fixed state: `gamma=1.4`, `C_v=2.5`, `T_0=100`,
  `rho_inf=1`, `T_inf=100`, `v_n_inf=4`
- `visc_bulk = 0`
- `viscosity` is sampled log-uniformly over `[1e-3, 5e-3]`
- `thermal_cond = viscosity * cp / Pr`, with `cp = gamma * C_v`
  and `Pr = 0.71`
- intended use: focused ID study of how shear viscosity affects late-stage
  roll-up behind a fixed obstacle

`ood_visc_only`:

- same fixed ideal-gas state and fixed-Prandtl policy as `id_visc_only`
- `viscosity` is sampled log-uniformly over `[5e-3, 2e-2]`
- intended use: disjoint higher-viscosity OOD test for the focused roll-up
  study

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

New gridded files can also contain `boundary_mask` with shape `(6, H, W)`.
It is a one-hot raster used only by the ML input pipeline:

```text
0 = fluid_interior
1 = inlet
2 = outlet
3 = obstacle_wall
4 = channel_wall
5 = solid
```

The solver-facing boundary tags are unchanged. For fixed-ellipse raw data,
the sweep writes detailed mesh tags such as `Inlet`, `Outlet`,
`ObstacleWall`, and `ChannelWall`; otherwise the grid adapter falls back to a
geometric rule using the fluid mask and the grid coordinates. Enable these
channels with `--use-boundary-channels`. Input channels are ordered as:

```text
physical [V_x,V_y,rho,T]
then optional derivative channels
then optional boundary one-hot channels
then optional legacy binary fluid mask channel
```

Boundary channels are kept as raw `0/1` features: they are not z-score
normalised and receive no training input noise. Old gridded datasets without
`boundary_mask` still load, but only a minimal fluid/solid default is available;
regenerate grids when you want inlet/outlet/wall information.

Two optional boundary-focused training refinements are available for the wall
roll-up experiments:

- `--boundary-loss-weight 0.5` adds an extra prediction loss over inlet,
  outlet, obstacle-wall, and channel-wall fluid pixels. Solid pixels are never
  included.
- `--boundary-aware-refine` lets the post-patch CNN refinement head see the
  final context-step six-channel boundary mask by concatenating it with the
  decoded update before refinement.

Both are off by default. Use them as controlled ablations rather than silently
mixing them into every baseline.

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
.\.venv\Scripts\python.exe -m ml.smoke_boundary_mask
.\.venv\Scripts\python.exe -m ml.smoke_plot_fields
.\.venv\Scripts\python.exe -m sweep.smoke_family_sampling
.\.venv\Scripts\python.exe -m sweep.smoke_visc_only_family
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
  --use-boundary-channels --use-mask-channel --mask-loss `
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

## Viscosity-Only Focused Experiments

These commands support the focused Report 1 feedback question:

```text
How does shear viscosity affect late-stage roll-up behind a fixed obstacle?
```

Use `--geometry-mode fixed_ellipse` so geometry is held fixed while only shear
viscosity varies. These families are low-Mach ideal-gas sweeps, so use
`--max-mach 5` as a stricter validation threshold than the general default.
For long trajectories with `--save-t 0.3`, the supervisor feedback is to train
and evaluate on the later roll-up stage rather than the first transient seconds.
Use `--t-start 2.0` in `ml.train`, `ml.evaluate`, and `ml.plot_predictions`;
this is approximately equivalent to `--start-offset 7` because `2.0 / 0.3`
rounds up to saved-frame index 7. Prefer `--t-start` when saved `times`
exist, and use `--start-offset 7` only for older gridded files without physical
times. Make sure the chosen `--t-start` matches the actual saved physical time
scale for the run.

### Viscosity-Only Smoke, N=3

```powershell
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw\report1_visc_only_smoke_id3 `
  --n 3 --family id_visc_only --geometry-mode fixed_ellipse --device cpu `
  --n-iter 80 --save-t 0.0025 --dt 5e-4 `
  --min-A 0.2 --max-A 0.4 --lnscale 3 `
  --max-mesh-retries 2 --mesh-attempt-timeout-s 30 --timeout-s 300 `
  --min-snapshots 4 --validate-physics --max-mach 5
```

### Viscosity-Only ID20 Debug

```powershell
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw\report1_visc_only_id_20 `
  --n 20 --family id_visc_only --geometry-mode fixed_ellipse --device cpu `
  --n-iter 500 --save-t 0.01 --dt 5e-4 `
  --min-A 0.2 --max-A 0.4 --lnscale 3 `
  --max-mesh-retries 2 --mesh-attempt-timeout-s 30 --timeout-s 900 `
  --min-snapshots 10 --validate-physics --max-mach 5
```

### Viscosity-Only ID200 Main

```powershell
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw\report1_visc_only_id_200 `
  --n 200 --family id_visc_only --geometry-mode fixed_ellipse --device cpu `
  --n-iter 500 --save-t 0.01 --dt 5e-4 `
  --min-A 0.2 --max-A 0.4 --lnscale 3 `
  --max-mesh-retries 2 --mesh-attempt-timeout-s 30 --timeout-s 900 `
  --min-snapshots 10 --validate-physics --max-mach 5
```

### Viscosity-Only OOD100

```powershell
.\.venv\Scripts\python.exe -m sweep.sweep_fvm --out datasets\raw\report1_visc_only_ood_100 `
  --n 100 --family ood_visc_only --geometry-mode fixed_ellipse --device cpu `
  --n-iter 500 --save-t 0.01 --dt 5e-4 `
  --min-A 0.2 --max-A 0.4 --lnscale 3 `
  --max-mesh-retries 2 --mesh-attempt-timeout-s 30 --timeout-s 900 `
  --min-snapshots 10 --validate-physics --max-mach 5
```

Grid conversion follows the same convention as the main datasets:

```powershell
.\.venv\Scripts\python.exe -m ml.grid_adapter --sweep datasets\raw\report1_visc_only_id_20 `
  --out datasets\gridded\report1_visc_only_id_20 --H 64 --W 96

.\.venv\Scripts\python.exe -m ml.grid_adapter --sweep datasets\raw\report1_visc_only_id_200 `
  --out datasets\gridded\report1_visc_only_id_200 --H 64 --W 96

.\.venv\Scripts\python.exe -m ml.grid_adapter --sweep datasets\raw\report1_visc_only_ood_100 `
  --out datasets\gridded\report1_visc_only_ood_100 --H 64 --W 96
```

Later-stage training/evaluation on long viscosity-only trajectories:

```powershell
.\.venv\Scripts\python.exe -m ml.train `
  --grid datasets\gridded\report1_visc_only_id_200 `
  --out checkpoints\report1\visc_only_id200_factorized_late `
  --epochs 10 --batch 4 --context 4 --horizon 1 --device cpu `
  --d-model 128 --heads 4 --layers 4 --patch 8 `
  --attention-type factorized --pos-encoding sinusoidal `
  --prediction-mode derivative --integrator euler `
  --use-derivatives --use-boundary-channels --use-mask-channel --mask-loss `
  --pde-aux-loss --pde-normalize --pde-log-transport `
  --pde-cont-loss huber --t-start 2.0

.\.venv\Scripts\python.exe -m ml.evaluate `
  --grid datasets\gridded\report1_visc_only_id_200 `
  --ckpt checkpoints\report1\visc_only_id200_factorized_late\best_model.pt `
  --out eval\report1\visc_only_id200_late `
  --context 4 --horizon 1 --batch 4 --device cpu --t-start 2.0

.\.venv\Scripts\python.exe -m ml.plot_predictions `
  --grid datasets\gridded\report1_visc_only_id_200 `
  --ckpt checkpoints\report1\visc_only_id200_factorized_late\best_model.pt `
  --out figures\report1\predictions\visc_only_id200_late `
  --context 4 --horizon 1 --stride 1 --num-examples 4 --device cpu --t-start 2.0
```

### Viscosity-Only Boundary Variants

Use the same ID/OOD grids and late-stage filtering so the comparison isolates
the boundary-aware choices:

A. Baseline, no detailed boundary channels:

```powershell
.\.venv\Scripts\python.exe -m ml.train `
  --grid datasets\gridded\report1_visc_only_id_200 `
  --out checkpoints\report1\visc_only_A_baseline `
  --epochs 10 --batch 4 --context 4 --horizon 1 --device cpu `
  --d-model 128 --heads 4 --layers 4 --patch 8 `
  --attention-type factorized --pos-encoding sinusoidal `
  --prediction-mode derivative --integrator euler `
  --use-derivatives --use-mask-channel --mask-loss `
  --pde-aux-loss --pde-normalize --pde-log-transport `
  --pde-cont-loss huber --t-start 2.0
```

B. Add boundary one-hot channels:

```powershell
.\.venv\Scripts\python.exe -m ml.train `
  --grid datasets\gridded\report1_visc_only_id_200 `
  --out checkpoints\report1\visc_only_B_boundary_channels `
  --epochs 10 --batch 4 --context 4 --horizon 1 --device cpu `
  --d-model 128 --heads 4 --layers 4 --patch 8 `
  --attention-type factorized --pos-encoding sinusoidal `
  --prediction-mode derivative --integrator euler `
  --use-derivatives --use-boundary-channels --use-mask-channel --mask-loss `
  --pde-aux-loss --pde-normalize --pde-log-transport `
  --pde-cont-loss huber --t-start 2.0
```

C. Add boundary-weighted loss:

```powershell
.\.venv\Scripts\python.exe -m ml.train `
  --grid datasets\gridded\report1_visc_only_id_200 `
  --out checkpoints\report1\visc_only_C_boundary_loss `
  --epochs 10 --batch 4 --context 4 --horizon 1 --device cpu `
  --d-model 128 --heads 4 --layers 4 --patch 8 `
  --attention-type factorized --pos-encoding sinusoidal `
  --prediction-mode derivative --integrator euler `
  --use-derivatives --use-boundary-channels --use-mask-channel --mask-loss `
  --boundary-loss-weight 0.5 `
  --pde-aux-loss --pde-normalize --pde-log-transport `
  --pde-cont-loss huber --t-start 2.0
```

D. Add boundary-aware refinement:

```powershell
.\.venv\Scripts\python.exe -m ml.train `
  --grid datasets\gridded\report1_visc_only_id_200 `
  --out checkpoints\report1\visc_only_D_boundary_refine `
  --epochs 10 --batch 4 --context 4 --horizon 1 --device cpu `
  --d-model 128 --heads 4 --layers 4 --patch 8 `
  --attention-type factorized --pos-encoding sinusoidal `
  --prediction-mode derivative --integrator euler `
  --use-derivatives --use-boundary-channels --use-mask-channel --mask-loss `
  --boundary-loss-weight 0.5 --boundary-aware-refine `
  --pde-aux-loss --pde-normalize --pde-log-transport `
  --pde-cont-loss huber --t-start 2.0
```

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

New experiments use a deeper post-patch CNN refinement head after the
Transformer patch decoder. It is still a post-processing refinement of the
predicted update field, not a CNN stem before attention. Checkpoints created
before this refinement-head change may not load because the `refine` layer
shapes differ; keep old and new experiment folders separate.

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
  --use-boundary-channels --use-mask-channel --mask-loss `
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
  --use-boundary-channels --use-mask-channel --mask-loss `
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

### Collect Report 1 Figures And CSV Tables

After training and evaluation runs have written `history.json` and
`metrics.json`, collect the report-ready summary files with:

```powershell
.\.venv\Scripts\python.exe -m ml.report1_collect_results `
  --runs checkpoints\report1 `
  --eval-root eval\report1 `
  --out-figures figures\report1 `
  --out-tables tables\report1
```

This command is robust to incomplete experiment sets: if a run or metric is
missing, it prints a warning and writes the tables/figures that can be built
from the available JSON files.

Expected CSV outputs when data exists:

```text
tables/report1/main_results.csv
tables/report1/ablation_results.csv
tables/report1/pde_aux_metrics.csv
tables/report1/training_history_summary.csv
```

Expected figure outputs when data exists:

```text
figures/report1/loss_curves.png
figures/report1/id_vs_ood_mse.png
figures/report1/channel_mse.png
figures/report1/pde_aux_metrics.png
```

`main_results.csv` and `id_vs_ood_mse.png` are built from evaluation folders
whose names do not contain `ablat`. Folders containing `ablat` are collected in
`ablation_results.csv`, so use clear folder names such as
`ablate_global_on_id`.

### Prediction vs Ground Truth Fields

```powershell
.\.venv\Scripts\python.exe -m ml.plot_predictions `
  --grid datasets\gridded\report1_id_200 `
  --ckpt checkpoints\report1\id200_factorized_derivative_pde\best_model.pt `
  --out figures\report1\id200_predictions `
  --context 4 --horizon 1 --stride 1 --num-examples 4 --device cpu `
  --dpi 300 --fig-scale 1.25
```

This produces high-DPI PNG comparisons for context, target, prediction, and
absolute error. Omitting `--field` preserves the legacy `rho` and velocity
magnitude outputs. For Report 1 roll-up figures, prefer additional physical
fields:

```powershell
.\.venv\Scripts\python.exe -m ml.plot_predictions `
  --grid datasets\gridded\report1_visc_only_id_200 `
  --ckpt checkpoints\report1\visc_only_id200_factorized_late\best_model.pt `
  --out figures\report1\predictions\visc_only_selected_vorticity `
  --context 4 --horizon 1 --stride 1 --device cpu `
  --field vorticity --sim-ids "42,1157,1614" --t-start 2.0 `
  --max-plots 6 --dpi 350 --fig-scale 1.3 --save-pdf

.\.venv\Scripts\python.exe -m ml.plot_predictions `
  --grid datasets\gridded\report1_visc_only_id_200 `
  --ckpt checkpoints\report1\visc_only_id200_factorized_late\best_model.pt `
  --out figures\report1\predictions\visc_only_selected_schlieren `
  --context 4 --horizon 1 --stride 1 --device cpu `
  --field schlieren --sim-ids "42,1157,1614" --t-start 2.0 `
  --max-plots 6 --dpi 350 --fig-scale 1.3 --save-pdf
```

Supported prediction fields are `rho`, `T`, `V_x`, `V_y`, `vorticity`,
`grad_rho`, and `schlieren`. Vorticity and density-gradient/schlieren fields
are derived from primitive gridded states for visualisation only; they are not
training targets, losses, or evaluation metrics.

For pure saved-simulation figures without a model checkpoint, use
`ml.plot_fields`:

```powershell
.\.venv\Scripts\python.exe -m ml.plot_fields `
  --grid datasets\gridded\report1_visc_only_id_200 `
  --out figures\report1\fields\visc_only_vorticity `
  --field vorticity --sim-ids "42,1157,1614" --time-indices 7,10,15 `
  --dpi 350 --fig-scale 1.25 --save-pdf

.\.venv\Scripts\python.exe -m ml.plot_fields `
  --grid datasets\gridded\report1_visc_only_id_200 `
  --out figures\report1\fields\visc_only_schlieren_last `
  --field schlieren --sim-ids "42,1157,1614" --last `
  --dpi 350 --fig-scale 1.25 --save-pdf
```

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

Use the collector above to generate `figures/report1/loss_curves.png` and
`tables/report1/training_history_summary.csv` from those files. The raw
`history.json` remains the source of truth if you need custom notebook plots.

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
- Use `--use-boundary-channels --use-mask-channel --mask-loss` for the main
  ellipse/obstacle experiments unless running a boundary/mask ablation.
- Use `--attention-type factorized` for the main scalable model and
  `--attention-type global` only as a baseline/ablation.
- Do not mix pre-refinement-head checkpoints with new experiments; the deeper
  post-patch CNN changes `refine` layer shapes.
- ID-only law/EOS classification accuracy is degenerate. Use mixed/OOD data for
  meaningful PDE-form identification claims.
- Generated data, checkpoints, eval outputs, and figures must not be committed.
