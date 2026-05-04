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
- `x_coords`, `y_coords`, `dx`, `dy`, and `bbox` for physical derivative features
- `metadata_json` with simulation id, seed, family, physical parameters, mesh
  parameters, time settings, validation status, and source folder
- `cfg_json` for backward-compatible access to the raw simulation config
- `pde_vec` for diagnostics

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
