"""Programmatic parameter-sweep wrapper around the supervisor's solver.

The supervisor's ``run_fvm.py`` runs **one** ``ConfigEllipse`` per execution.
This project, however, requires a **family** of PDE configurations, because
the foundation model is trained to identify the dynamics from the observed
context history (in-context learning over CFD). This script samples a
different physical configuration per simulation and groups the resulting
solver outputs under a single dataset directory.

Sampled physics (see ``FamilySpec``):

    gamma         — ratio of specific heats
    viscosity     — reference dynamic viscosity (Sutherland's mu_0)
    visc_bulk     — reference bulk viscosity
    thermal_cond  — thermal conductivity
    C_v           — specific heat at constant volume
    T_0           — reference temperature
    rho_inf, T_inf, v_n_inf  — inlet/outlet farfield state

Each simulation runs in its own Python sub-process to keep PyTorch caches
and the solver's mesh/edge/cell state isolated between runs.

Usage::

    python -m sweep.sweep_fvm --out datasets/family_v1 --n 32
    python -m sweep.sweep_fvm --out datasets/ood       --n  8 --ood
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


# --------------------------------------------------------------------------
# 1. The PDE-family sampler
# --------------------------------------------------------------------------
@dataclass
class FamilySpec:
    """Defines the parameter ranges for the in-distribution and OOD families.

    The OOD ranges are deliberately disjoint from the in-distribution ones
    on the parameters most likely to change the qualitative dynamics
    (gamma, viscosity, inflow speed). Other parameters share the same
    range to isolate which axis of variation matters.
    """
    gamma: Tuple[float, float] = (1.20, 1.45)
    viscosity: Tuple[float, float] = (5e-4, 5e-3)
    visc_bulk: Tuple[float, float] = (1e-3, 2e-2)
    thermal_cond: Tuple[float, float] = (1e-7, 1e-5)
    C_v: Tuple[float, float] = (1.5, 3.5)
    T_0: Tuple[float, float] = (80.0, 120.0)
    rho_inf: Tuple[float, float] = (0.8, 1.2)
    T_inf: Tuple[float, float] = (80.0, 120.0)
    v_n_inf: Tuple[float, float] = (3.0, 8.0)
    ood_gamma: Tuple[float, float] = (1.45, 1.65)
    ood_viscosity: Tuple[float, float] = (5e-3, 1e-2)
    ood_v_n_inf: Tuple[float, float] = (8.0, 12.0)

    def sample(self, rng: np.random.Generator, ood: bool = False) -> Dict:
        return dict(
            gamma=float(rng.uniform(*(self.ood_gamma if ood else self.gamma))),
            viscosity=float(rng.uniform(*(self.ood_viscosity if ood else self.viscosity))),
            visc_bulk=float(rng.uniform(*self.visc_bulk)),
            thermal_cond=float(rng.uniform(*self.thermal_cond)),
            C_v=float(rng.uniform(*self.C_v)),
            T_0=float(rng.uniform(*self.T_0)),
            rho_inf=float(rng.uniform(*self.rho_inf)),
            T_inf=float(rng.uniform(*self.T_inf)),
            v_n_inf=float(rng.uniform(*(self.ood_v_n_inf if ood else self.v_n_inf))),
        )


# --------------------------------------------------------------------------
# 2. Sub-process runner (string template baked at submit time)
# --------------------------------------------------------------------------
RUNNER_TEMPLATE = r"""
import os, sys, json, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **kw: None
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'time_fvm'))


def _run():
    with open(PARAMS_FILE) as f:
        params = json.load(f)

    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    from time_fvm.config_fvm import ConfigEllipse, ConfigNozzle
    from time_fvm.fvm_equation import FVMEquation, PhysicalSetup
    from time_fvm.run_fvm import generate_mesh, init_conds_ellipses, init_conds_nozzle
    from time_fvm.fvm_mesh import FVMMesh

    cfg_cls = ConfigEllipse if params['problem'] == 'ellipse' else ConfigNozzle
    cfg = cfg_cls()
    cfg.device = params['device']
    cfg.compile = params['compile']
    cfg.dt = params['dt']
    cfg.n_iter = params['n_iter']
    cfg.end_t = params['end_t']
    cfg.print_i = params['print_i']
    cfg.plot_t = 1e9                           # disable interactive plotting
    cfg.save_t = params['save_t']
    cfg.lnscale = params['lnscale']
    cfg.min_A = params['min_A']
    cfg.max_A = params['max_A']
    # PDE physics
    cfg.gamma = params['gamma']
    cfg.viscosity = params['viscosity']
    cfg.visc_bulk = params['visc_bulk']
    cfg.thermal_cond = params['thermal_cond']
    cfg.C_v = params['C_v']
    cfg.T_0 = params['T_0']
    cfg.inlet_cfg.rho_inf = params['rho_inf']
    cfg.inlet_cfg.T_inf = params['T_inf']
    cfg.inlet_cfg.v_n_inf = params['v_n_inf']
    cfg.exit_cfg.rho_inf = params['rho_inf']
    cfg.exit_cfg.T_inf = params['T_inf']
    cfg.exit_cfg.v_n_inf = -params['v_n_inf']

    phy = PhysicalSetup(cfg)
    prob = generate_mesh(cfg)
    Xs, tri_idx, all_edgs, bc_edge_mask, edge_tag, bound_edgs = prob
    mesh = FVMMesh(Xs, tri_idx, all_edgs, bc_edge_mask, device=cfg.device)
    init_fn = init_conds_ellipses if params['problem'] == 'ellipse' else init_conds_nozzle
    bc_tags, us_init = init_fn(mesh, edge_tag, bound_edgs, phy, cfg)

    # Re-route the saver's output into our chosen sim_NNNN/ directory
    import time_fvm.ds_generation.saving as saving
    _orig_init = saving.Saver.__init__
    def _patched_init(self, E_props, save_dir=None):
        return _orig_init(self, E_props, save_dir=params['out_dir'])
    saving.Saver.__init__ = _patched_init

    solver = FVMEquation(cfg, phy, mesh, cfg.N_comp, bc_tags, us_init=us_init)
    solver.solve()
    print('SIM_DONE', params['out_dir'])


if __name__ == '__main__':
    # Required: mesh_gen/create_mesh.py uses multiprocessing 'spawn'
    _run()
"""


def run_one_sim(repo_root: Path, out_dir: Path, params: Dict,
                python_exe: str = sys.executable) -> bool:
    """Run a single simulation in a fresh Python sub-process."""
    out_dir.mkdir(parents=True, exist_ok=True)
    params_file = out_dir / "config.json"
    params_file.write_text(json.dumps(params, indent=2))

    runner = (RUNNER_TEMPLATE
              .replace("REPO_ROOT", repr(str(repo_root.resolve())))
              .replace("PARAMS_FILE", repr(str(params_file.resolve()))))
    runner_path = out_dir / "_runner.py"
    runner_path.write_text(runner)

    t0 = time.time()
    try:
        proc = subprocess.run(
            [python_exe, str(runner_path)],
            cwd=str(repo_root),
            timeout=params.get("timeout_s", 1800),
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        print(f"  [{out_dir.name}] TIMEOUT")
        return False

    elapsed = time.time() - t0
    (out_dir / "run.log").write_text(
        f"--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}\n"
        f"--- elapsed: {elapsed:.1f}s    return code: {proc.returncode} ---\n"
    )

    has_snapshots = any(p.name.startswith("t_") for p in out_dir.iterdir())
    success = proc.returncode == 0 and "SIM_DONE" in proc.stdout and has_snapshots
    print(f"  [{out_dir.name}] {'OK' if success else 'FAIL'} ({elapsed:.1f}s)  "
          f"gamma={params['gamma']:.2f}  mu={params['viscosity']:.2e}  "
          f"v={params['v_n_inf']:.1f}")
    return success


# --------------------------------------------------------------------------
# 3. Main sweep
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".",
                    help="path to the repo root (default: current directory)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ood", action="store_true",
                    help="sample from the held-out OOD parameter ranges")
    ap.add_argument("--problem", choices=["ellipse", "nozzle"], default="ellipse")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n-iter", type=int, default=2500)
    ap.add_argument("--end-t", type=float, default=2.0)
    ap.add_argument("--save-t", type=float, default=0.05)
    ap.add_argument("--dt", type=float, default=5e-4)
    ap.add_argument("--lnscale", type=float, default=4.0)
    ap.add_argument("--min-A", type=float, default=2e-3)
    ap.add_argument("--max-A", type=float, default=4e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=1800)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Apply the future-annotations patch idempotently before launching
    sys.path.insert(0, str(repo / "scripts"))
    try:
        from patch_fvm_solver import main as _patch_main
        print(f"Applying compatibility patch to {repo}")
        _patch_main(str(repo))
    except ImportError:
        print("WARNING: scripts/patch_fvm_solver.py not found - skipping patch.")

    family = FamilySpec()
    rng = np.random.default_rng(args.seed)

    print(f"Sweep: n={args.n}  ood={args.ood}  problem={args.problem}")
    print(f"Repo : {repo}\nOut  : {out_root}")

    n_ok = 0
    for i in range(args.n):
        physics = family.sample(rng, ood=args.ood)
        params = {
            "sim_id": i,
            "seed": int(args.seed + 1000 * (10 if args.ood else 1) + i),
            "problem": args.problem,
            "device": args.device,
            "compile": bool(args.compile),
            "dt": args.dt, "n_iter": args.n_iter, "end_t": args.end_t,
            "save_t": args.save_t, "print_i": max(50, args.n_iter // 10),
            "lnscale": args.lnscale, "min_A": args.min_A, "max_A": args.max_A,
            "out_dir": str(out_root / f"sim_{i:04d}"),
            "timeout_s": args.timeout_s,
            **physics,
        }
        if run_one_sim(repo, Path(params["out_dir"]), params):
            n_ok += 1

    manifest = {
        "n_sims": args.n, "n_ok": n_ok, "ood": args.ood,
        "problem": args.problem, "seed": args.seed,
        "family_spec": asdict(family), "out_root": str(out_root),
    }
    (out_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone: {n_ok}/{args.n} simulations succeeded.")
    print(f"Manifest at {out_root / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
