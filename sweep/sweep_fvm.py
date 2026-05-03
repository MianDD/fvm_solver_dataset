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
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np


# --------------------------------------------------------------------------
# 1. The PDE-family sampler
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FamilySpec:
    """Controlled physical/PDE parameter family for one sweep split."""

    name: str
    description: str
    physical: Mapping[str, Tuple[float, float]]
    mesh: Mapping[str, Tuple[float, float]]

    def sample_physics(self, rng: np.random.Generator) -> Dict:
        return {k: float(rng.uniform(*v)) for k, v in self.physical.items()}

    def sample_mesh(self, rng: np.random.Generator) -> Dict:
        return {k: float(rng.uniform(*v)) for k, v in self.mesh.items()}


FAMILY_SPECS: Dict[str, FamilySpec] = {
    "id": FamilySpec(
        name="id",
        description=(
            "In-distribution ideal-gas compressible Navier-Stokes family. "
            "Ranges are narrow enough for stable smoke/CSD3 sweeps while still "
            "varying gas constants, transport coefficients, and inlet state."
        ),
        physical={
            "gamma": (1.22, 1.36),
            "viscosity": (8e-4, 3e-3),
            "visc_bulk": (2e-3, 1.2e-2),
            "thermal_cond": (5e-7, 5e-6),
            "C_v": (1.8, 2.8),
            "T_0": (90.0, 110.0),
            "rho_inf": (0.9, 1.1),
            "T_inf": (90.0, 110.0),
            "v_n_inf": (4.0, 7.0),
        },
        mesh={
            "lnscale": (4.0, 6.0),
            "min_A": (0.02, 0.04),
            "max_A": (0.04, 0.08),
        },
    ),
    "ood_mild": FamilySpec(
        name="ood_mild",
        description=(
            "Mild OOD family adjacent to ID: slightly higher gamma, viscosity, "
            "thermal conductivity, and inflow speed without pushing far into "
            "known unstable regimes."
        ),
        physical={
            "gamma": (1.36, 1.45),
            "viscosity": (3e-3, 5e-3),
            "visc_bulk": (8e-3, 2e-2),
            "thermal_cond": (4e-6, 9e-6),
            "C_v": (1.6, 3.0),
            "T_0": (85.0, 115.0),
            "rho_inf": (0.85, 1.15),
            "T_inf": (85.0, 115.0),
            "v_n_inf": (7.0, 8.5),
        },
        mesh={
            "lnscale": (4.0, 6.5),
            "min_A": (0.02, 0.05),
            "max_A": (0.04, 0.10),
        },
    ),
    "ood_hard": FamilySpec(
        name="ood_hard",
        description=(
            "Harder OOD family for evaluation, not the first training target: "
            "larger gas-constant and transport shifts plus faster inflow. "
            "Use more retries and validation filtering."
        ),
        physical={
            "gamma": (1.45, 1.58),
            "viscosity": (5e-3, 8e-3),
            "visc_bulk": (1.5e-2, 3e-2),
            "thermal_cond": (8e-6, 1.5e-5),
            "C_v": (1.4, 3.2),
            "T_0": (80.0, 125.0),
            "rho_inf": (0.8, 1.2),
            "T_inf": (80.0, 125.0),
            "v_n_inf": (8.5, 10.0),
        },
        mesh={
            "lnscale": (4.5, 7.0),
            "min_A": (0.03, 0.06),
            "max_A": (0.06, 0.12),
        },
    ),
}


# --------------------------------------------------------------------------
# 2. Sub-process runner (string template baked at submit time)
# --------------------------------------------------------------------------
RUNNER_TEMPLATE = r"""
import os, sys, json, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **kw: None
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'time_fvm'))


KEY_STATUS_FIELDS = (
    "sim_id", "seed", "mesh_seed", "family", "problem", "lnscale", "min_A", "max_A",
    "gamma", "viscosity", "visc_bulk", "thermal_cond", "C_v",
    "T_0", "rho_inf", "T_inf", "v_n_inf",
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _log(stage, message):
    print(f"[{_now()}] {stage}: {message}", flush=True)


def _count_snapshots(out_dir):
    out_path = Path(out_dir)
    return len([p for p in out_path.iterdir()
                if p.name.startswith("t_") and p.name.endswith(".npz")])


def _status_record(params, status, failed_stage=None, invalid_stage=None,
                   reason=None, snapshots_saved=None, validation=None):
    record = {
        "status": status,
        "timestamp": _now(),
        "output_folder": params["out_dir"],
    }
    for key in KEY_STATUS_FIELDS:
        if key in params:
            record[key] = params[key]
    if failed_stage:
        record["failed_stage"] = failed_stage
    if invalid_stage:
        record["invalid_stage"] = invalid_stage
    if reason:
        record["reason"] = reason
    if snapshots_saved is not None:
        record["number_of_snapshots_saved"] = int(snapshots_saved)
    if validation is not None:
        record["validation"] = validation
    return record


def _write_status(params, status, failed_stage=None, invalid_stage=None,
                  reason=None, snapshots_saved=None, validation=None):
    out_dir = Path(params["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    record = _status_record(
        params, status, failed_stage, invalid_stage, reason,
        snapshots_saved, validation,
    )
    (out_dir / "status.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _configure_cfg(cfg, params):
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
    cfg.mesh_attempt_timeout_s = params.get('mesh_attempt_timeout_s', 10)
    cfg.mesh_worker_retries = 1

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


def _mesh_failure_reason(exc):
    text = str(exc).strip()
    if "timeout" in text.lower():
        return "mesh generation timeout"
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _snapshot_files(out_dir):
    out_path = Path(out_dir)
    files = sorted(p for p in out_path.iterdir()
                   if p.name.startswith("t_") and p.name.endswith(".npz"))
    files.sort(key=lambda p: float(p.name[len("t_"):-len(".npz")]))
    return files


def _load_primitives(path):
    z = np.load(path)
    mean = z["prim_mean"].astype(np.float32)
    std = z["prim_std"].astype(np.float32)
    cells = z["cell_primatives"].astype(np.float32) * std + mean
    return cells


def _validate_saved_snapshots(params):
    files = _snapshot_files(params["out_dir"])
    validation = {
        "number_of_snapshots_saved": len(files),
        "min_snapshots_required": int(params.get("min_snapshots", 1)),
        "validate_physics": bool(params.get("validate_physics", True)),
    }
    if len(files) < validation["min_snapshots_required"]:
        return False, "snapshot_validation", (
            f"only {len(files)} snapshots saved; "
            f"minimum is {validation['min_snapshots_required']}"
        ), validation

    if not params.get("validate_physics", True):
        return True, None, None, validation

    rho_min = float(params.get("rho_min", 1e-9))
    T_min = float(params.get("T_min", 1e-9))
    max_abs_value = float(params.get("max_abs_value", 1e6))
    validation.update({
        "rho_min": rho_min,
        "T_min": T_min,
        "max_abs_value": max_abs_value,
        "rho_observed_min": None,
        "T_observed_min": None,
        "max_abs_observed": None,
    })

    rho_obs = []
    T_obs = []
    abs_obs = []
    for path in files:
        cells = _load_primitives(path)
        if not np.all(np.isfinite(cells)):
            validation["bad_file"] = path.name
            return False, "snapshot_validation", f"non-finite values in {path.name}", validation
        rho = cells[:, 2]
        T = cells[:, 3]
        rho_obs.append(float(np.min(rho)))
        T_obs.append(float(np.min(T)))
        abs_obs.append(float(np.max(np.abs(cells))))

    validation["rho_observed_min"] = float(np.min(rho_obs))
    validation["T_observed_min"] = float(np.min(T_obs))
    validation["max_abs_observed"] = float(np.max(abs_obs))
    if validation["rho_observed_min"] <= rho_min:
        return False, "snapshot_validation", (
            f"density below lower bound: {validation['rho_observed_min']:.4g} <= {rho_min:.4g}"
        ), validation
    if validation["T_observed_min"] <= T_min:
        return False, "snapshot_validation", (
            f"temperature below lower bound: {validation['T_observed_min']:.4g} <= {T_min:.4g}"
        ), validation
    if validation["max_abs_observed"] > max_abs_value:
        return False, "snapshot_validation", (
            f"value explosion: max abs {validation['max_abs_observed']:.4g} > {max_abs_value:.4g}"
        ), validation
    return True, None, None, validation


def _run():
    params = None
    stage = "config_loading"
    try:
        _log(stage, f"loading {PARAMS_FILE}")
        with open(PARAMS_FILE) as f:
            params = json.load(f)
        Path(params["out_dir"]).mkdir(parents=True, exist_ok=True)
        _log(stage, "loaded")
        _log(
            "starting_simulation",
            f"sim_id={params['sim_id']} seed={params['seed']} problem={params['problem']}",
        )

        np.random.seed(params['seed'])
        torch.manual_seed(params['seed'])

        stage = "imports"
        from time_fvm.config_fvm import ConfigEllipse, ConfigNozzle
        from time_fvm.fvm_equation import FVMEquation, PhysicalSetup
        from time_fvm.run_fvm import generate_mesh, init_conds_ellipses, init_conds_nozzle
        from time_fvm.fvm_mesh import FVMMesh

        cfg_cls = ConfigEllipse if params['problem'] == 'ellipse' else ConfigNozzle
        cfg = cfg_cls()
        _configure_cfg(cfg, params)

        stage = "physical_setup"
        _log(stage, "start")
        phy = PhysicalSetup(cfg)
        _log(stage, "end")

        base_min_A = float(params["min_A"])
        base_max_A = float(params["max_A"])
        base_lnscale = float(params["lnscale"])
        max_mesh_retries = max(1, int(params.get("max_mesh_retries", 2)))
        mesh_attempt_timeout_s = float(params.get("mesh_attempt_timeout_s", 10))
        prob = None
        last_mesh_reason = "mesh generation timeout"

        for attempt in range(1, max_mesh_retries + 1):
            stage = "mesh_generation"
            retry_seed = int(params.get("mesh_seed", params["seed"])) + attempt - 1
            coarsen = 1.5 ** (attempt - 1)
            np.random.seed(retry_seed)
            torch.manual_seed(retry_seed)
            cfg.min_A = base_min_A * coarsen
            cfg.max_A = base_max_A * coarsen
            cfg.lnscale = base_lnscale
            cfg.mesh_worker_retries = 1
            cfg.mesh_attempt_timeout_s = mesh_attempt_timeout_s

            _log(
                "mesh_generation_start",
                "attempt={}/{} seed={} min_A={:.4g} max_A={:.4g} lnscale={:.4g}".format(
                    attempt, max_mesh_retries, retry_seed,
                    cfg.min_A, cfg.max_A, cfg.lnscale,
                ),
            )
            try:
                prob = generate_mesh(cfg)
                Xs, tri_idx, all_edgs, bc_edge_mask, edge_tag, bound_edgs = prob
                _log(
                    "mesh_generation_success",
                    f"attempt={attempt} cells={len(tri_idx)} edges={len(all_edgs)}",
                )
                break
            except Exception as exc:
                last_mesh_reason = _mesh_failure_reason(exc)
                _log(
                    "mesh_generation_failure",
                    f"attempt={attempt}/{max_mesh_retries} reason={last_mesh_reason}",
                )

        if prob is None:
            _write_status(
                params,
                "failed",
                failed_stage="mesh_generation",
                reason=last_mesh_reason,
                snapshots_saved=0,
            )
            _log("final_simulation_status", f"failed stage=mesh_generation reason={last_mesh_reason}")
            print("SIM_FAILED", params["out_dir"], flush=True)
            return 2

        stage = "fvm_mesh_construction"
        _log(stage, "start")
        mesh = FVMMesh(Xs, tri_idx, all_edgs, bc_edge_mask, device=cfg.device)
        _log(stage, f"end cells={mesh.n_cells} edges={mesh.n_edges}")

        stage = "initial_condition_setup"
        _log(stage, "start")
        init_fn = init_conds_ellipses if params['problem'] == 'ellipse' else init_conds_nozzle
        bc_tags, us_init = init_fn(mesh, edge_tag, bound_edgs, phy, cfg)
        _log(stage, f"end boundary_edges={len(bc_tags)}")

        # Re-route the saver's output into our chosen sim_NNNN/ directory.
        import time_fvm.ds_generation.saving as saving
        _orig_init = saving.Saver.__init__
        def _patched_init(self, E_props, save_dir=None):
            return _orig_init(self, E_props, save_dir=params['out_dir'])
        saving.Saver.__init__ = _patched_init

        stage = "solver_construction"
        _log(stage, "start")
        solver = FVMEquation(cfg, phy, mesh, cfg.N_comp, bc_tags, us_init=us_init)
        _log(stage, "end")

        stage = "time_stepping"
        _log(stage, "start")
        solver.solve()
        _log(stage, "end")

        snapshots = _count_snapshots(params["out_dir"])
        _log("snapshot_saving_summary", f"snapshots_saved={snapshots}")
        valid, invalid_stage, reason, validation = _validate_saved_snapshots(params)
        if not valid:
            _write_status(
                params,
                "invalid",
                invalid_stage=invalid_stage,
                reason=reason,
                snapshots_saved=snapshots,
                validation=validation,
            )
            _log("final_simulation_status", f"invalid stage={invalid_stage} reason={reason}")
            print("SIM_INVALID", params["out_dir"], flush=True)
            return 3

        _write_status(params, "success", snapshots_saved=snapshots, validation=validation)
        _log("final_simulation_status", "success")
        print('SIM_DONE', params['out_dir'], flush=True)
        return 0
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        if params is not None:
            try:
                snapshots = _count_snapshots(params["out_dir"])
                _write_status(
                    params,
                    "failed",
                    failed_stage=stage,
                    reason=reason,
                    snapshots_saved=snapshots,
                )
            except Exception as status_exc:
                print(f"Failed to write status.json: {status_exc}", flush=True)
        _log("final_simulation_status", f"failed stage={stage} reason={reason}")
        if params is not None:
            print("SIM_FAILED", params["out_dir"], flush=True)
        return 1


if __name__ == '__main__':
    # Required: mesh_gen/create_mesh.py uses multiprocessing 'spawn'
    sys.exit(_run())
"""


MANIFEST_PARAM_KEYS = (
    "family", "mesh_seed",
    "gamma", "viscosity", "visc_bulk", "thermal_cond", "C_v",
    "T_0", "rho_inf", "T_inf", "v_n_inf", "lnscale", "min_A", "max_A",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _count_snapshots(out_dir: Path) -> int:
    return len([p for p in out_dir.iterdir()
                if p.name.startswith("t_") and p.name.endswith(".npz")])


def _decode_timeout_stream(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _write_status(out_dir: Path, params: Dict, status: str, *,
                  failed_stage: str | None = None,
                  reason: str | None = None,
                  snapshots_saved: int | None = None) -> Dict:
    record = {
        "status": status,
        "timestamp": _utc_now(),
        "output_folder": str(out_dir),
        "sim_id": params.get("sim_id"),
        "seed": params.get("seed"),
        "mesh_seed": params.get("mesh_seed"),
        "family": params.get("family"),
        "problem": params.get("problem"),
        "lnscale": params.get("lnscale"),
        "min_A": params.get("min_A"),
        "max_A": params.get("max_A"),
        "gamma": params.get("gamma"),
        "viscosity": params.get("viscosity"),
        "visc_bulk": params.get("visc_bulk"),
        "thermal_cond": params.get("thermal_cond"),
        "C_v": params.get("C_v"),
        "T_0": params.get("T_0"),
        "rho_inf": params.get("rho_inf"),
        "T_inf": params.get("T_inf"),
        "v_n_inf": params.get("v_n_inf"),
    }
    if failed_stage:
        record["failed_stage"] = failed_stage
    if reason:
        record["reason"] = reason
    if snapshots_saved is not None:
        record["number_of_snapshots_saved"] = int(snapshots_saved)
    (out_dir / "status.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _read_status(out_dir: Path) -> Dict | None:
    status_path = out_dir / "status.json"
    if not status_path.exists():
        return None
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": "failed",
            "failed_stage": "status_read",
            "reason": f"invalid status.json: {exc}",
        }


def _manifest_record(params: Dict, out_dir: Path, status_record: Dict,
                     elapsed_s: float | None = None,
                     return_code: int | None = None) -> Dict:
    rec = {
        "sim_id": params["sim_id"],
        "seed": params["seed"],
        "status": status_record.get("status", "failed"),
        "output_dir": str(out_dir),
    }
    for key in MANIFEST_PARAM_KEYS:
        rec[key] = params.get(key)
    if "reason" in status_record:
        if status_record.get("status") == "invalid":
            rec["invalid_reason"] = status_record["reason"]
        else:
            rec["failure_reason"] = status_record["reason"]
    if "failed_stage" in status_record:
        rec["failed_stage"] = status_record["failed_stage"]
    if "invalid_stage" in status_record:
        rec["invalid_stage"] = status_record["invalid_stage"]
    if "number_of_snapshots_saved" in status_record:
        rec["number_of_snapshots_saved"] = status_record["number_of_snapshots_saved"]
    if elapsed_s is not None:
        rec["elapsed_s"] = round(elapsed_s, 3)
    if return_code is not None:
        rec["return_code"] = return_code
    return rec


def run_one_sim(repo_root: Path, out_dir: Path, params: Dict,
                python_exe: str = sys.executable) -> Dict:
    """Run a single simulation in a fresh Python sub-process."""
    out_dir.mkdir(parents=True, exist_ok=True)
    params_file = out_dir / "config.json"
    params_file.write_text(json.dumps(params, indent=2), encoding="utf-8")

    runner = (RUNNER_TEMPLATE
              .replace("REPO_ROOT", repr(str(repo_root.resolve())))
              .replace("PARAMS_FILE", repr(str(params_file.resolve()))))
    runner_path = out_dir / "_runner.py"
    runner_path.write_text(runner, encoding="utf-8")

    print(
        f"  [{out_dir.name}] START sim_id={params['sim_id']} seed={params['seed']}",
        flush=True,
    )
    t0 = time.time()
    try:
        proc = subprocess.run(
            [python_exe, str(runner_path)],
            cwd=str(repo_root),
            timeout=params.get("timeout_s", 1800),
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - t0
        stdout = _decode_timeout_stream(exc.stdout)
        stderr = _decode_timeout_stream(exc.stderr)
        reason = f"simulation subprocess timeout after {params.get('timeout_s', 1800)}s"
        (out_dir / "run.log").write_text(
            f"--- STDOUT ---\n{stdout}\n"
            f"--- STDERR ---\n{stderr}\n"
            f"--- elapsed: {elapsed:.1f}s    return code: TIMEOUT ---\n",
            encoding="utf-8",
        )
        status = _write_status(
            out_dir,
            params,
            "failed",
            failed_stage="subprocess_timeout",
            reason=reason,
            snapshots_saved=_count_snapshots(out_dir),
        )
        print(f"  [{out_dir.name}] FAIL ({elapsed:.1f}s) {reason}", flush=True)
        return _manifest_record(params, out_dir, status, elapsed_s=elapsed)

    elapsed = time.time() - t0
    (out_dir / "run.log").write_text(
        f"--- STDOUT ---\n{proc.stdout}\n"
        f"--- STDERR ---\n{proc.stderr}\n"
        f"--- elapsed: {elapsed:.1f}s    return code: {proc.returncode} ---\n",
        encoding="utf-8",
    )

    status = _read_status(out_dir)
    if status is None:
        snapshots = _count_snapshots(out_dir)
        success = proc.returncode == 0 and "SIM_DONE" in proc.stdout and snapshots > 0
        status = _write_status(
            out_dir,
            params,
            "success" if success else "failed",
            failed_stage=None if success else "runner",
            reason=None if success else "runner exited without status.json",
            snapshots_saved=snapshots,
        )

    success = status.get("status") == "success" and proc.returncode == 0
    reason = status.get("reason", "")
    print(
        f"  [{out_dir.name}] {'OK' if success else 'FAIL'} ({elapsed:.1f}s)  "
        f"gamma={params['gamma']:.2f}  mu={params['viscosity']:.2e}  "
        f"v={params['v_n_inf']:.1f}"
        + (f"  reason={reason}" if reason and not success else ""),
        flush=True,
    )
    return _manifest_record(params, out_dir, status, elapsed_s=elapsed,
                            return_code=proc.returncode)


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
                    help="backward-compatible alias for --family ood_mild")
    ap.add_argument("--family", choices=sorted(FAMILY_SPECS), default="id",
                    help="controlled parameter family to sample")
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
    ap.add_argument("--max-mesh-retries", type=int, default=2,
                    help="full mesh-generation attempts per simulation")
    ap.add_argument("--mesh-attempt-timeout-s", type=float, default=10,
                    help="timeout for each mesh worker attempt")
    ap.add_argument("--sample-mesh-params", action="store_true",
                    help="sample lnscale/min_A/max_A from the selected family")
    ap.add_argument("--min-snapshots", type=int, default=1,
                    help="minimum saved snapshots required for a usable simulation")
    ap.add_argument("--validate-physics", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="validate finite values and positive rho/T before marking success")
    ap.add_argument("--rho-min", type=float, default=1e-9,
                    help="strict lower bound for density during validation")
    ap.add_argument("--T-min", type=float, default=1e-9,
                    help="strict lower bound for temperature during validation")
    ap.add_argument("--max-abs-value", type=float, default=1e6,
                    help="upper bound for absolute primitive values during validation")
    args = ap.parse_args()

    family_name = "ood_mild" if args.ood and args.family == "id" else args.family

    repo = Path(args.repo).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Apply the future-annotations patch idempotently before launching
    sys.path.insert(0, str(repo / "scripts"))
    try:
        from patch_fvm_solver import main as _patch_main
        print(f"Applying compatibility patch to {repo}", flush=True)
        _patch_main(str(repo))
    except ImportError:
        print("WARNING: scripts/patch_fvm_solver.py not found - skipping patch.", flush=True)

    family = FAMILY_SPECS[family_name]
    rng = np.random.default_rng(args.seed)

    print(
        f"Sweep: n={args.n}  family={family.name}  ood_flag={args.ood}  problem={args.problem}",
        flush=True,
    )
    print(f"Repo : {repo}\nOut  : {out_root}", flush=True)

    n_success = 0
    n_failed = 0
    n_invalid = 0
    sim_records = []
    for i in range(args.n):
        physics = family.sample_physics(rng)
        mesh_params = (
            family.sample_mesh(rng)
            if args.sample_mesh_params
            else {"lnscale": args.lnscale, "min_A": args.min_A, "max_A": args.max_A}
        )
        sim_seed = int(args.seed + 1000 * (10 if family.name != "id" else 1) + i)
        mesh_seed = sim_seed
        params = {
            "sim_id": i,
            "seed": sim_seed,
            "mesh_seed": mesh_seed,
            "family": family.name,
            "problem": args.problem,
            "device": args.device,
            "compile": bool(args.compile),
            "dt": args.dt, "n_iter": args.n_iter, "end_t": args.end_t,
            "save_t": args.save_t, "print_i": max(50, args.n_iter // 10),
            "lnscale": mesh_params["lnscale"],
            "min_A": mesh_params["min_A"],
            "max_A": mesh_params["max_A"],
            "out_dir": str(out_root / f"sim_{i:04d}"),
            "timeout_s": args.timeout_s,
            "max_mesh_retries": args.max_mesh_retries,
            "mesh_attempt_timeout_s": args.mesh_attempt_timeout_s,
            "min_snapshots": args.min_snapshots,
            "validate_physics": args.validate_physics,
            "rho_min": args.rho_min,
            "T_min": args.T_min,
            "max_abs_value": args.max_abs_value,
            **physics,
        }
        sim_record = run_one_sim(repo, Path(params["out_dir"]), params)
        sim_records.append(sim_record)
        if sim_record["status"] == "success":
            n_success += 1
        elif sim_record["status"] == "invalid":
            n_invalid += 1
        else:
            n_failed += 1

    manifest = {
        "out_root": str(out_root),
        "created_at": _utc_now(),
        "family": family.name,
        "family_spec": asdict(family),
        "number_requested": args.n,
        "n_sims": args.n,
        "n_success": n_success,
        "n_failed": n_failed,
        "n_invalid": n_invalid,
        "n_usable": n_success,
        "n_ok": n_success,
        "ood": args.ood,
        "problem": args.problem,
        "seed": args.seed,
        "max_mesh_retries": args.max_mesh_retries,
        "mesh_attempt_timeout_s": args.mesh_attempt_timeout_s,
        "timeout_s": args.timeout_s,
        "global_settings": {
            "dt": args.dt,
            "n_iter": args.n_iter,
            "save_t": args.save_t,
            "end_t": args.end_t,
            "device": args.device,
            "compile": bool(args.compile),
            "sample_mesh_params": args.sample_mesh_params,
            "base_mesh": {
                "lnscale": args.lnscale,
                "min_A": args.min_A,
                "max_A": args.max_A,
            },
            "min_snapshots": args.min_snapshots,
            "validate_physics": args.validate_physics,
            "rho_min": args.rho_min,
            "T_min": args.T_min,
            "max_abs_value": args.max_abs_value,
        },
        "simulations": sim_records,
    }
    (out_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"\nDone: success={n_success} failed={n_failed} invalid={n_invalid} requested={args.n}.",
        flush=True,
    )
    print(f"Manifest at {out_root / 'MANIFEST.json'}", flush=True)


if __name__ == "__main__":
    main()
