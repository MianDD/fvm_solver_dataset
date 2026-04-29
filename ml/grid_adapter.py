"""Adapter: unstructured FVM snapshots --> regular-grid arrays.

The supervisor's solver produces snapshots on a triangular mesh whose
node count varies between simulations (because each run's mesh is
re-generated with different random ellipses). Foundation-model
architectures based on patch-encoders (ViT / FLUID-LLM / Walrus) need
**fixed-shape** inputs, so we resample the cell-centred primitives
(``[V_x, V_y, rho, T]``) onto a regular Cartesian grid by linear
interpolation, falling back to nearest-neighbour outside the convex hull.

This module also deals with the supervisor's storage convention:
snapshots are saved in fp16 and z-score-normalised per-snapshot, so we
de-normalise here before interpolating.

Output of :func:`assemble_dataset` for a sweep folder is a single
``.npz`` per sim with the interpolated tensor of shape
``(T, 4, H, W)`` plus a ``pde_vec`` fingerprint for diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


# --------------------------------------------------------------------------
# Per-snapshot loader (handles the fp16 / mean / std encoding)
# --------------------------------------------------------------------------
def load_step(path: str | Path) -> Tuple[float, np.ndarray]:
    """Load one ``t_*.npz`` and return ``(t, cell_primitives)``.

    cell_primitives has shape ``(n_cells, 4)`` in physical units
    ``[V_x, V_y, rho, T]``.
    """
    z = np.load(path)
    mean = z["prim_mean"].astype(np.float32)
    std = z["prim_std"].astype(np.float32)
    cells = z["cell_primatives"].astype(np.float32) * std + mean    # (n_cells, 4)
    return float(z["t"]), cells


def load_mesh(path: str | Path) -> Dict:
    """Load ``mesh_props.npz`` and return a dict of arrays."""
    z = np.load(path)
    return {
        "vertices": z["vertices"],         # (n_vert, 2)
        "triangles": z["triangles"],       # (n_cells, 3)
        "centroids": z["centroids"],       # (n_cells, 2)
        "edges": z["edges"],
        "bc_edge_mask": z["bc_edge_masK"],
        "bc_type_str": z["bc_type_str"],
    }


# --------------------------------------------------------------------------
# Per-simulation interpolator
# --------------------------------------------------------------------------
class GridInterpolator:
    """Cache-friendly: triangulation is built once, then re-used for every
    snapshot of the simulation.
    """

    def __init__(self, centroids: np.ndarray, grid_H: int, grid_W: int,
                 bbox: Tuple[float, float, float, float] | None = None):
        if bbox is None:
            xmin, ymin = centroids.min(axis=0)
            xmax, ymax = centroids.max(axis=0)
            pad = 0.01 * max(xmax - xmin, ymax - ymin)
            bbox = (xmin - pad, ymin - pad, xmax + pad, ymax + pad)
        self.bbox = bbox
        self.H = grid_H
        self.W = grid_W

        xs = np.linspace(bbox[0], bbox[2], grid_W)
        ys = np.linspace(bbox[1], bbox[3], grid_H)
        X, Y = np.meshgrid(xs, ys, indexing="xy")           # (H, W)
        self.grid_xy = np.stack([X.ravel(), Y.ravel()], axis=1)
        self.X, self.Y = X, Y
        self.centroids = centroids
        # Triangulation built lazily on first call (it's expensive)
        self._lin = None
        self._near = None

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """values: (n_cells, n_components) -> (n_components, H, W) float32."""
        values = np.asarray(values, dtype=np.float64)
        if self._lin is None:
            # Both interpolators reuse the same Delaunay triangulation
            self._lin = LinearNDInterpolator(self.centroids.astype(np.float64), values)
            self._near = NearestNDInterpolator(self.centroids.astype(np.float64), values)
        else:
            self._lin.values = values
            self._near.values = values
        out_lin = self._lin(self.grid_xy)
        # Fill NaNs (outside the convex hull) with nearest-neighbour values
        nan_mask = np.isnan(out_lin)
        if np.any(nan_mask):
            out_lin[nan_mask] = self._near(self.grid_xy)[nan_mask]
        out = out_lin.reshape(self.H, self.W, -1).transpose(2, 0, 1)
        return out.astype(np.float32)


# --------------------------------------------------------------------------
# Sim-level conversion
# --------------------------------------------------------------------------
def convert_one_sim(sim_dir: Path, grid_H: int = 64, grid_W: int = 96
                    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Convert all snapshots in ``sim_dir`` to a regular grid.

    Returns
    -------
    snaps : (T, 4, H, W) float32
    times : (T,)        float32
    cfg   : dict (the per-sim config that the sweep saved)
    """
    mesh = load_mesh(sim_dir / "mesh_props.npz")
    interp = GridInterpolator(mesh["centroids"], grid_H=grid_H, grid_W=grid_W)

    t_files = sorted(p for p in sim_dir.iterdir()
                     if p.name.startswith("t_") and p.name.endswith(".npz"))
    t_files.sort(key=lambda p: float(p.name[len("t_"):-len(".npz")]))

    times: List[float] = []
    snaps: List[np.ndarray] = []
    for tf in t_files:
        t, cells = load_step(tf)
        grid = interp(cells)              # (4, H, W)
        snaps.append(grid)
        times.append(t)

    cfg_path = sim_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    return np.stack(snaps, axis=0), np.array(times, dtype=np.float32), cfg


def pde_fingerprint(cfg: Dict) -> np.ndarray:
    """Numerical fingerprint used only for diagnostics (NOT fed to model)."""
    return np.array([
        cfg.get("gamma", 0.0),
        cfg.get("viscosity", 0.0),
        cfg.get("visc_bulk", 0.0),
        cfg.get("thermal_cond", 0.0),
        cfg.get("C_v", 0.0),
        cfg.get("T_0", 0.0),
        cfg.get("rho_inf", 0.0),
        cfg.get("T_inf", 0.0),
        cfg.get("v_n_inf", 0.0),
    ], dtype=np.float32)


def timestep_files(sim_dir: Path) -> List[Path]:
    """Return sorted FVM timestep files in ``sim_dir``."""
    t_files = sorted(p for p in sim_dir.iterdir()
                     if p.name.startswith("t_") and p.name.endswith(".npz"))
    t_files.sort(key=lambda p: float(p.name[len("t_"):-len(".npz")]))
    return t_files


def simulation_skip_reason(sim_dir: Path) -> str | None:
    """Return a human-readable skip reason, or None if the sim is usable."""
    status_path = sim_dir / "status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return f"invalid status.json ({exc})"
        if status.get("status") != "success":
            failed_stage = status.get("failed_stage", "unknown")
            reason = status.get("reason", "no reason recorded")
            return f"status={status.get('status')} failed_stage={failed_stage} reason={reason}"

    if not (sim_dir / "mesh_props.npz").exists():
        return "no mesh_props.npz"
    if not timestep_files(sim_dir):
        return "no t_*.npz snapshots"
    return None


# --------------------------------------------------------------------------
# Sweep-level conversion
# --------------------------------------------------------------------------
def assemble_dataset(sweep_dir: str | Path, out_dir: str | Path,
                     grid_H: int = 64, grid_W: int = 96) -> List[str]:
    """Convert every successful sim in ``sweep_dir`` to a packed ``.npz``.

    The output directory ends up containing one ``sim_NNNN.npz`` per
    simulation, which is the format consumed by the ML training script.
    """
    sweep_dir = Path(sweep_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_dirs = sorted(p for p in sweep_dir.iterdir()
                      if p.is_dir() and p.name.startswith("sim_"))
    saved: List[str] = []
    for sd in sim_dirs:
        skip_reason = simulation_skip_reason(sd)
        if skip_reason is not None:
            print(f"  SKIP {sd.name}: {skip_reason}")
            continue
        snaps, times, cfg = convert_one_sim(sd, grid_H=grid_H, grid_W=grid_W)
        if not np.all(np.isfinite(snaps)):
            print(f"  SKIP {sd.name}: non-finite values after interp")
            continue
        out_path = out_dir / f"{sd.name}.npz"
        np.savez_compressed(
            out_path,
            snapshots=snaps.astype(np.float32),
            times=times,
            pde_vec=pde_fingerprint(cfg),
            cfg_json=json.dumps(cfg),
        )
        saved.append(str(out_path))
        print(f"  WRITE {out_path.name}: snaps={snaps.shape}  "
              f"gamma={cfg.get('gamma', 0):.2f}  mu={cfg.get('viscosity', 0):.1e}")
    if not saved:
        raise RuntimeError(
            f"No successful simulations found in {sweep_dir}. "
            "Check status.json, mesh_props.npz, and t_*.npz files."
        )
    return saved


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="sweep root directory")
    ap.add_argument("--out", required=True, help="output directory for .npz files")
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=96)
    args = ap.parse_args()
    try:
        saved = assemble_dataset(args.sweep, args.out, args.H, args.W)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(f"\nWrote {len(saved)} simulations to {args.out}")


if __name__ == "__main__":
    main()
