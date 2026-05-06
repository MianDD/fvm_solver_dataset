"""Adapter: unstructured FVM snapshots --> regular-grid arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

try:
    from .boundary import (
        BOUNDARY_CLASS_NAMES,
        BOUNDARY_MASK_VERSION,
        N_BOUNDARY_CLASSES,
    )
    from .pde import DEFAULT_PDE_VEC_NAMES, EOS_TYPE_NAMES, VISCOSITY_LAW_NAMES
except ImportError:  # pragma: no cover - allows direct ``python ml/grid_adapter.py``
    from boundary import BOUNDARY_CLASS_NAMES, BOUNDARY_MASK_VERSION, N_BOUNDARY_CLASSES
    from pde import DEFAULT_PDE_VEC_NAMES, EOS_TYPE_NAMES, VISCOSITY_LAW_NAMES


CHANNEL_NAMES = ["V_x", "V_y", "rho", "T"]
PHYSICAL_KEYS = [
    "gamma", "viscosity", "visc_bulk", "thermal_cond", "C_v",
    "T_0", "rho_inf", "T_inf", "v_n_inf",
    "viscosity_law", "power_law_n", "eos_type", "p_inf", "p_inf_ratio",
]
VISCOSITY_LAWS = list(VISCOSITY_LAW_NAMES)
EOS_TYPES = list(EOS_TYPE_NAMES)
MESH_KEYS = ["lnscale", "min_A", "max_A", "mesh_seed"]
TIME_KEYS = ["dt", "save_t", "n_iter", "end_t"]


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
    z = np.load(path, allow_pickle=True)
    mesh = {
        "vertices": z["vertices"],         # (n_vert, 2)
        "triangles": z["triangles"],       # (n_cells, 3)
        "centroids": z["centroids"],       # (n_cells, 2)
        "edges": z["edges"],
        "bc_edge_mask": z["bc_edge_masK"],
        "bc_type_str": z["bc_type_str"],
    }
    if "bc_type_detailed" in z:
        mesh["bc_type_detailed"] = z["bc_type_detailed"]
    return mesh


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
        self.x_coords = xs.astype(np.float32)
        self.y_coords = ys.astype(np.float32)
        self.dx = float(xs[1] - xs[0]) if grid_W > 1 else 1.0
        self.dy = float(ys[1] - ys[0]) if grid_H > 1 else 1.0
        X, Y = np.meshgrid(xs, ys, indexing="xy")           # (H, W)
        self.grid_xy = np.stack([X.ravel(), Y.ravel()], axis=1)
        self.X, self.Y = X, Y
        self.centroids = centroids
        # Triangulation built lazily on first call (it's expensive)
        self._lin = None
        self._near = None

    def mesh_mask(self, vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """Return a binary fluid mask on the regular grid.

        The mask is 1 for grid points lying inside at least one fluid triangle
        and 0 elsewhere.  This catches obstacle holes because those regions are
        not covered by the simulation triangles even though interpolation can
        still fill them via nearest neighbours.
        """
        points = self.grid_xy.astype(np.float64)
        fluid = np.zeros(points.shape[0], dtype=bool)
        verts = np.asarray(vertices, dtype=np.float64)
        tris = np.asarray(triangles, dtype=np.int64)
        eps = 1e-12
        for tri in tris:
            a, b, c = verts[tri]
            xmin = min(a[0], b[0], c[0]) - eps
            xmax = max(a[0], b[0], c[0]) + eps
            ymin = min(a[1], b[1], c[1]) - eps
            ymax = max(a[1], b[1], c[1]) + eps
            candidates = (
                (~fluid)
                & (points[:, 0] >= xmin) & (points[:, 0] <= xmax)
                & (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
            )
            if not np.any(candidates):
                continue
            p = points[candidates]
            v0 = c - a
            v1 = b - a
            v2 = p - a
            den = v0[0] * v1[1] - v1[0] * v0[1]
            if abs(den) < eps:
                continue
            u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / den
            v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / den
            inside = (u >= -eps) & (v >= -eps) & ((u + v) <= 1.0 + eps)
            idx = np.flatnonzero(candidates)
            fluid[idx[inside]] = True
        return fluid.reshape(self.H, self.W).astype(np.float32)

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


def _neighbour_dilation(mask: np.ndarray) -> np.ndarray:
    """One-cell 8-neighbour dilation implemented with NumPy slices."""
    src = np.asarray(mask, dtype=bool)
    out = src.copy()
    H, W = src.shape
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            if dj == 0 and di == 0:
                continue
            src_j0 = max(0, -dj)
            src_j1 = min(H, H - dj)
            src_i0 = max(0, -di)
            src_i1 = min(W, W - di)
            dst_j0 = max(0, dj)
            dst_j1 = min(H, H + dj)
            dst_i0 = max(0, di)
            dst_i1 = min(W, W + di)
            out[dst_j0:dst_j1, dst_i0:dst_i1] |= src[src_j0:src_j1, src_i0:src_i1]
    return out


def _outer_boundary_bands(interp: GridInterpolator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return left/right/channel-wall bands on the regular grid.

    Coordinate-aware thresholds mark roughly two grid layers.  The fallback is
    the corresponding two edge rows/columns, which keeps old grid files useful
    even when bbox/coords are unavailable.
    """
    H, W = int(interp.H), int(interp.W)
    left = np.zeros((H, W), dtype=bool)
    right = np.zeros((H, W), dtype=bool)
    channel = np.zeros((H, W), dtype=bool)
    x = getattr(interp, "x_coords", None)
    y = getattr(interp, "y_coords", None)
    dx = float(getattr(interp, "dx", 0.0) or 0.0)
    dy = float(getattr(interp, "dy", 0.0) or 0.0)
    if x is not None and len(x) >= 2 and dx > 0:
        x_arr = np.asarray(x, dtype=np.float32)
        left_cols = x_arr <= (float(x_arr[0]) + 1.5 * dx)
        right_cols = x_arr >= (float(x_arr[-1]) - 1.5 * dx)
        left[:, left_cols] = True
        right[:, right_cols] = True
    else:
        left[:, :min(2, W)] = True
        right[:, max(0, W - 2):] = True
    if y is not None and len(y) >= 2 and dy > 0:
        y_arr = np.asarray(y, dtype=np.float32)
        wall_rows = (y_arr <= (float(y_arr[0]) + 1.5 * dy)) | (
            y_arr >= (float(y_arr[-1]) - 1.5 * dy)
        )
        channel[wall_rows, :] = True
    else:
        channel[:min(2, H), :] = True
        channel[max(0, H - 2):, :] = True
    return left, right, channel


def _normalised_detail_tags(mesh: Dict[str, Any] | None) -> set[str]:
    if not mesh or "bc_type_detailed" not in mesh:
        return set()
    tags = mesh.get("bc_type_detailed")
    try:
        values = np.asarray(tags).ravel().tolist()
    except Exception:
        return set()
    return {str(v).lower() for v in values}


def compute_boundary_mask(mask: np.ndarray, interp: GridInterpolator,
                          mesh: Dict[str, Any] | None = None
                          ) -> Tuple[np.ndarray, List[str], str]:
    """Build one-hot boundary classes for ML input.

    The saved binary mask remains the source of truth for fluid/non-fluid.  If
    detailed mesh tags are present, they indicate which physical boundary types
    exist; rasterisation still uses robust grid geometry so old and new meshes
    share the same output convention.
    """
    fluid = np.asarray(mask, dtype=np.float32) > 0.5
    if fluid.shape != (int(interp.H), int(interp.W)):
        raise ValueError(f"mask shape {fluid.shape} does not match grid {(interp.H, interp.W)}")
    solid = ~fluid
    tags = _normalised_detail_tags(mesh)
    use_all_fallback = not tags
    has_inlet = use_all_fallback or bool({"inlet", "left"} & tags)
    has_outlet = use_all_fallback or bool({"outlet", "right"} & tags)
    has_channel = use_all_fallback or bool({"channelwall", "channel_wall"} & tags)
    has_obstacle = use_all_fallback or bool({"obstaclewall", "obstacle_wall"} & tags)

    left_band, right_band, channel_band = _outer_boundary_bands(interp)
    outer = left_band | right_band | channel_band
    adjacent_to_solid = _neighbour_dilation(solid) & fluid

    labels = np.zeros(fluid.shape, dtype=np.int64)
    labels[solid] = 5

    assigned = solid.copy()
    if has_inlet:
        inlet = fluid & left_band & ~assigned
        labels[inlet] = 1
        assigned |= inlet
    if has_outlet:
        outlet = fluid & right_band & ~assigned
        labels[outlet] = 2
        assigned |= outlet
    if has_channel:
        channel = fluid & channel_band & ~assigned
        labels[channel] = 4
        assigned |= channel
    if has_obstacle:
        obstacle = fluid & adjacent_to_solid & ~outer & ~assigned
        labels[obstacle] = 3
        assigned |= obstacle
    interior = fluid & ~assigned
    labels[interior] = 0

    out = np.zeros((N_BOUNDARY_CLASSES, *fluid.shape), dtype=np.float32)
    for cls in range(N_BOUNDARY_CLASSES):
        out[cls] = labels == cls
    return out, list(BOUNDARY_CLASS_NAMES), BOUNDARY_MASK_VERSION


# --------------------------------------------------------------------------
# Sim-level conversion
# --------------------------------------------------------------------------
def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _metadata(sim_dir: Path, cfg: Dict[str, Any], status: Dict[str, Any],
              times: np.ndarray, interp: GridInterpolator,
              mask: np.ndarray,
              neutral_values: np.ndarray | None = None,
              boundary_mask: np.ndarray | None = None,
              boundary_mask_source: str = "geometric_fallback") -> Dict[str, Any]:
    """Build portable metadata for the packed grid file."""
    physical = {k: status.get(k, cfg.get(k)) for k in PHYSICAL_KEYS}
    mesh = {k: status.get(k, cfg.get(k)) for k in MESH_KEYS}
    time_info = {k: cfg.get(k) for k in TIME_KEYS if k in cfg}
    time_info["saved_times"] = [float(t) for t in times.tolist()]
    return {
        "sim_id": status.get("sim_id", cfg.get("sim_id", sim_dir.name)),
        "seed": status.get("seed", cfg.get("seed")),
        "family": status.get("family", cfg.get("family", "unknown")),
        "problem": status.get("problem", cfg.get("problem")),
        "status": status.get("status", "success"),
        "failed_stage": status.get("failed_stage"),
        "invalid_stage": status.get("invalid_stage"),
        "reason": status.get("reason"),
        "number_of_snapshots": int(len(times)),
        "number_of_snapshots_saved": int(status.get("number_of_snapshots_saved", len(times))),
        "physical_parameters": physical,
        "mesh_parameters": mesh,
        "time": time_info,
        "channel_names": CHANNEL_NAMES,
        "mask_semantics": {
            "0": "solid_or_invalid",
            "1": "fluid",
        },
        "boundary_mask": {
            "version": BOUNDARY_MASK_VERSION,
            "class_names": list(BOUNDARY_CLASS_NAMES),
            "shape": (
                [int(v) for v in boundary_mask.shape]
                if boundary_mask is not None else None
            ),
            "source": boundary_mask_source,
            "semantics": {
                str(i): name for i, name in enumerate(BOUNDARY_CLASS_NAMES)
            },
        },
        "solid_neutralization": {
            "applied": True,
            "state_convention": "primitive",
            "channel_names": CHANNEL_NAMES,
            "mask_value_neutralized": 0,
            "neutral_values": (
                [float(v) for v in neutral_values.tolist()]
                if neutral_values is not None else None
            ),
            "nonfluid_pixels": int(np.sum(np.asarray(mask) <= 0.5)),
        },
        "grid_shape": [int(interp.H), int(interp.W)],
        "grid": {
            "bbox": [float(v) for v in interp.bbox],
            "dx": float(interp.dx),
            "dy": float(interp.dy),
            "x_min": float(interp.x_coords[0]),
            "x_max": float(interp.x_coords[-1]),
            "y_min": float(interp.y_coords[0]),
            "y_max": float(interp.y_coords[-1]),
        },
        "source_dir": str(sim_dir),
    }


def _positive_from_metadata(*sources: Dict[str, Any],
                            keys: Tuple[str, ...],
                            default: float | None = None) -> float | None:
    for source in sources:
        for key in keys:
            try:
                value = float(source.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0.0:
                return value
    return default


def _positive_fluid_median(snaps: np.ndarray, mask: np.ndarray,
                           channel: int, default: float) -> float:
    fluid = np.asarray(mask) > 0.5
    if np.any(fluid):
        values = np.asarray(snaps[:, channel, :, :], dtype=np.float32)[:, fluid]
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size:
            median = float(np.median(values))
            if np.isfinite(median) and median > 0.0:
                return median
    return float(default)


def neutral_primitive_values(cfg: Dict[str, Any], status: Dict[str, Any],
                             snaps: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return neutral primitive values [V_x, V_y, rho, T] for non-fluid cells."""
    rho_ref = _positive_from_metadata(
        status,
        cfg,
        keys=("rho_inf",),
    )
    T_ref = _positive_from_metadata(
        status,
        cfg,
        keys=("T_inf", "T_0"),
    )
    if rho_ref is None:
        rho_ref = _positive_fluid_median(snaps, mask, channel=2, default=1.0)
    if T_ref is None:
        T_ref = _positive_fluid_median(snaps, mask, channel=3, default=1.0)
    return np.array([0.0, 0.0, rho_ref, T_ref], dtype=np.float32)


def neutralize_nonfluid_primitives(snaps: np.ndarray, mask: np.ndarray,
                                   cfg: Dict[str, Any] | None = None,
                                   status: Dict[str, Any] | None = None
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Replace mask==0 primitive values with neutral physical values.

    ``snaps`` uses the grid adapter's primitive state convention
    ``[V_x, V_y, rho, T]``.  Interpolation can fill obstacle holes with
    plausible-looking fluid values; this function prevents those values from
    being saved as targets by setting non-fluid cells to zero velocity and
    freestream/reference density and temperature.
    """
    snaps = np.asarray(snaps, dtype=np.float32)
    if snaps.ndim != 4 or snaps.shape[1] != len(CHANNEL_NAMES):
        raise ValueError(
            f"expected snaps with shape (T, {len(CHANNEL_NAMES)}, H, W), got {snaps.shape}"
        )
    mask_arr = np.asarray(mask, dtype=np.float32)
    if mask_arr.shape != snaps.shape[-2:]:
        raise ValueError(f"mask shape {mask_arr.shape} does not match states grid {snaps.shape[-2:]}")
    cfg = cfg or {}
    status = status or {}
    neutral = neutral_primitive_values(cfg, status, snaps, mask_arr)
    out = snaps.copy()
    nonfluid = mask_arr <= 0.5
    if np.any(nonfluid):
        for channel, value in enumerate(neutral):
            out[:, channel, nonfluid] = float(value)
    return out.astype(np.float32), neutral


def convert_one_sim(sim_dir: Path, grid_H: int = 64, grid_W: int = 96
                    ) -> Tuple[
                        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                        Dict, Dict, GridInterpolator,
                    ]:
    """Convert all snapshots in ``sim_dir`` to a regular grid.

    Returns
    -------
    snaps : (T, 4, H, W) float32
    times : (T,)        float32
    cfg   : dict (the per-sim config that the sweep saved)
    meta  : dict (portable metadata for downstream ML/evaluation)
    mask  : (H, W) binary fluid mask
    boundary_mask : (6, H, W) one-hot boundary classes
    interp: GridInterpolator with grid coordinates/spacing
    """
    mesh = load_mesh(sim_dir / "mesh_props.npz")
    interp = GridInterpolator(mesh["centroids"], grid_H=grid_H, grid_W=grid_W)
    mask = interp.mesh_mask(mesh["vertices"], mesh["triangles"])
    boundary_mask, _, _ = compute_boundary_mask(mask, interp, mesh)

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

    cfg = _read_json(sim_dir / "config.json")
    status = _read_json(sim_dir / "status.json")
    times_arr = np.array(times, dtype=np.float32)
    snaps_arr = np.stack(snaps, axis=0)
    snaps_arr, neutral_values = neutralize_nonfluid_primitives(snaps_arr, mask, cfg, status)
    boundary_source = "bc_type_detailed" if "bc_type_detailed" in mesh else "geometric_fallback"
    meta = _metadata(
        sim_dir, cfg, status, times_arr, interp, mask, neutral_values,
        boundary_mask=boundary_mask,
        boundary_mask_source=boundary_source,
    )
    return snaps_arr, times_arr, mask, boundary_mask, cfg, meta, interp


def viscosity_law_one_hot(law: str | None) -> np.ndarray:
    law = str(law or "sutherland")
    return np.array([1.0 if law == name else 0.0 for name in VISCOSITY_LAWS], dtype=np.float32)


def eos_type_one_hot(eos_type: str | None) -> np.ndarray:
    eos_type = str(eos_type or "ideal")
    return np.array([1.0 if eos_type == name else 0.0 for name in EOS_TYPES], dtype=np.float32)


def pde_fingerprint(cfg: Dict) -> np.ndarray:
    """Numerical fingerprint used only for diagnostics (NOT fed to model)."""
    gamma = float(cfg.get("gamma", 0.0))
    C_v = float(cfg.get("C_v", 0.0))
    rho_inf = float(cfg.get("rho_inf", 0.0))
    T_inf = float(cfg.get("T_inf", 0.0))
    p_inf_value = float(cfg.get("p_inf", 0.0))
    pressure_scale = C_v * (gamma - 1.0) * rho_inf * T_inf
    p_inf_ratio_value = float(cfg.get("p_inf_ratio", 0.0))
    if "p_inf_ratio" not in cfg and pressure_scale > 0:
        p_inf_ratio_value = gamma * p_inf_value / pressure_scale
    base = np.array([
        gamma,
        cfg.get("viscosity", 0.0),
        cfg.get("visc_bulk", 0.0),
        cfg.get("thermal_cond", 0.0),
        C_v,
        cfg.get("T_0", 0.0),
        rho_inf,
        T_inf,
        cfg.get("v_n_inf", 0.0),
    ], dtype=np.float32)
    law = str(cfg.get("viscosity_law", "sutherland"))
    power_law_n = np.array([cfg.get("power_law_n", 0.75)], dtype=np.float32)
    eos_type = str(cfg.get("eos_type", "ideal"))
    p_inf = np.array([p_inf_value], dtype=np.float32)
    p_inf_ratio = np.array([p_inf_ratio_value], dtype=np.float32)
    return np.concatenate([
        base,
        viscosity_law_one_hot(law),
        power_law_n,
        eos_type_one_hot(eos_type),
        p_inf,
        p_inf_ratio,
    ]).astype(np.float32)


def pde_fingerprint_names() -> List[str]:
    return list(DEFAULT_PDE_VEC_NAMES)


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
            failed_stage = status.get("failed_stage") or status.get("invalid_stage", "unknown")
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
    skipped: Dict[str, int] = {}
    for sd in sim_dirs:
        skip_reason = simulation_skip_reason(sd)
        if skip_reason is not None:
            skipped[skip_reason] = skipped.get(skip_reason, 0) + 1
            print(f"  SKIP {sd.name}: {skip_reason}")
            continue
        snaps, times, mask, boundary_mask, cfg, meta, interp = convert_one_sim(sd, grid_H=grid_H, grid_W=grid_W)
        if not np.all(np.isfinite(snaps)):
            skip_reason = "non-finite values after interp"
            skipped[skip_reason] = skipped.get(skip_reason, 0) + 1
            print(f"  SKIP {sd.name}: {skip_reason}")
            continue
        out_path = out_dir / f"{sd.name}.npz"
        np.savez_compressed(
            out_path,
            states=snaps.astype(np.float32),
            snapshots=snaps.astype(np.float32),
            mask=mask.astype(np.float32),
            boundary_mask=boundary_mask.astype(np.float32),
            boundary_class_names=np.array(BOUNDARY_CLASS_NAMES),
            boundary_mask_version=np.array(BOUNDARY_MASK_VERSION),
            boundary_mask_shape=np.array(boundary_mask.shape, dtype=np.int32),
            times=times,
            channel_names=np.array(CHANNEL_NAMES),
            x_coords=interp.x_coords,
            y_coords=interp.y_coords,
            dx=np.float32(interp.dx),
            dy=np.float32(interp.dy),
            bbox=np.array(interp.bbox, dtype=np.float32),
            metadata_json=json.dumps(meta),
            pde_vec=pde_fingerprint(cfg),
            pde_vec_names=np.array(pde_fingerprint_names()),
            cfg_json=json.dumps(cfg),
        )
        saved.append(str(out_path))
        print(f"  WRITE {out_path.name}: snaps={snaps.shape}  "
              f"mask_fluid={float(mask.mean()):.3f}  "
              f"gamma={cfg.get('gamma', 0):.2f}  mu={cfg.get('viscosity', 0):.1e}")
    print(f"\nGrid adapter summary: converted={len(saved)} skipped={sum(skipped.values())}")
    for reason, count in sorted(skipped.items(), key=lambda item: (-item[1], item[0])):
        print(f"  skipped {count}: {reason}")
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
