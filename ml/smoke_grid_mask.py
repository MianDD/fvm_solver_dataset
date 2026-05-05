"""Smoke test for grid-adapter non-fluid neutralization."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .dataset import CFDWindowDataset
from .grid_adapter import CHANNEL_NAMES, assemble_dataset
from .pde import DEFAULT_PDE_VEC_NAMES


def _vertex_id(i: int, j: int, n: int = 4) -> int:
    return j * n + i


def _write_synthetic_raw_sim(sim_dir: Path) -> None:
    sim_dir.mkdir(parents=True, exist_ok=True)
    coords = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    vertices = np.array([(x, y) for y in coords for x in coords], dtype=np.float32)

    triangles = []
    for j in range(3):
        for i in range(3):
            if i == 1 and j == 1:
                continue  # one obstacle/non-fluid hole in the middle
            v00 = _vertex_id(i, j)
            v10 = _vertex_id(i + 1, j)
            v01 = _vertex_id(i, j + 1)
            v11 = _vertex_id(i + 1, j + 1)
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])
    triangles_arr = np.asarray(triangles, dtype=np.int64)
    centroids = vertices[triangles_arr].mean(axis=1).astype(np.float32)

    np.savez_compressed(
        sim_dir / "mesh_props.npz",
        vertices=vertices,
        triangles=triangles_arr,
        centroids=centroids,
        edges=np.zeros((0, 2), dtype=np.int64),
        bc_edge_masK=np.zeros((0,), dtype=np.int64),
        bc_type_str=np.array([], dtype="<U8"),
    )

    primitives = np.zeros((triangles_arr.shape[0], len(CHANNEL_NAMES)), dtype=np.float32)
    primitives[:, 0] = 5.0
    primitives[:, 1] = -2.0
    primitives[:, 2] = 9.0
    primitives[:, 3] = 99.0
    for step, t in enumerate([0.0, 0.1, 0.2]):
        np.savez_compressed(
            sim_dir / f"t_{t:.3f}.npz",
            t=np.float32(t),
            prim_mean=np.zeros((len(CHANNEL_NAMES),), dtype=np.float32),
            prim_std=np.ones((len(CHANNEL_NAMES),), dtype=np.float32),
            cell_primatives=primitives,
        )

    cfg = {
        "sim_id": 0,
        "seed": 123,
        "family": "ood_mild",
        "status": "success",
        "gamma": 1.4,
        "viscosity": 1e-3,
        "visc_bulk": 1e-3,
        "thermal_cond": 1e-6,
        "C_v": 2.5,
        "T_0": 100.0,
        "rho_inf": 1.23,
        "T_inf": 123.0,
        "v_n_inf": 6.0,
        "viscosity_law": "sutherland",
        "power_law_n": 0.75,
        "eos_type": "stiffened_gas",
        "p_inf": 4.0,
        "p_inf_ratio": 0.05,
        "dt": 0.1,
        "save_t": 0.1,
        "n_iter": 3,
    }
    (sim_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    status = {
        **cfg,
        "status": "success",
        "number_of_snapshots_saved": 3,
    }
    (sim_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def main() -> None:
    root = Path("datasets") / "_codex_smoke_grid_mask"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    raw = root / "raw"
    grid = root / "grid"
    try:
        _write_synthetic_raw_sim(raw / "sim_0000")
        saved = assemble_dataset(raw, grid, grid_H=24, grid_W=24)
        if len(saved) != 1:
            raise RuntimeError(f"expected one converted file, got {saved}")
        out_path = Path(saved[0])
        z = np.load(out_path, allow_pickle=True)
        states = z["states"].astype(np.float32)
        mask = z["mask"].astype(np.float32)
        pde_vec = z["pde_vec"].astype(np.float32)
        pde_names = [str(x) for x in z["pde_vec_names"].tolist()]
        metadata = json.loads(z["metadata_json"].item())

        if states.shape != (3, 4, 24, 24):
            raise RuntimeError(f"unexpected state shape {states.shape}")
        if pde_vec.shape != (len(DEFAULT_PDE_VEC_NAMES),):
            raise RuntimeError(f"expected current 17D pde_vec, got {pde_vec.shape}")
        if pde_names != list(DEFAULT_PDE_VEC_NAMES):
            raise RuntimeError("pde_vec_names do not match current default schema")
        if metadata.get("mask_semantics", {}).get("0") != "solid_or_invalid":
            raise RuntimeError("mask semantics missing from metadata")

        nonfluid = mask <= 0.5
        fluid = mask > 0.5
        if not np.any(nonfluid):
            raise RuntimeError("synthetic mesh did not produce non-fluid mask pixels")
        if not np.any(fluid):
            raise RuntimeError("synthetic mesh did not produce fluid mask pixels")

        neutral = np.array([0.0, 0.0, 1.23, 123.0], dtype=np.float32)
        for channel, value in enumerate(neutral):
            values = states[:, channel, :, :][:, nonfluid]
            if not np.allclose(values, value, rtol=0.0, atol=1e-6):
                raise RuntimeError(
                    f"non-fluid channel {CHANNEL_NAMES[channel]} was not neutralized to {value}"
                )

        expected_fluid = np.array([5.0, -2.0, 9.0, 99.0], dtype=np.float32)
        for channel, value in enumerate(expected_fluid):
            values = states[:, channel, :, :][:, fluid]
            if not np.allclose(values, value, rtol=0.0, atol=1e-5):
                raise RuntimeError(
                    f"fluid channel {CHANNEL_NAMES[channel]} was unexpectedly overwritten"
                )

        neutral_meta = metadata.get("solid_neutralization", {})
        if neutral_meta.get("state_convention") != "primitive":
            raise RuntimeError("neutralization metadata should record primitive state convention")
        if neutral_meta.get("neutral_values") != [float(v) for v in neutral.tolist()]:
            raise RuntimeError(f"unexpected neutralization metadata: {neutral_meta}")

        ds = CFDWindowDataset([out_path], context_length=2, prediction_horizon=1)
        sample = ds[0]
        if sample["input_states"].shape[1:] != (4, 24, 24):
            raise RuntimeError("converted grid did not load through CFDWindowDataset")
        if sample["pde_vec"].numel() != len(DEFAULT_PDE_VEC_NAMES):
            raise RuntimeError("dataset did not preserve current pde_vec length")

        print(
            "OK grid mask smoke: "
            f"states={states.shape} nonfluid={int(np.sum(nonfluid))} "
            f"neutral={neutral.tolist()} pde_dim={pde_vec.shape[0]}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
