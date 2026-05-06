"""Smoke test for boundary one-hot masks and input channels."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .boundary import BOUNDARY_CLASS_NAMES, BOUNDARY_MASK_VERSION
from .dataset import CFDWindowDataset
from .grid_adapter import compute_boundary_mask
from .pde import DEFAULT_PDE_VEC_NAMES
from .train import TrainConfig, train


def _pde_vec() -> np.ndarray:
    out = np.zeros((len(DEFAULT_PDE_VEC_NAMES),), dtype=np.float32)
    out[:9] = np.array([1.4, 1e-3, 0.0, 3.5e-3, 2.5, 100.0, 1.0, 100.0, 5.0])
    if out.shape[0] >= 12:
        out[9:12] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if out.shape[0] >= 15:
        out[13:15] = np.array([1.0, 0.0], dtype=np.float32)
    return out


def _mask(H: int = 16, W: int = 24) -> np.ndarray:
    mask = np.ones((H, W), dtype=np.float32)
    mask[6:10, 10:14] = 0.0
    return mask


def _boundary_for(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    interp = SimpleNamespace(
        H=H,
        W=W,
        x_coords=np.linspace(-1.0, 3.0, W, dtype=np.float32),
        y_coords=np.linspace(-1.0, 1.0, H, dtype=np.float32),
        dx=np.float32(4.0 / max(1, W - 1)),
        dy=np.float32(2.0 / max(1, H - 1)),
    )
    boundary, names, version = compute_boundary_mask(mask, interp)
    if names != list(BOUNDARY_CLASS_NAMES):
        raise RuntimeError(f"unexpected class names: {names}")
    if version != BOUNDARY_MASK_VERSION:
        raise RuntimeError(f"unexpected boundary version: {version}")
    return boundary


def _write_grid(path: Path, sim_idx: int, mask: np.ndarray, boundary: np.ndarray) -> None:
    T, C, H, W = 5, 4, *mask.shape
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, H, dtype=np.float32),
        np.linspace(0.0, 1.0, W, dtype=np.float32),
        indexing="ij",
    )
    states = np.zeros((T, C, H, W), dtype=np.float32)
    for t in range(T):
        states[t, 0] = 1.0 + 0.1 * t + xx
        states[t, 1] = -0.5 + yy
        states[t, 2] = 1.0 + 0.01 * sim_idx + 0.1 * xx
        states[t, 3] = 100.0 + yy
    solid = mask <= 0.5
    states[:, 0, solid] = 0.0
    states[:, 1, solid] = 0.0
    states[:, 2, solid] = 1.0
    states[:, 3, solid] = 100.0
    metadata = {
        "sim_id": f"sim_{sim_idx:04d}",
        "family": "synthetic_boundary",
        "status": "success",
        "number_of_snapshots": T,
        "channel_names": ["V_x", "V_y", "rho", "T"],
    }
    np.savez_compressed(
        path,
        states=states,
        snapshots=states,
        mask=mask.astype(np.float32),
        boundary_mask=boundary.astype(np.float32),
        boundary_class_names=np.array(BOUNDARY_CLASS_NAMES),
        boundary_mask_version=np.array(BOUNDARY_MASK_VERSION),
        boundary_mask_shape=np.array(boundary.shape, dtype=np.int32),
        times=np.arange(T, dtype=np.float32) * 0.1,
        channel_names=np.array(["V_x", "V_y", "rho", "T"]),
        dx=np.float32(1.0 / max(1, W - 1)),
        dy=np.float32(1.0 / max(1, H - 1)),
        metadata_json=json.dumps(metadata),
        cfg_json=json.dumps({"sim_id": sim_idx, "family": "synthetic_boundary"}),
        pde_vec=_pde_vec(),
        pde_vec_names=np.array(DEFAULT_PDE_VEC_NAMES),
    )


def _check_boundary_mask(boundary: np.ndarray, mask: np.ndarray) -> None:
    if boundary.shape != (6, *mask.shape):
        raise RuntimeError(f"unexpected boundary mask shape {boundary.shape}")
    if not np.allclose(boundary.sum(axis=0), 1.0, atol=1e-6):
        raise RuntimeError("boundary mask is not one-hot at every grid pixel")
    labels = boundary.argmax(axis=0)
    if not np.array_equal(labels == 5, mask <= 0.5):
        raise RuntimeError("solid boundary class does not match binary mask==0")
    inlet_cols = np.where(labels == 1)[1]
    outlet_cols = np.where(labels == 2)[1]
    wall_rows = np.where(labels == 4)[0]
    obstacle = labels == 3
    if inlet_cols.size == 0 or int(inlet_cols.max()) > 1:
        raise RuntimeError("inlet class should appear only near the left boundary")
    if outlet_cols.size == 0 or int(outlet_cols.min()) < mask.shape[1] - 2:
        raise RuntimeError("outlet class should appear only near the right boundary")
    if wall_rows.size == 0 or not np.all((wall_rows <= 1) | (wall_rows >= mask.shape[0] - 2)):
        raise RuntimeError("channel_wall class should appear only near top/bottom boundaries")
    if not np.any(obstacle):
        raise RuntimeError("obstacle_wall class was not detected next to the solid obstacle")
    solid = mask <= 0.5
    adjacent = np.zeros_like(solid, dtype=bool)
    H, W = solid.shape
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
            adjacent[dst_j0:dst_j1, dst_i0:dst_i1] |= solid[src_j0:src_j1, src_i0:src_i1]
    if not np.all(obstacle <= (adjacent & (mask > 0.5))):
        raise RuntimeError("obstacle_wall pixels should be fluid pixels adjacent to solid")


def main() -> None:
    root = Path("datasets") / "_codex_smoke_boundary_mask"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    grid = root / "grid"
    try:
        grid.mkdir(parents=True, exist_ok=True)
        mask = _mask()
        boundary = _boundary_for(mask)
        _check_boundary_mask(boundary, mask)
        for sim_idx in range(2):
            _write_grid(grid / f"sim_{sim_idx:04d}.npz", sim_idx, mask, boundary)

        ds = CFDWindowDataset(
            sorted(grid.glob("*.npz")),
            context_length=2,
            prediction_horizon=1,
            use_derivatives=True,
            use_boundary_channels=True,
            use_mask_channel=True,
        )
        sample = ds[0]
        expected_C = 4 + 12 + 6 + 1
        if sample["input_states"].shape[1] != expected_C:
            raise RuntimeError(
                f"expected {expected_C} input channels with derivatives/boundaries/mask, "
                f"got {sample['input_states'].shape[1]}"
            )
        boundary_slice = sample["input_states"][:, 16:22]
        if not np.allclose(boundary_slice.numpy().sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError("dataset boundary channels are not one-hot over time")

        result = train(
            TrainConfig(
                grid_dir=str(grid),
                out_dir=str(root / "train"),
                context_length=2,
                prediction_horizon=1,
                n_epochs=1,
                batch_size=2,
                val_frac=0.5,
                seed=21,
                patch_size=8,
                d_model=16,
                n_heads=4,
                n_layers=1,
                attention_type="factorized",
                pos_encoding="sinusoidal",
                use_derivatives=True,
                use_boundary_channels=True,
                use_mask_channel=True,
                mask_loss=True,
            ),
            device="cpu",
        )
        loss = result["metrics"]["final_train_loss"]
        if not np.isfinite(loss):
            raise RuntimeError(f"training loss is not finite: {loss}")
        normalizer = json.loads((root / "train" / "normalizer.json").read_text(encoding="utf-8"))
        if normalizer.get("boundary_channel_indices") != [16, 17, 18, 19, 20, 21]:
            raise RuntimeError(f"unexpected boundary channel indices: {normalizer}")
        for idx in normalizer["boundary_channel_indices"]:
            if normalizer["mean"][idx] != 0.0 or normalizer["std"][idx] != 1.0:
                raise RuntimeError("boundary channels should remain raw 0/1 under normalisation")

        print(
            "OK boundary mask smoke: "
            f"boundary_shape={boundary.shape} C_in={expected_C} train_loss={loss:.4e}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
