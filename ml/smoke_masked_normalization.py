"""Smoke test for mask-aware channel normalisation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import CFDWindowDataset
from .pde import DEFAULT_PDE_VEC_NAMES
from .train import TrainConfig, compute_channel_stats, train


def _pde_vec(sim_idx: int) -> np.ndarray:
    base = np.array([
        1.35 + 0.01 * sim_idx,
        1.0e-3,
        2.0e-3,
        1.0e-6,
        2.5,
        100.0,
        1.0,
        100.0,
        6.0,
    ], dtype=np.float32)
    law = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    power_law_n = np.array([0.75], dtype=np.float32)
    eos = np.array([1.0, 0.0], dtype=np.float32)
    p_inf = np.array([0.0, 0.0], dtype=np.float32)
    return np.concatenate([base, law, power_law_n, eos, p_inf]).astype(np.float32)


def _write_grid(path: Path, with_mask: bool, sim_idx: int,
                solid_value: float = 1000.0) -> None:
    rng = np.random.default_rng(100 + sim_idx)
    T, C, H, W = 5, 4, 8, 8
    states = np.zeros((T, C, H, W), dtype=np.float32)
    mask = np.ones((H, W), dtype=np.float32)
    mask[:3, :3] = 0.0
    fluid = mask > 0.5
    solid = ~fluid

    base = np.array([10.0 + sim_idx, -3.0, 2.0, 50.0], dtype=np.float32)
    for t in range(T):
        states[t] = base[:, None, None]
        states[t, 0, fluid] += 0.1 * t
        states[t, 1, fluid] += 0.05 * rng.normal(size=int(np.sum(fluid)))
        states[t, 2, fluid] += 0.01 * t
        states[t, 3, fluid] += 0.2 * t
        states[t, :, solid] = solid_value

    metadata = {
        "sim_id": f"sim_{sim_idx:04d}",
        "family": "synthetic",
        "status": "success",
        "number_of_snapshots": T,
        "channel_names": ["V_x", "V_y", "rho", "T"],
        "mask_semantics": {"0": "solid_or_invalid", "1": "fluid"},
    }
    payload = {
        "states": states,
        "snapshots": states,
        "times": np.arange(T, dtype=np.float32) * 0.1,
        "channel_names": np.array(["V_x", "V_y", "rho", "T"]),
        "dx": np.float32(0.1),
        "dy": np.float32(0.1),
        "metadata_json": json.dumps(metadata),
        "cfg_json": json.dumps({"sim_id": sim_idx, "family": "synthetic"}),
        "pde_vec": _pde_vec(sim_idx),
        "pde_vec_names": np.array(DEFAULT_PDE_VEC_NAMES),
    }
    if with_mask:
        payload["mask"] = mask
    np.savez_compressed(path, **payload)


def _check_masked_stats(grid_dir: Path) -> None:
    ds = CFDWindowDataset(
        sorted(grid_dir.glob("*.npz")),
        context_length=2,
        prediction_horizon=1,
        use_derivatives=False,
        use_mask_channel=True,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    mean, std = compute_channel_stats(
        loader,
        n_channels=5,
        use_fluid_mask=True,
        mask_channel_index=4,
    )
    # Sliding windows with context=2 weight frames [0, 1, 2, 3] as [1, 2, 2, 1],
    # so the temporal ramp contributes an average frame index of 1.5.
    expected = torch.tensor([10.65, -3.0, 2.015, 50.3], dtype=torch.float32)
    if not torch.allclose(mean[:4], expected, atol=0.05):
        raise RuntimeError(f"masked stats included solid values: mean={mean.tolist()}")
    if not torch.isclose(mean[4], torch.tensor(0.0)) or not torch.isclose(std[4], torch.tensor(1.0)):
        raise RuntimeError(f"mask channel should use mean=0/std=1, got mean={mean[4]} std={std[4]}")

    sample = ds[0]["input_states"]
    normalised = (sample - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    mask_values = torch.unique(normalised[:, -1])
    if set(float(v) for v in mask_values.tolist()) != {0.0, 1.0}:
        raise RuntimeError(f"mask channel changed after normalisation: {mask_values.tolist()}")


def _check_unmasked_compat(grid_dir: Path) -> None:
    ds = CFDWindowDataset(
        sorted(grid_dir.glob("*.npz")),
        context_length=2,
        prediction_horizon=1,
        use_derivatives=False,
        use_mask_channel=False,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    mean, std = compute_channel_stats(loader, n_channels=4, use_fluid_mask=False)
    if not torch.all(torch.isfinite(mean)) or not torch.all(torch.isfinite(std)):
        raise RuntimeError("unmasked stats should remain finite")
    if float(mean[0]) < 100.0:
        raise RuntimeError("unmasked compatibility path should include all grid cells")


def main() -> None:
    root = Path("datasets") / "_codex_smoke_masked_normalization"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    grid = root / "grid"
    grid_nomask = root / "grid_nomask"
    try:
        grid.mkdir(parents=True)
        grid_nomask.mkdir(parents=True)
        for sim_idx in range(2):
            _write_grid(grid / f"sim_{sim_idx:04d}.npz", with_mask=True, sim_idx=sim_idx)
            _write_grid(grid_nomask / f"sim_{sim_idx:04d}.npz", with_mask=False, sim_idx=sim_idx)

        _check_masked_stats(grid)
        _check_unmasked_compat(grid_nomask)

        result = train(
            TrainConfig(
                grid_dir=str(grid),
                out_dir=str(root / "train"),
                context_length=2,
                prediction_horizon=1,
                n_epochs=1,
                batch_size=2,
                val_frac=0.5,
                seed=11,
                patch_size=4,
                d_model=16,
                n_heads=4,
                n_layers=1,
                attention_type="factorized",
                pos_encoding="sinusoidal",
                use_derivatives=True,
                use_mask_channel=True,
                mask_loss=True,
                pde_aux_loss=True,
                pde_aux_weight=0.01,
                pde_normalize=True,
                pde_log_transport=True,
                pde_cont_loss="huber",
                input_noise_std=0.005,
            ),
            device="cpu",
        )
        metrics = result["metrics"]
        if not np.isfinite(metrics["final_train_loss"]):
            raise RuntimeError("masked-normalisation training loss is not finite")
        normalizer = json.loads((root / "train" / "normalizer.json").read_text(encoding="utf-8"))
        if not normalizer.get("masked_channel_stats"):
            raise RuntimeError("normalizer.json should record masked_channel_stats=true")
        if normalizer.get("mask_channel_index") != len(normalizer["mean"]) - 1:
            raise RuntimeError("normalizer.json should record the final mask-channel index")
        if normalizer["mean"][-1] != 0.0 or normalizer["std"][-1] != 1.0:
            raise RuntimeError("saved normalizer should leave mask channel raw")

        print(
            "OK masked normalization smoke: "
            f"C_in={len(normalizer['mean'])} mask_index={normalizer['mask_channel_index']} "
            f"train_loss={metrics['final_train_loss']:.4e}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
