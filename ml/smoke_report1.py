"""Synthetic smoke test for Report 1 model/data options.

This creates tiny grid ``.npz`` files with pde_vec and a fluid mask, loads them
through ``CFDWindowDataset``, runs a model forward/backward pass with
sinusoidal positions, mask channel, masked loss, and PDE auxiliary loss.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .dataset import CFDWindowDataset
from .model import FoundationCFDModel
from .train import weighted_masked_mse


def _write_npz(path: Path, with_mask: bool) -> None:
    rng = np.random.default_rng(123)
    T, C, H, W = 6, 4, 16, 24
    states = rng.normal(size=(T, C, H, W)).astype(np.float32)
    states[:, 2] = np.abs(states[:, 2]) + 1.0
    states[:, 3] = np.abs(states[:, 3]) + 100.0
    times = np.arange(T, dtype=np.float32) * 0.01
    mask = np.ones((H, W), dtype=np.float32)
    mask[4:8, 9:14] = 0.0
    pde_vec = np.linspace(0.1, 1.3, 13, dtype=np.float32)
    payload = {
        "states": states,
        "snapshots": states,
        "times": times,
        "channel_names": np.array(["V_x", "V_y", "rho", "T"]),
        "dx": np.float32(0.1),
        "dy": np.float32(0.1),
        "metadata_json": json.dumps({"sim_id": path.stem, "family": "synthetic", "status": "success"}),
        "cfg_json": json.dumps({"sim_id": path.stem, "family": "synthetic"}),
        "pde_vec": pde_vec,
        "pde_vec_names": np.array([f"pde_{i}" for i in range(len(pde_vec))]),
    }
    if with_mask:
        payload["mask"] = mask
    np.savez_compressed(path, **payload)


def main() -> None:
    root = Path("datasets") / "_codex_smoke_report1"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        _write_npz(root / "sim_0000.npz", with_mask=True)
        _write_npz(root / "sim_0001.npz", with_mask=False)
        paths = sorted(root.glob("*.npz"))
        ds = CFDWindowDataset(
            paths,
            context_length=3,
            prediction_horizon=1,
            strides="1,2",
            use_derivatives=True,
            use_mask_channel=True,
        )
        if len(ds) == 0:
            raise RuntimeError("synthetic dataset produced no windows")
        batch = ds[0]
        x = batch["input_states"].unsqueeze(0)
        target = batch["target_states"][0].unsqueeze(0)
        mask = batch["target_mask"].unsqueeze(0)
        pde = batch["pde_vec"].unsqueeze(0)
        model = FoundationCFDModel(
            n_channels=4,
            n_input_channels=x.shape[2],
            n_target_channels=4,
            H=x.shape[-2],
            W=x.shape[-1],
            patch_size=8,
            d_model=32,
            n_heads=4,
            n_layers=1,
            max_context=3,
            attention_type="factorized",
            pos_encoding="sinusoidal",
            pde_dim=pde.shape[1],
        )
        update, pde_pred = model.predict_update_and_pde(x)
        pred = model.integrate_update(batch["context_states"][-1].unsqueeze(0), update[:, -1])
        pred_loss = weighted_masked_mse(pred, target, mask=mask)
        pde_loss = torch.mean((pde_pred - pde) ** 2)
        total = pred_loss + 0.01 * pde_loss
        total.backward()
        if tuple(pred.shape) != tuple(target.shape):
            raise RuntimeError(f"prediction shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        if pde_pred.shape != pde.shape:
            raise RuntimeError(f"pde shape mismatch: {tuple(pde_pred.shape)} vs {tuple(pde.shape)}")
        print(
            "OK synthetic report1 smoke: "
            f"windows={len(ds)} C_in={x.shape[2]} pred_loss={float(pred_loss.detach()):.4e} "
            f"pde_loss={float(pde_loss.detach()):.4e}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
