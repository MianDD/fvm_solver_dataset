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
from .evaluate import evaluate
from .model import FoundationCFDModel
from .pde import (
    BASE_PDE_NAMES,
    DEFAULT_PDE_VEC_NAMES,
    compute_pde_metrics,
    compute_pde_normalizer,
    infer_pde_schema,
    pde_loss_components,
    pde_vectors_from_records,
)
from .train import weighted_masked_mse


def _pde_vec_13(law_index: int) -> np.ndarray:
    base = np.array([1.4, 1e-3, 1e-3, 1e-6, 2.5, 100.0, 1.0, 100.0, 8.0], dtype=np.float32)
    one_hot = np.zeros(3, dtype=np.float32)
    one_hot[int(law_index)] = 1.0
    return np.concatenate([base, one_hot, np.array([0.75 + 0.05 * law_index], dtype=np.float32)])


def _write_npz(path: Path, with_mask: bool, pde_dim: int = 13,
               law_index: int = 0, family: str = "synthetic") -> None:
    rng = np.random.default_rng(123)
    T, C, H, W = 6, 4, 16, 24
    states = rng.normal(size=(T, C, H, W)).astype(np.float32)
    states[:, 2] = np.abs(states[:, 2]) + 1.0
    states[:, 3] = np.abs(states[:, 3]) + 100.0
    times = np.arange(T, dtype=np.float32) * 0.01
    mask = np.ones((H, W), dtype=np.float32)
    mask[4:8, 9:14] = 0.0
    if pde_dim == 13:
        pde_vec = _pde_vec_13(law_index)
        pde_names = np.array(DEFAULT_PDE_VEC_NAMES)
    else:
        pde_vec = _pde_vec_13(law_index)[:9]
        pde_names = np.array(BASE_PDE_NAMES)
    payload = {
        "states": states,
        "snapshots": states,
        "times": times,
        "channel_names": np.array(["V_x", "V_y", "rho", "T"]),
        "dx": np.float32(0.1),
        "dy": np.float32(0.1),
        "metadata_json": json.dumps({"sim_id": path.stem, "family": family, "status": "success"}),
        "cfg_json": json.dumps({"sim_id": path.stem, "family": family}),
        "pde_vec": pde_vec,
        "pde_vec_names": pde_names,
    }
    if with_mask:
        payload["mask"] = mask
    np.savez_compressed(path, **payload)


def _realistic_pde_vectors(n: int = 96) -> np.ndarray:
    rng = np.random.default_rng(321)
    laws = np.arange(n) % 3
    one_hot = np.zeros((n, 3), dtype=np.float32)
    one_hot[np.arange(n), laws] = 1.0
    vectors = np.column_stack([
        rng.uniform(1.2, 1.6, size=n),              # gamma
        10 ** rng.uniform(-4, -2, size=n),          # viscosity
        10 ** rng.uniform(-4, -2, size=n),          # visc_bulk
        10 ** rng.uniform(-7, -5, size=n),          # thermal_cond
        rng.uniform(1.5, 3.5, size=n),              # C_v
        rng.uniform(50.0, 150.0, size=n),           # T_0
        rng.uniform(0.7, 1.3, size=n),              # rho_inf
        rng.uniform(50.0, 150.0, size=n),           # T_inf
        rng.uniform(3.0, 12.0, size=n),             # v_n_inf
        one_hot,
        rng.uniform(0.3, 1.5, size=(n, 1)),         # power_law_n
    ])
    return vectors.astype(np.float32)


def _stress_pde_normalizer() -> None:
    vectors = _realistic_pde_vectors()
    schema = infer_pde_schema(DEFAULT_PDE_VEC_NAMES, vectors.shape[1])
    normalizer = compute_pde_normalizer(vectors, names=DEFAULT_PDE_VEC_NAMES)
    if normalizer.get("log_indices") != [1, 2, 3]:
        raise RuntimeError(f"unexpected log_indices: {normalizer.get('log_indices')}")
    if not np.all(np.isfinite(normalizer["mean"])) or not np.all(np.isfinite(normalizer["std"])):
        raise RuntimeError("PDE normalizer produced non-finite mean/std")

    mean_raw = np.asarray(normalizer["raw_mean"], dtype=np.float32)
    mean_transformed = np.asarray(normalizer["mean"], dtype=np.float32)
    for idx in normalizer["log_indices"]:
        mean_raw[idx] = np.exp(mean_transformed[idx])
    mean_pred = np.repeat(mean_raw[None, :], vectors.shape[0], axis=0)
    parts = pde_loss_components(
        torch.from_numpy(mean_pred),
        torch.from_numpy(vectors),
        schema,
        normalizer=normalizer,
        normalize=True,
    )
    cont_loss = float(parts["continuous"].detach())
    if not np.isfinite(cont_loss) or not (0.1 <= cont_loss <= 3.0):
        raise RuntimeError(f"mean-predictor transformed PDE loss not O(1): {cont_loss}")

    perturbed = vectors.copy()
    perturbed[:, 1:4] *= 1.05
    perturbed_parts = pde_loss_components(
        torch.from_numpy(perturbed),
        torch.from_numpy(vectors),
        schema,
        normalizer=normalizer,
        normalize=True,
    )
    perturb_loss = float(perturbed_parts["continuous"].detach())
    if not np.isfinite(perturb_loss) or perturb_loss > 0.5:
        raise RuntimeError(f"5% transport perturbation produced exploding loss: {perturb_loss}")


def main() -> None:
    root = Path("datasets") / "_codex_smoke_report1"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        _stress_pde_normalizer()
        grid13 = root / "grid13"
        grid9 = root / "grid9"
        grid13.mkdir()
        grid9.mkdir()
        _write_npz(grid13 / "sim_0000.npz", with_mask=True, pde_dim=13, law_index=0, family="id")
        _write_npz(grid13 / "sim_0001.npz", with_mask=False, pde_dim=13, law_index=2, family="ood_mild")
        _write_npz(grid9 / "sim_0000.npz", with_mask=True, pde_dim=9, law_index=0, family="id")
        _write_npz(grid9 / "sim_0001.npz", with_mask=True, pde_dim=9, law_index=1, family="ood_mild")
        paths = sorted(grid13.glob("*.npz"))
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
        pde_normalizer = compute_pde_normalizer(pde_vectors_from_records(ds._cache), names=ds.pde_vec_names)
        pde_schema = infer_pde_schema(ds.pde_vec_names, ds.pde_dim)
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
        with torch.no_grad():
            model.pde_head[-1].weight.zero_()
            model.pde_head[-1].bias.copy_(torch.as_tensor(pde_normalizer["raw_mean"], dtype=torch.float32))
        update, pde_pred = model.predict_update_and_pde(x)
        pred = model.integrate_update(batch["context_states"][-1].unsqueeze(0), update[:, -1])
        pred_loss = weighted_masked_mse(pred, target, mask=mask)
        pde_parts = pde_loss_components(
            pde_pred,
            pde,
            pde_schema,
            normalizer=pde_normalizer,
            normalize=True,
        )
        total = pred_loss + 0.01 * pde_parts["total"]
        total.backward()
        if tuple(pred.shape) != tuple(target.shape):
            raise RuntimeError(f"prediction shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        if pde_pred.shape != pde.shape:
            raise RuntimeError(f"pde shape mismatch: {tuple(pde_pred.shape)} vs {tuple(pde.shape)}")
        pde_metrics = compute_pde_metrics(
            pde_pred.detach().numpy(),
            pde.numpy(),
            schema=pde_schema,
            normalizer=pde_normalizer,
        )
        if pde_metrics["viscosity_law"] is None:
            raise RuntimeError("13D synthetic pde_vec did not produce viscosity-law metrics")

        old_ds = CFDWindowDataset(sorted(grid9.glob("*.npz")), context_length=3, prediction_horizon=1)
        old_schema = infer_pde_schema(old_ds.pde_vec_names, old_ds.pde_dim)
        if old_schema.get("viscosity_law_indices"):
            raise RuntimeError("old 9D pde_vec unexpectedly produced law indices")
        old_pde = old_ds[0]["pde_vec"].unsqueeze(0)
        old_pred = old_pde.clone()
        old_norm = compute_pde_normalizer(pde_vectors_from_records(old_ds._cache))
        old_parts = pde_loss_components(old_pred, old_pde, old_schema, normalizer=old_norm, normalize=True)
        if not torch.isfinite(old_parts["total"]):
            raise RuntimeError("old 9D normalized PDE loss was not finite")

        ckpt_path = root / "synthetic_pde.pt"
        torch.save({
            "model": model.state_dict(),
            "model_config": {
                "n_channels": 4,
                "n_input_channels": x.shape[2],
                "n_target_channels": 4,
                "H": x.shape[-2],
                "W": x.shape[-1],
                "patch_size": 8,
                "d_model": 32,
                "n_heads": 4,
                "n_layers": 1,
                "max_context": 3,
                "attention_type": "factorized",
                "pos_encoding": "sinusoidal",
                "pde_dim": pde.shape[1],
                "pde_schema": pde_schema,
                "pde_normalizer": pde_normalizer,
                "use_derivatives": True,
                "use_mask_channel": True,
                "derivative_mode": "central",
                "prediction_mode": "delta",
                "integrator": "euler",
                "strides": [1],
            },
            "config": {
                "context_length": 3,
                "prediction_horizon": 1,
                "use_derivatives": True,
                "use_mask_channel": True,
                "derivative_mode": "central",
                "prediction_mode": "delta",
                "integrator": "euler",
                "strides": "1",
                "pde_aux_loss": True,
                "pde_normalize": True,
            },
            "pde_schema": pde_schema,
            "pde_normalizer": pde_normalizer,
        }, ckpt_path)
        eval_metrics = evaluate(grid13, ckpt_path, root / "eval", context=3, horizon=1,
                                batch_size=1, device="cpu", rollout_steps="")
        if "pde_identification" not in eval_metrics:
            raise RuntimeError("synthetic evaluation did not produce PDE identification metrics")
        if "by_family" not in eval_metrics["pde_identification"]:
            raise RuntimeError("synthetic evaluation did not produce by_family PDE metrics")
        pde_json = root / "eval" / "pde_metrics.json"
        if not pde_json.exists():
            raise RuntimeError("synthetic evaluation did not write pde_metrics.json")
        print(
            "OK synthetic report1 smoke: "
            f"windows={len(ds)} C_in={x.shape[2]} pred_loss={float(pred_loss.detach()):.4e} "
            f"pde_cont={float(pde_parts['continuous'].detach()):.4e} "
            f"pde_law={float(pde_parts['law'].detach()):.4e} "
            f"old9_loss={float(old_parts['total'].detach()):.4e}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
