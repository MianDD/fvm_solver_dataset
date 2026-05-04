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
from .evaluate_context_scaling import evaluate_context_scaling
from .model import FoundationCFDModel
from .pde import (
    BASE_PDE_NAMES,
    DEFAULT_PDE_VEC_NAMES,
    OLD_13D_PDE_VEC_NAMES,
    compute_pde_metrics,
    compute_pde_normalizer,
    infer_pde_schema,
    pde_loss_components,
    pde_vectors_from_records,
)
from .grid_adapter import pde_fingerprint, pde_fingerprint_names
from .plot_report1_metrics import _plot_confusion
from .train import pde_head_bias_from_normalizer, weighted_masked_mse


def _pde_vec_13(law_index: int) -> np.ndarray:
    base = np.array([1.4, 1e-3, 1e-3, 1e-6, 2.5, 100.0, 1.0, 100.0, 8.0], dtype=np.float32)
    one_hot = np.zeros(3, dtype=np.float32)
    one_hot[int(law_index)] = 1.0
    return np.concatenate([base, one_hot, np.array([0.75 + 0.05 * law_index], dtype=np.float32)])


def _pde_vec_16(law_index: int, eos_index: int, p_inf: float) -> np.ndarray:
    eos_one_hot = np.zeros(2, dtype=np.float32)
    eos_one_hot[int(eos_index)] = 1.0
    return np.concatenate([
        _pde_vec_13(law_index),
        eos_one_hot,
        np.array([p_inf], dtype=np.float32),
    ]).astype(np.float32)


def _write_npz(path: Path, with_mask: bool, pde_dim: int = 13,
               law_index: int = 0, eos_index: int = 0,
               p_inf: float = 0.0, family: str = "synthetic") -> None:
    rng = np.random.default_rng(123)
    T, C, H, W = 6, 4, 16, 24
    states = rng.normal(size=(T, C, H, W)).astype(np.float32)
    states[:, 2] = np.abs(states[:, 2]) + 1.0
    states[:, 3] = np.abs(states[:, 3]) + 100.0
    times = np.arange(T, dtype=np.float32) * 0.01
    mask = np.ones((H, W), dtype=np.float32)
    mask[4:8, 9:14] = 0.0
    if pde_dim == 16:
        pde_vec = _pde_vec_16(law_index, eos_index, p_inf)
        pde_names = np.array(DEFAULT_PDE_VEC_NAMES)
    elif pde_dim == 13:
        pde_vec = _pde_vec_13(law_index)
        pde_names = np.array(OLD_13D_PDE_VEC_NAMES)
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
    eos_classes = np.arange(n) % 2
    eos_one_hot = np.zeros((n, 2), dtype=np.float32)
    eos_one_hot[np.arange(n), eos_classes] = 1.0
    p_inf = np.where(eos_classes == 1, rng.uniform(1.0, 100.0, size=n), 0.0).astype(np.float32)
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
        eos_one_hot,
        p_inf.reshape(n, 1),
    ])
    return vectors.astype(np.float32)


def _realistic_pde_vectors_id(n: int = 64) -> np.ndarray:
    rng = np.random.default_rng(322)
    one_hot = np.zeros((n, 3), dtype=np.float32)
    one_hot[:, 0] = 1.0
    eos_one_hot = np.zeros((n, 2), dtype=np.float32)
    eos_one_hot[:, 0] = 1.0
    vectors = np.column_stack([
        rng.uniform(1.22, 1.36, size=n),
        10 ** rng.uniform(np.log10(8e-4), np.log10(3e-3), size=n),
        10 ** rng.uniform(np.log10(2e-3), np.log10(1.2e-2), size=n),
        10 ** rng.uniform(np.log10(5e-7), np.log10(5e-6), size=n),
        rng.uniform(1.8, 2.8, size=n),
        rng.uniform(90.0, 110.0, size=n),
        rng.uniform(0.9, 1.1, size=n),
        rng.uniform(90.0, 110.0, size=n),
        rng.uniform(4.0, 7.0, size=n),
        one_hot,
        np.full((n, 1), 0.75, dtype=np.float32),
        eos_one_hot,
        np.zeros((n, 1), dtype=np.float32),
    ])
    return vectors.astype(np.float32)


def _realistic_pde_vectors_ood_mild(n: int = 96) -> np.ndarray:
    rng = np.random.default_rng(323)
    laws = np.arange(n) % 3
    one_hot = np.zeros((n, 3), dtype=np.float32)
    one_hot[np.arange(n), laws] = 1.0
    power_law_n = np.full(n, 0.75, dtype=np.float32)
    power_law_n[laws == 2] = rng.uniform(0.55, 0.95, size=int(np.sum(laws == 2)))
    eos_classes = np.arange(n) % 2
    eos_one_hot = np.zeros((n, 2), dtype=np.float32)
    eos_one_hot[np.arange(n), eos_classes] = 1.0
    p_inf = np.zeros(n, dtype=np.float32)
    p_inf[eos_classes == 1] = rng.uniform(1.0, 100.0, size=int(np.sum(eos_classes == 1)))
    vectors = np.column_stack([
        rng.uniform(1.36, 1.45, size=n),
        10 ** rng.uniform(np.log10(3e-3), np.log10(5e-3), size=n),
        10 ** rng.uniform(np.log10(8e-3), np.log10(2e-2), size=n),
        10 ** rng.uniform(np.log10(4e-6), np.log10(9e-6), size=n),
        rng.uniform(1.6, 3.0, size=n),
        rng.uniform(85.0, 115.0, size=n),
        rng.uniform(0.85, 1.15, size=n),
        rng.uniform(85.0, 115.0, size=n),
        rng.uniform(7.0, 8.5, size=n),
        one_hot,
        power_law_n.reshape(n, 1),
        eos_one_hot,
        p_inf.reshape(n, 1),
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


def _stress_robust_continuous_loss() -> None:
    for name, vectors in {
        "id": _realistic_pde_vectors_id(),
        "ood_mild": _realistic_pde_vectors_ood_mild(),
    }.items():
        schema = infer_pde_schema(DEFAULT_PDE_VEC_NAMES, vectors.shape[1])
        normalizer = compute_pde_normalizer(vectors, names=DEFAULT_PDE_VEC_NAMES)
        if not normalizer.get("active_continuous_indices"):
            raise RuntimeError(f"{name} normalizer has no active continuous indices")
        if name == "id":
            skipped = set(normalizer.get("skipped_continuous_names", []))
            if {"power_law_n", "p_inf"} - skipped:
                raise RuntimeError(f"ID should skip meaningless constant dimensions, got {skipped}")

        mean_bias = pde_head_bias_from_normalizer(normalizer).numpy()
        mean_pred = np.repeat(mean_bias[None, :], vectors.shape[0], axis=0).astype(np.float32)
        mean_parts = pde_loss_components(
            torch.from_numpy(mean_pred),
            torch.from_numpy(vectors),
            schema,
            normalizer=normalizer,
            normalize=True,
        )
        mean_loss = float(mean_parts["continuous"].detach())
        if not np.isfinite(mean_loss) or mean_loss > 5.0:
            raise RuntimeError(f"{name} mean predictor continuous loss unstable: {mean_loss}")
        if float(mean_parts["cont_num_active_dims"]) <= 0 or float(mean_parts["cont_num_valid_entries"]) <= 0:
            raise RuntimeError(f"{name} continuous loss reported no active/valid entries")

        rng = np.random.default_rng(324)
        perturbed = mean_pred.copy()
        perturbed += rng.normal(scale=0.02, size=perturbed.shape).astype(np.float32)
        perturbed[:, 1:4] *= rng.uniform(0.95, 1.05, size=(perturbed.shape[0], 3)).astype(np.float32)
        # These dimensions are conditionally meaningful. Extreme mistakes on
        # irrelevant samples must not dominate the loss.
        non_power = vectors[:, 11] < 0.5
        ideal = vectors[:, 13] > 0.5
        perturbed[non_power, 12] = 999.0
        perturbed[ideal, 15] = 999.0
        parts = pde_loss_components(
            torch.from_numpy(perturbed),
            torch.from_numpy(vectors),
            schema,
            normalizer=normalizer,
            normalize=True,
        )
        cont_loss = float(parts["continuous"].detach())
        if not np.isfinite(cont_loss) or cont_loss >= 50.0:
            raise RuntimeError(f"{name} robust continuous loss exploded: {cont_loss}")
        if name == "ood_mild":
            max_possible = len(normalizer.get("active_continuous_indices", [])) * vectors.shape[0]
            if float(parts["cont_num_valid_entries"]) >= max_possible:
                raise RuntimeError("OOD conditional dimensions did not reduce valid continuous entries")
        if not torch.isfinite(parts["law"]) or not torch.isfinite(parts["eos"]):
            raise RuntimeError(f"{name} categorical PDE losses were not finite")


def _test_pde_schema_and_fingerprint() -> None:
    schema9 = infer_pde_schema(pde_dim=9)
    schema13 = infer_pde_schema(pde_dim=13)
    schema16 = infer_pde_schema(pde_dim=16)
    if schema9.get("has_viscosity_law") or schema9.get("has_eos_type"):
        raise RuntimeError("9D schema should not expose categorical PDE slices")
    if not schema13.get("has_viscosity_law") or schema13.get("has_eos_type"):
        raise RuntimeError("13D schema should expose viscosity law but not EOS")
    if not schema16.get("has_viscosity_law") or not schema16.get("has_eos_type"):
        raise RuntimeError("16D schema should expose viscosity law and EOS")
    if schema16.get("p_inf_index") != 15 or 15 not in schema16.get("continuous_indices", []):
        raise RuntimeError("16D schema should treat p_inf as continuous index 15")

    base_cfg = {
        "gamma": 1.4,
        "viscosity": 1e-3,
        "visc_bulk": 2e-3,
        "thermal_cond": 1e-6,
        "C_v": 2.5,
        "T_0": 100.0,
        "rho_inf": 1.0,
        "T_inf": 100.0,
        "v_n_inf": 6.0,
        "viscosity_law": "sutherland",
        "power_law_n": 0.75,
    }
    ideal = pde_fingerprint({**base_cfg, "eos_type": "ideal", "p_inf": 0.0})
    stiffened = pde_fingerprint({**base_cfg, "eos_type": "stiffened_gas", "p_inf": 12.5})
    if ideal.shape != (16,) or stiffened.shape != (16,):
        raise RuntimeError(f"new pde_fingerprint length should be 16, got {ideal.shape}, {stiffened.shape}")
    if pde_fingerprint_names() != list(DEFAULT_PDE_VEC_NAMES):
        raise RuntimeError("pde_fingerprint_names does not match DEFAULT_PDE_VEC_NAMES")
    if np.allclose(ideal, stiffened):
        raise RuntimeError("ideal and stiffened_gas pde fingerprints should differ")
    if not np.allclose(ideal[-3:], np.array([1.0, 0.0, 0.0], dtype=np.float32)):
        raise RuntimeError(f"unexpected ideal EOS suffix: {ideal[-3:]}")
    if not np.allclose(stiffened[-3:], np.array([0.0, 1.0, 12.5], dtype=np.float32)):
        raise RuntimeError(f"unexpected stiffened EOS suffix: {stiffened[-3:]}")


def main() -> None:
    root = Path("datasets") / "_codex_smoke_report1"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        _test_pde_schema_and_fingerprint()
        _stress_pde_normalizer()
        _stress_robust_continuous_loss()
        grid16 = root / "grid16"
        grid13 = root / "grid13"
        grid9 = root / "grid9"
        grid16.mkdir()
        grid13.mkdir()
        grid9.mkdir()
        _write_npz(grid16 / "sim_0000.npz", with_mask=True, pde_dim=16, law_index=0, eos_index=0, p_inf=0.0, family="id")
        _write_npz(grid16 / "sim_0001.npz", with_mask=False, pde_dim=16, law_index=2, eos_index=1, p_inf=20.0, family="ood_mild")
        _write_npz(grid13 / "sim_0000.npz", with_mask=True, pde_dim=13, law_index=0, family="id")
        _write_npz(grid13 / "sim_0001.npz", with_mask=False, pde_dim=13, law_index=2, family="ood_mild")
        _write_npz(grid9 / "sim_0000.npz", with_mask=True, pde_dim=9, law_index=0, family="id")
        _write_npz(grid9 / "sim_0001.npz", with_mask=True, pde_dim=9, law_index=1, family="ood_mild")
        paths = sorted(grid16.glob("*.npz"))
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
            model.pde_head[-1].bias.copy_(pde_head_bias_from_normalizer(pde_normalizer))
        update, pde_pred = model.predict_update_and_pde(x)
        pred = model.integrate_update(batch["context_states"][-1].unsqueeze(0), update[:, -1])
        pred_loss = weighted_masked_mse(pred, target, mask=mask)
        zero_mask_loss = weighted_masked_mse(pred, target, mask=torch.zeros_like(mask))
        if not torch.isfinite(zero_mask_loss) or float(zero_mask_loss.detach()) != 0.0:
            raise RuntimeError("weighted_masked_mse should return finite zero for an all-zero mask")
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
        if pde_metrics["eos_type"] is None:
            raise RuntimeError("16D synthetic pde_vec did not produce EOS metrics")
        _plot_confusion(pde_metrics, root)
        if not (root / "eos_type_confusion.png").exists():
            raise RuntimeError("EOS confusion plot was not written")

        old13_ds = CFDWindowDataset(sorted(grid13.glob("*.npz")), context_length=3, prediction_horizon=1)
        old13_schema = infer_pde_schema(old13_ds.pde_vec_names, old13_ds.pde_dim)
        if not old13_schema.get("has_viscosity_law") or old13_schema.get("has_eos_type"):
            raise RuntimeError("old 13D pde_vec schema compatibility failed")

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
        eval_metrics = evaluate(grid16, ckpt_path, root / "eval", context=3, horizon=1,
                                batch_size=1, device="cpu", rollout_steps="")
        if "pde_identification" not in eval_metrics:
            raise RuntimeError("synthetic evaluation did not produce PDE identification metrics")
        if "by_family" not in eval_metrics["pde_identification"]:
            raise RuntimeError("synthetic evaluation did not produce by_family PDE metrics")
        pde_json = root / "eval" / "pde_metrics.json"
        if not pde_json.exists():
            raise RuntimeError("synthetic evaluation did not write pde_metrics.json")
        eval_pde_payload = json.loads(pde_json.read_text(encoding="utf-8"))
        if eval_pde_payload.get("eos_type") is None:
            raise RuntimeError("synthetic evaluation did not write EOS metrics")
        scaling = evaluate_context_scaling(
            grid16,
            ckpt_path,
            root / "context_scaling",
            contexts=[2, 3, 8],
            batch_size=1,
            device="cpu",
            rollout_steps="",
        )
        if scaling["results"]["2"].get("status") != "success":
            raise RuntimeError("context=2 scaling evaluation did not succeed")
        if scaling["results"]["8"].get("status") != "skipped":
            raise RuntimeError("too-long synthetic context was not skipped")
        if not (root / "context_scaling" / "context_scaling_metrics.json").exists():
            raise RuntimeError("context scaling metrics JSON was not written")
        print(
            "OK synthetic report1 smoke: "
            f"windows={len(ds)} C_in={x.shape[2]} pred_loss={float(pred_loss.detach()):.4e} "
            f"pde_cont={float(pde_parts['continuous'].detach()):.4e} "
            f"pde_law={float(pde_parts['law'].detach()):.4e} "
            f"pde_eos={float(pde_parts['eos'].detach()):.4e} "
            f"old9_loss={float(old_parts['total'].detach()):.4e}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
