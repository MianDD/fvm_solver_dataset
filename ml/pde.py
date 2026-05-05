"""Utilities for PDE-parameter auxiliary learning and evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


BASE_PDE_NAMES = [
    "gamma",
    "viscosity",
    "visc_bulk",
    "thermal_cond",
    "C_v",
    "T_0",
    "rho_inf",
    "T_inf",
    "v_n_inf",
]
VISCOSITY_LAW_NAMES = ["sutherland", "constant", "power_law"]
VISCOSITY_LAW_VEC_NAMES = [f"viscosity_law_{name}" for name in VISCOSITY_LAW_NAMES]
POWER_LAW_N_NAME = "power_law_n"
OLD_13D_PDE_VEC_NAMES = BASE_PDE_NAMES + VISCOSITY_LAW_VEC_NAMES + [POWER_LAW_N_NAME]
EOS_TYPE_NAMES = ["ideal", "stiffened_gas"]
EOS_TYPE_VEC_NAMES = [f"eos_type_{name}" for name in EOS_TYPE_NAMES]
P_INF_NAME = "p_inf"
OLD_16D_PDE_VEC_NAMES = OLD_13D_PDE_VEC_NAMES + EOS_TYPE_VEC_NAMES + [P_INF_NAME]
P_INF_RATIO_NAME = "p_inf_ratio"
DEFAULT_PDE_VEC_NAMES = OLD_16D_PDE_VEC_NAMES + [P_INF_RATIO_NAME]
DEFAULT_LOG_PDE_NAMES = ["viscosity", "visc_bulk", "thermal_cond"]
DEFAULT_LOG_INDICES = [1, 2, 3]
DEFAULT_ACTIVE_STD_THRESHOLD = 1e-4


def default_pde_names(dim: int) -> List[str]:
    """Return stable names for known pde_vec layouts."""
    dim = int(dim)
    if dim == len(DEFAULT_PDE_VEC_NAMES):
        return list(DEFAULT_PDE_VEC_NAMES)
    if dim == len(OLD_16D_PDE_VEC_NAMES):
        return list(OLD_16D_PDE_VEC_NAMES)
    if dim == len(OLD_13D_PDE_VEC_NAMES):
        return list(OLD_13D_PDE_VEC_NAMES)
    if dim == len(BASE_PDE_NAMES):
        return list(BASE_PDE_NAMES)
    return [f"pde_{i}" for i in range(dim)]


def infer_pde_schema(names: Sequence[str] | None = None,
                     pde_dim: int | None = None) -> Dict:
    """Infer continuous and categorical slices from pde_vec names.

    The current 17D layout is the previous 16D layout plus ``p_inf_ratio``.
    The old 9D, 13D, and 16D layouts are inferred explicitly so older datasets
    and checkpoints remain usable.
    """
    if names:
        vec_names = [str(name) for name in names]
        dim = len(vec_names)
    else:
        dim = int(pde_dim or 0)
        vec_names = default_pde_names(dim)
    if pde_dim is not None and len(vec_names) != int(pde_dim):
        vec_names = default_pde_names(int(pde_dim))
    dim = len(vec_names)

    law_indices: List[int] = []
    if all(name in vec_names for name in VISCOSITY_LAW_VEC_NAMES):
        law_indices = [vec_names.index(name) for name in VISCOSITY_LAW_VEC_NAMES]
    eos_indices: List[int] = []
    if all(name in vec_names for name in EOS_TYPE_VEC_NAMES):
        eos_indices = [vec_names.index(name) for name in EOS_TYPE_VEC_NAMES]
    categorical_index_set = set(law_indices) | set(eos_indices)
    continuous_indices = [i for i in range(dim) if i not in categorical_index_set]
    power_law_n_index = vec_names.index(POWER_LAW_N_NAME) if POWER_LAW_N_NAME in vec_names else None
    p_inf_index = vec_names.index(P_INF_NAME) if P_INF_NAME in vec_names else None
    p_inf_ratio_index = vec_names.index(P_INF_RATIO_NAME) if P_INF_RATIO_NAME in vec_names else None
    return {
        "pde_vec_names": vec_names,
        "pde_dim": dim,
        "continuous_indices": continuous_indices,
        "continuous_names": [vec_names[i] for i in continuous_indices],
        "viscosity_law_indices": law_indices,
        "viscosity_law_names": list(VISCOSITY_LAW_NAMES) if law_indices else [],
        "power_law_n_index": power_law_n_index,
        "has_viscosity_law": bool(law_indices),
        "eos_type_indices": eos_indices,
        "eos_type_names": list(EOS_TYPE_NAMES) if eos_indices else [],
        "has_eos_type": bool(eos_indices),
        "p_inf_index": p_inf_index,
        "p_inf_ratio_index": p_inf_ratio_index,
    }


def default_log_indices(names: Sequence[str] | None = None,
                        pde_dim: int | None = None) -> List[int]:
    """Return default log-transform dimensions for transport coefficients."""
    if names:
        vec_names = [str(name) for name in names]
        indices = [vec_names.index(name) for name in DEFAULT_LOG_PDE_NAMES if name in vec_names]
        if indices:
            return indices
        dim = len(vec_names)
    else:
        dim = int(pde_dim or 0)
    return [idx for idx in DEFAULT_LOG_INDICES if idx < dim]


def pde_vectors_from_records(records: Iterable[Dict]) -> np.ndarray:
    vectors = [
        np.asarray(rec["pde_vec"], dtype=np.float32)
        for rec in records
        if bool(rec.get("pde_vec_available", False))
    ]
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(vectors, axis=0).astype(np.float32)


def transform_pde_vectors_np(vectors: np.ndarray,
                             log_indices: Sequence[int] | None = None,
                             log_eps: float = 1e-12) -> np.ndarray:
    """Apply the stored PDE-space transform to numpy arrays.

    Transport coefficients are positive and span orders of magnitude, so the
    default normalizer works in log-space for those dimensions.
    """
    out = np.asarray(vectors, dtype=np.float32).copy()
    if out.ndim == 1:
        out = out[None, :]
    for idx in [int(i) for i in (log_indices or []) if int(i) < out.shape[1]]:
        out[:, idx] = np.log(np.maximum(out[:, idx], float(log_eps)))
    return out.astype(np.float32)


def transform_pde_tensor(values: torch.Tensor,
                         normalizer: Dict | None = None) -> torch.Tensor:
    """Apply a checkpoint-compatible PDE transform to a torch tensor."""
    if normalizer is None:
        return values
    log_indices = [int(i) for i in normalizer.get("log_indices", [])]
    if not log_indices:
        return values
    out = values.clone()
    log_eps = float(normalizer.get("log_eps", 1e-12))
    valid = [idx for idx in log_indices if idx < out.shape[1]]
    if valid:
        idx_t = torch.as_tensor(valid, dtype=torch.long, device=out.device)
        logged = torch.log(out.index_select(1, idx_t).clamp_min(log_eps))
        out.index_copy_(1, idx_t, logged)
    return out


def compute_pde_normalizer(vectors: np.ndarray, min_std: float = 1e-6,
                           log_indices: Sequence[int] | None = None,
                           log_eps: float = 1e-12,
                           log_transport: bool = True,
                           names: Sequence[str] | None = None,
                           active_std_threshold: float | None = None) -> Dict:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Cannot compute PDE normalizer from an empty vector set.")
    if log_indices is None:
        log_indices = default_log_indices(names, vectors.shape[1]) if log_transport else []
    log_indices = [int(i) for i in log_indices if 0 <= int(i) < vectors.shape[1]]
    transformed = transform_pde_vectors_np(vectors, log_indices=log_indices, log_eps=log_eps)
    mean = transformed.mean(axis=0).astype(np.float32)
    transformed_std = transformed.std(axis=0).astype(np.float32)
    std = np.maximum(transformed_std, float(min_std)).astype(np.float32)
    threshold = (
        max(100.0 * float(min_std), DEFAULT_ACTIVE_STD_THRESHOLD)
        if active_std_threshold is None else float(active_std_threshold)
    )
    schema = infer_pde_schema(names, vectors.shape[1])
    continuous_indices = [int(i) for i in schema.get("continuous_indices", [])]
    active_continuous_indices = [
        idx for idx in continuous_indices
        if idx < len(transformed_std) and float(transformed_std[idx]) >= threshold
    ]
    skipped_continuous_indices = [
        idx for idx in continuous_indices
        if idx not in set(active_continuous_indices)
    ]
    vec_names = schema.get("pde_vec_names", default_pde_names(vectors.shape[1]))
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "transformed_std": transformed_std.tolist(),
        "raw_mean": vectors.mean(axis=0).astype(np.float32).tolist(),
        "raw_std": np.maximum(vectors.std(axis=0).astype(np.float32), float(min_std)).tolist(),
        "min_std": float(min_std),
        "log_indices": log_indices,
        "log_eps": float(log_eps),
        "transform": "log_transport" if log_indices else "identity",
        "n_train_vectors": int(vectors.shape[0]),
        "active_std_threshold": threshold,
        "active_continuous_indices": active_continuous_indices,
        "active_continuous_names": [vec_names[i] for i in active_continuous_indices],
        "skipped_continuous_indices": skipped_continuous_indices,
        "skipped_continuous_names": [vec_names[i] for i in skipped_continuous_indices],
    }


def _active_continuous_indices(schema: Dict, normalizer: Dict | None,
                               pred_dim: int,
                               active_std_threshold: float | None = None) -> tuple[List[int], List[int]]:
    """Return continuous dimensions active for loss and skipped by low variance."""
    continuous = [
        int(i) for i in schema.get("continuous_indices", [])
        if 0 <= int(i) < pred_dim
    ]
    if not continuous:
        return [], []
    if normalizer is None:
        return continuous, []
    if "active_continuous_indices" in normalizer:
        active_set = {
            int(i) for i in normalizer.get("active_continuous_indices", [])
            if 0 <= int(i) < pred_dim
        }
        active = [idx for idx in continuous if idx in active_set]
        skipped = [idx for idx in continuous if idx not in active_set]
        return active, skipped
    std = normalizer.get("transformed_std", normalizer.get("std", []))
    min_std = float(normalizer.get("min_std", 1e-6))
    threshold = (
        float(normalizer.get("active_std_threshold"))
        if normalizer.get("active_std_threshold") is not None
        else (max(100.0 * min_std, DEFAULT_ACTIVE_STD_THRESHOLD)
              if active_std_threshold is None else float(active_std_threshold))
    )
    active = [
        idx for idx in continuous
        if idx < len(std) and float(std[idx]) >= threshold
    ]
    skipped = [idx for idx in continuous if idx not in set(active)]
    return active, skipped


def _class_mask(true: torch.Tensor, indices: Sequence[int], names: Sequence[str],
                target_name: str) -> torch.Tensor | None:
    indices = [int(i) for i in indices]
    if not indices or target_name not in names:
        return None
    valid = [idx for idx in indices if idx < true.shape[1]]
    if len(valid) != len(indices):
        return None
    idx_t = torch.as_tensor(valid, dtype=torch.long, device=true.device)
    target_class = true.index_select(1, idx_t).argmax(dim=1)
    class_id = int(list(names).index(target_name))
    return target_class == class_id


def pde_loss_components(pred: torch.Tensor,
                        true: torch.Tensor,
                        schema: Dict,
                        normalizer: Dict | None = None,
                        normalize: bool = True,
                        cont_weight: float = 1.0,
                        law_weight: float = 1.0,
                        eos_weight: float = 1.0,
                        cont_loss_mode: str = "huber",
                        huber_beta: float = 1.0,
                        active_std_threshold: float | None = None) -> Dict[str, torch.Tensor]:
    """Compute schema-aware PDE auxiliary losses.

    The continuous slice is regressed in physical units, but the MSE is computed
    after optional z-scoring by training-set pde_vec statistics.  The viscosity
    law one-hot slice, when present, is interpreted as logits and trained with
    cross entropy.
    """
    if pred is None:
        raise ValueError("pred must not be None when PDE auxiliary loss is enabled.")
    if pred.shape != true.shape:
        raise ValueError(f"pde prediction shape {tuple(pred.shape)} != target {tuple(true.shape)}")
    zero = pred.new_zeros(())
    cont_loss = zero
    law_loss = zero
    law_acc = zero
    eos_loss = zero
    eos_acc = zero

    cont_indices, skipped_cont_indices = _active_continuous_indices(
        schema,
        normalizer if normalize else None,
        pred.shape[1],
        active_std_threshold=active_std_threshold,
    )
    cont_valid_entries = pred.new_zeros(())
    if cont_indices:
        idx = torch.as_tensor(cont_indices, dtype=torch.long, device=pred.device)
        pred_cont = pred.index_select(1, idx)
        true_cont = true.index_select(1, idx)
        if normalize and normalizer is not None:
            pred_t = transform_pde_tensor(pred, normalizer)
            true_t = transform_pde_tensor(true, normalizer)
            pred_cont = pred_t.index_select(1, idx)
            true_cont = true_t.index_select(1, idx)
            mean = torch.as_tensor(normalizer["mean"], dtype=pred.dtype, device=pred.device).index_select(0, idx)
            std = torch.as_tensor(normalizer["std"], dtype=pred.dtype, device=pred.device).index_select(0, idx)
            pred_cont = (pred_cont - mean.view(1, -1)) / std.view(1, -1).clamp_min(1e-6)
            true_cont = (true_cont - mean.view(1, -1)) / std.view(1, -1).clamp_min(1e-6)
        valid_mask = torch.ones_like(pred_cont, dtype=torch.bool)
        power_law_n_index = schema.get("power_law_n_index")
        if power_law_n_index in cont_indices:
            power_mask = _class_mask(
                true,
                schema.get("viscosity_law_indices", []),
                schema.get("viscosity_law_names", []),
                "power_law",
            )
            if power_mask is not None:
                col = cont_indices.index(int(power_law_n_index))
                valid_mask[:, col] &= power_mask
        p_inf_index = schema.get("p_inf_index")
        if p_inf_index in cont_indices:
            stiffened_mask = _class_mask(
                true,
                schema.get("eos_type_indices", []),
                schema.get("eos_type_names", []),
                "stiffened_gas",
            )
            if stiffened_mask is not None:
                col = cont_indices.index(int(p_inf_index))
                valid_mask[:, col] &= stiffened_mask
        p_inf_ratio_index = schema.get("p_inf_ratio_index")
        if p_inf_ratio_index in cont_indices:
            stiffened_mask = _class_mask(
                true,
                schema.get("eos_type_indices", []),
                schema.get("eos_type_names", []),
                "stiffened_gas",
            )
            if stiffened_mask is not None:
                col = cont_indices.index(int(p_inf_ratio_index))
                valid_mask[:, col] &= stiffened_mask

        residual = pred_cont - true_cont
        if cont_loss_mode == "mse":
            per_entry = residual.square()
        elif cont_loss_mode == "huber":
            beta = max(float(huber_beta), 1e-12)
            abs_res = residual.abs()
            per_entry = torch.where(
                abs_res < beta,
                0.5 * residual.square() / beta,
                abs_res - 0.5 * beta,
            )
        else:
            raise ValueError(f"Unknown PDE continuous loss mode {cont_loss_mode!r}; use 'mse' or 'huber'.")
        valid_float = valid_mask.to(dtype=pred.dtype)
        cont_valid_entries = valid_float.sum()
        if bool(cont_valid_entries.detach().cpu().item() > 0):
            cont_loss = (per_entry * valid_float).sum() / cont_valid_entries.clamp_min(1.0)

    law_indices = [int(i) for i in schema.get("viscosity_law_indices", [])]
    if law_indices:
        idx = torch.as_tensor(law_indices, dtype=torch.long, device=pred.device)
        logits = pred.index_select(1, idx)
        target_class = true.index_select(1, idx).argmax(dim=1)
        law_loss = F.cross_entropy(logits, target_class)
        law_acc = (logits.argmax(dim=1) == target_class).to(dtype=pred.dtype).mean()

    eos_indices = [int(i) for i in schema.get("eos_type_indices", [])]
    if eos_indices:
        idx = torch.as_tensor(eos_indices, dtype=torch.long, device=pred.device)
        logits = pred.index_select(1, idx)
        target_class = true.index_select(1, idx).argmax(dim=1)
        eos_loss = F.cross_entropy(logits, target_class)
        eos_acc = (logits.argmax(dim=1) == target_class).to(dtype=pred.dtype).mean()

    total = (
        float(cont_weight) * cont_loss
        + float(law_weight) * law_loss
        + float(eos_weight) * eos_loss
    )
    return {
        "total": total,
        "continuous": cont_loss,
        "cont_loss": cont_loss,
        "law": law_loss,
        "law_accuracy": law_acc.detach(),
        "eos": eos_loss,
        "eos_accuracy": eos_acc.detach(),
        "cont_num_active_dims": pred.new_tensor(float(len(cont_indices))),
        "cont_num_valid_entries": cont_valid_entries.detach(),
        "active_continuous_indices": cont_indices,
        "active_continuous_names": [
            schema.get("pde_vec_names", default_pde_names(pred.shape[1]))[i]
            for i in cont_indices
        ],
        "skipped_continuous_indices": skipped_cont_indices,
        "skipped_continuous_names": [
            schema.get("pde_vec_names", default_pde_names(pred.shape[1]))[i]
            for i in skipped_cont_indices
        ],
    }


def _float_or_none(value: float | np.floating) -> float | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _categorical_metrics(pred: np.ndarray, true: np.ndarray,
                         indices: Sequence[int], names: Sequence[str]) -> Dict | None:
    indices = [int(i) for i in indices]
    if not indices:
        return None
    true_class = true[:, indices].argmax(axis=1)
    pred_class = pred[:, indices].argmax(axis=1)
    n_cls = len(indices)
    confusion = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(true_class, pred_class):
        confusion[int(t), int(p)] += 1
    per_class = {}
    class_counts = {}
    for i, name in enumerate(names):
        denom = int(confusion[i].sum())
        name = str(name)
        class_counts[name] = denom
        per_class[name] = None if denom == 0 else float(confusion[i, i] / denom)
    num_classes_present = int(sum(count > 0 for count in class_counts.values()))
    return {
        "indices": indices,
        "names": [str(name) for name in names],
        "accuracy": float(np.mean(pred_class == true_class)) if len(true_class) else None,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_rows_true_cols_pred": True,
        "per_class_accuracy": per_class,
        "class_counts": class_counts,
        "num_classes_present": num_classes_present,
        "degenerate": num_classes_present <= 1,
    }


def compute_pde_metrics(pred: np.ndarray,
                        true: np.ndarray,
                        schema: Dict | None = None,
                        names: Sequence[str] | None = None,
                        normalizer: Dict | None = None) -> Dict:
    """Return report-friendly PDE identification metrics."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.ndim != 2 or true.ndim != 2 or pred.shape != true.shape:
        raise ValueError(f"Expected matching 2D arrays, got pred={pred.shape}, true={true.shape}")
    schema = schema or infer_pde_schema(names, pred.shape[1])
    cont_indices = [int(i) for i in schema.get("continuous_indices", [])]

    continuous = {
        "indices": cont_indices,
        "names": [schema["pde_vec_names"][i] for i in cont_indices],
        "per_dimension": {},
        "mean_mae": None,
        "mean_rmse": None,
        "mean_r2": None,
        "mean_normalized_mae": None,
    }
    maes: List[float] = []
    rmses: List[float] = []
    r2s: List[float] = []
    nmaes: List[float] = []
    norm_std = None
    log_indices = set()
    pred_transformed = pred
    true_transformed = true
    if normalizer is not None and "std" in normalizer:
        norm_std = np.asarray(normalizer["std"], dtype=np.float64)
        log_indices = {int(i) for i in normalizer.get("log_indices", [])}
        pred_transformed = transform_pde_vectors_np(
            pred,
            log_indices=sorted(log_indices),
            log_eps=float(normalizer.get("log_eps", 1e-12)),
        ).astype(np.float64)
        true_transformed = transform_pde_vectors_np(
            true,
            log_indices=sorted(log_indices),
            log_eps=float(normalizer.get("log_eps", 1e-12)),
        ).astype(np.float64)
    continuous["log_indices"] = sorted(int(i) for i in log_indices if i in cont_indices)
    continuous["r2_space"] = "mixed_raw_log" if log_indices else "raw"

    for i in cont_indices:
        err = pred[:, i] - true[:, i]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        metric_pred = pred_transformed[:, i] if i in log_indices else pred[:, i]
        metric_true = true_transformed[:, i] if i in log_indices else true[:, i]
        metric_err = metric_pred - metric_true
        denom = float(np.sum((metric_true - np.mean(metric_true)) ** 2))
        r2 = None if denom <= 1e-12 else _float_or_none(1.0 - float(np.sum(metric_err ** 2)) / denom)
        log_r2 = r2 if i in log_indices else None
        normalized_mae = None
        if norm_std is not None and i < len(norm_std) and norm_std[i] > 0:
            normalized_mae = float(np.mean(np.abs(metric_err)) / max(float(norm_std[i]), 1e-6))
            nmaes.append(normalized_mae)
        name = schema["pde_vec_names"][i]
        continuous["per_dimension"][name] = {
            "index": i,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "r2_space": "log" if i in log_indices else "raw",
            "log_r2": log_r2,
            "normalized_mae": normalized_mae,
        }
        maes.append(mae)
        rmses.append(rmse)
        if r2 is not None:
            r2s.append(r2)
    if maes:
        continuous["mean_mae"] = float(np.mean(maes))
        continuous["mean_rmse"] = float(np.mean(rmses))
    if r2s:
        continuous["mean_r2"] = float(np.mean(r2s))
    if nmaes:
        continuous["mean_normalized_mae"] = float(np.mean(nmaes))

    law_indices = [int(i) for i in schema.get("viscosity_law_indices", [])]
    law_metrics = _categorical_metrics(
        pred,
        true,
        law_indices,
        list(schema.get("viscosity_law_names", VISCOSITY_LAW_NAMES)),
    )
    eos_indices = [int(i) for i in schema.get("eos_type_indices", [])]
    eos_metrics = _categorical_metrics(
        pred,
        true,
        eos_indices,
        list(schema.get("eos_type_names", EOS_TYPE_NAMES)),
    )

    return {
        "pde_schema": schema,
        "n_samples": int(pred.shape[0]),
        "continuous": continuous,
        "viscosity_law": law_metrics,
        "eos_type": eos_metrics,
        "overall": {
            "mean_continuous_r2": continuous["mean_r2"],
            "mean_continuous_mae": continuous["mean_mae"],
            "law_accuracy": None if law_metrics is None else law_metrics["accuracy"],
            "eos_accuracy": None if eos_metrics is None else eos_metrics["accuracy"],
        },
    }
