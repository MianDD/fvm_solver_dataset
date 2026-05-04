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
DEFAULT_PDE_VEC_NAMES = BASE_PDE_NAMES + VISCOSITY_LAW_VEC_NAMES + [POWER_LAW_N_NAME]
DEFAULT_LOG_PDE_NAMES = ["viscosity", "visc_bulk", "thermal_cond"]
DEFAULT_LOG_INDICES = [1, 2, 3]


def default_pde_names(dim: int) -> List[str]:
    """Return stable names for known pde_vec layouts."""
    dim = int(dim)
    if dim == len(DEFAULT_PDE_VEC_NAMES):
        return list(DEFAULT_PDE_VEC_NAMES)
    if dim == len(BASE_PDE_NAMES):
        return list(BASE_PDE_NAMES)
    return [f"pde_{i}" for i in range(dim)]


def infer_pde_schema(names: Sequence[str] | None = None,
                     pde_dim: int | None = None) -> Dict:
    """Infer continuous and categorical slices from pde_vec names.

    The current 13D layout is:
    continuous base parameters, three viscosity-law one-hot entries, and
    ``power_law_n``.  The old 9D layout has no categorical slice and is treated
    as all-continuous for backward compatibility.
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
    law_index_set = set(law_indices)
    continuous_indices = [i for i in range(dim) if i not in law_index_set]
    power_law_n_index = vec_names.index(POWER_LAW_N_NAME) if POWER_LAW_N_NAME in vec_names else None
    return {
        "pde_vec_names": vec_names,
        "pde_dim": dim,
        "continuous_indices": continuous_indices,
        "continuous_names": [vec_names[i] for i in continuous_indices],
        "viscosity_law_indices": law_indices,
        "viscosity_law_names": list(VISCOSITY_LAW_NAMES) if law_indices else [],
        "power_law_n_index": power_law_n_index,
        "has_viscosity_law": bool(law_indices),
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
                           names: Sequence[str] | None = None) -> Dict:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Cannot compute PDE normalizer from an empty vector set.")
    if log_indices is None:
        log_indices = default_log_indices(names, vectors.shape[1]) if log_transport else []
    log_indices = [int(i) for i in log_indices if 0 <= int(i) < vectors.shape[1]]
    transformed = transform_pde_vectors_np(vectors, log_indices=log_indices, log_eps=log_eps)
    mean = transformed.mean(axis=0).astype(np.float32)
    std = transformed.std(axis=0).astype(np.float32)
    std = np.maximum(std, float(min_std)).astype(np.float32)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "raw_mean": vectors.mean(axis=0).astype(np.float32).tolist(),
        "raw_std": np.maximum(vectors.std(axis=0).astype(np.float32), float(min_std)).tolist(),
        "min_std": float(min_std),
        "log_indices": log_indices,
        "log_eps": float(log_eps),
        "transform": "log_transport" if log_indices else "identity",
        "n_train_vectors": int(vectors.shape[0]),
    }


def pde_loss_components(pred: torch.Tensor,
                        true: torch.Tensor,
                        schema: Dict,
                        normalizer: Dict | None = None,
                        normalize: bool = True,
                        cont_weight: float = 1.0,
                        law_weight: float = 1.0) -> Dict[str, torch.Tensor]:
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

    cont_indices = [int(i) for i in schema.get("continuous_indices", [])]
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
        cont_loss = torch.mean((pred_cont - true_cont) ** 2)

    law_indices = [int(i) for i in schema.get("viscosity_law_indices", [])]
    if law_indices:
        idx = torch.as_tensor(law_indices, dtype=torch.long, device=pred.device)
        logits = pred.index_select(1, idx)
        target_class = true.index_select(1, idx).argmax(dim=1)
        law_loss = F.cross_entropy(logits, target_class)
        law_acc = (logits.argmax(dim=1) == target_class).to(dtype=pred.dtype).mean()

    total = float(cont_weight) * cont_loss + float(law_weight) * law_loss
    return {
        "total": total,
        "continuous": cont_loss,
        "law": law_loss,
        "law_accuracy": law_acc.detach(),
    }


def _float_or_none(value: float | np.floating) -> float | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


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

    law_metrics = None
    law_indices = [int(i) for i in schema.get("viscosity_law_indices", [])]
    if law_indices:
        law_names = list(schema.get("viscosity_law_names", VISCOSITY_LAW_NAMES))
        true_class = true[:, law_indices].argmax(axis=1)
        pred_class = pred[:, law_indices].argmax(axis=1)
        n_cls = len(law_indices)
        confusion = np.zeros((n_cls, n_cls), dtype=np.int64)
        for t, p in zip(true_class, pred_class):
            confusion[int(t), int(p)] += 1
        per_class = {}
        for i, name in enumerate(law_names):
            denom = int(confusion[i].sum())
            per_class[name] = None if denom == 0 else float(confusion[i, i] / denom)
        law_metrics = {
            "indices": law_indices,
            "names": law_names,
            "accuracy": float(np.mean(pred_class == true_class)) if len(true_class) else None,
            "confusion_matrix": confusion.tolist(),
            "confusion_matrix_rows_true_cols_pred": True,
            "per_class_accuracy": per_class,
        }

    return {
        "pde_schema": schema,
        "n_samples": int(pred.shape[0]),
        "continuous": continuous,
        "viscosity_law": law_metrics,
        "overall": {
            "mean_continuous_r2": continuous["mean_r2"],
            "mean_continuous_mae": continuous["mean_mae"],
            "law_accuracy": None if law_metrics is None else law_metrics["accuracy"],
        },
    }
