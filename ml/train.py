"""Training script for the patch-Transformer CFD surrogate."""

from __future__ import annotations

import argparse
import dataclasses as dc
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .boundary import BOUNDARY_CLASS_NAMES
from .dataset import (
    CFDWindowDataset,
    TARGET_CHANNEL_NAMES,
    input_channel_names,
    parse_strides,
    split_paths,
)
from .model import FoundationCFDModel, count_params
from .pde import (
    compute_pde_normalizer,
    infer_pde_schema,
    pde_loss_components,
    pde_vectors_from_records,
)


HORIZON_ERROR = "Only --horizon 1 is currently supported. Multi-horizon training is not implemented yet."


@dc.dataclass
class TrainConfig:
    grid_dir: str = "datasets/grid_main"
    out_dir: str = "checkpoints/run0"
    context_length: int = 4
    prediction_horizon: int = 1
    start_offset: int = 0
    t_start: float | None = None
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-5
    n_epochs: int = 30
    val_frac: float = 0.25
    seed: int = 0
    patch_size: int = 8
    patch_t: int = 1
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.0
    mlp_ratio: float = 4.0
    attention_type: str = "global"
    input_noise_std: float = 0.0
    pushforward_prob: float = 0.0
    rollout_train_steps: int = 1
    rollout_train_weight_decay: float = 1.0
    pos_encoding: str = "learned_absolute"
    use_boundary_channels: bool = False
    boundary_aware_refine: bool = False
    boundary_loss_weight: float = 0.0
    use_mask_channel: bool = False
    mask_loss: bool = False
    pde_aux_loss: bool = False
    pde_aux_weight: float = 0.01
    pde_normalize: bool = True
    pde_log_transport: bool = True
    pde_log_eps: float = 1e-12
    pde_cont_weight: float = 1.0
    pde_law_weight: float = 1.0
    pde_eos_weight: float = 1.0
    pde_cont_loss: str = "huber"
    pde_huber_beta: float = 1.0
    pde_dim: int = 0
    num_workers: int = 0
    save_every: int = 0
    resume: str | None = None
    use_derivatives: bool = False
    derivative_mode: str = "central"
    use_physical_derivatives: bool = True
    prediction_mode: str = "delta"
    integrator: str = "euler"
    strides: str = "1"


def weighted_mse(pred: torch.Tensor, true: torch.Tensor,
                 weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)) -> torch.Tensor:
    w = torch.as_tensor(weights, dtype=pred.dtype, device=pred.device)
    w = w.view(*([1] * (pred.dim() - 3)), -1, 1, 1)
    return (w * (pred - true) ** 2).mean()


def weighted_masked_mse(pred: torch.Tensor, true: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)) -> torch.Tensor:
    w = torch.as_tensor(weights, dtype=pred.dtype, device=pred.device)
    w = w.view(*([1] * (pred.dim() - 3)), -1, 1, 1)
    err = w * (pred - true) ** 2
    if mask is None:
        return err.mean()
    mask = mask.to(dtype=pred.dtype, device=pred.device)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4:
        raise ValueError(f"mask must have shape (B,H,W) or (B,1,H,W), got {tuple(mask.shape)}")
    denom = mask.sum() * pred.shape[1]
    if bool(denom.detach().cpu().item() <= 0):
        return err.sum() * 0.0
    return (err * mask).sum() / denom.clamp_min(1.0)


def boundary_fluid_mask(batch: Dict, device: str | torch.device) -> torch.Tensor | None:
    """Return inlet/outlet/obstacle/channel-wall pixels, excluding solid."""
    boundary = batch.get("boundary_mask")
    if boundary is None:
        return None
    boundary = boundary.to(device=device)
    if boundary.dim() == 3:
        boundary = boundary.unsqueeze(0)
    if boundary.dim() != 4 or boundary.shape[1] < 6:
        return None
    region = boundary[:, 1:5].sum(dim=1)
    solid = boundary[:, 5]
    return ((region > 0.5) & (solid <= 0.5)).to(dtype=torch.float32)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _input_channel_names(use_derivatives: bool) -> List[str]:
    return input_channel_names(use_derivatives, False, TARGET_CHANNEL_NAMES)


def _configured_input_channel_names(cfg: TrainConfig) -> List[str]:
    return input_channel_names(
        cfg.use_derivatives,
        cfg.use_mask_channel,
        TARGET_CHANNEL_NAMES,
        use_boundary_channels=cfg.use_boundary_channels,
    )


def input_raw_channel_indices(cfg: TrainConfig, n_channels: int) -> List[int]:
    names = _configured_input_channel_names(cfg)[:n_channels]
    raw = [i for i, name in enumerate(names) if str(name).startswith("boundary_")]
    if cfg.use_mask_channel:
        raw.append(n_channels - 1)
    return sorted({int(i) for i in raw if 0 <= int(i) < n_channels})


def configured_boundary_channel_indices(cfg: TrainConfig) -> List[int]:
    return [
        i for i, name in enumerate(_configured_input_channel_names(cfg))
        if str(name).startswith("boundary_")
    ]


def compute_channel_stats(loader: DataLoader, n_channels: int,
                          use_fluid_mask: bool = False,
                          mask_channel_index: int | None = None,
                          raw_channel_indices: List[int] | None = None,
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute input-channel statistics, optionally over fluid cells only.

    When gridded files include a binary fluid mask, all non-mask input channels
    are normalised from mask==1 cells only.  This keeps neutral obstacle values
    out of physical and derivative-feature statistics.  The optional static
    mask channel is intentionally left raw by forcing mean=0 and std=1.
    """
    sums = torch.zeros(n_channels)
    sums2 = torch.zeros(n_channels)
    counts = torch.zeros(n_channels)
    mask_idx = None if mask_channel_index is None else int(mask_channel_index)
    if mask_idx is not None and not (0 <= mask_idx < n_channels):
        raise ValueError(f"mask_channel_index={mask_idx} outside n_channels={n_channels}")
    raw_indices = {int(i) for i in (raw_channel_indices or [])}
    if mask_idx is not None:
        raw_indices.add(mask_idx)
    bad_raw = [i for i in raw_indices if not (0 <= i < n_channels)]
    if bad_raw:
        raise ValueError(f"raw_channel_indices={bad_raw} outside n_channels={n_channels}")
    with torch.no_grad():
        for batch in loader:
            x = batch["input_states"]                    # (B, context, C_in, H, W)
            _, _, C, H, W = x.shape
            if C != n_channels:
                raise ValueError(f"expected {n_channels} input channels, got {C}")
            if use_fluid_mask:
                mask = batch["mask"].to(dtype=x.dtype)    # (B, H, W)
                if mask.dim() == 4 and mask.shape[1] == 1:
                    mask = mask[:, 0]
                if mask.shape != (x.shape[0], H, W):
                    raise ValueError(f"mask shape {tuple(mask.shape)} does not match batch grid {(x.shape[0], H, W)}")
                weights = mask[:, None, None, :, :].expand(-1, x.shape[1], 1, -1, -1)
                for ch in range(C):
                    if ch in raw_indices:
                        continue
                    values = x[:, :, ch:ch + 1, :, :]
                    sums[ch] += (values * weights).sum()
                    sums2[ch] += ((values ** 2) * weights).sum()
                    counts[ch] += weights.sum()
            else:
                xx = x.reshape(-1, C, H * W)
                sums += xx.sum(dim=(0, 2))
                sums2 += (xx ** 2).sum(dim=(0, 2))
                counts += xx.shape[0] * xx.shape[2]
    for idx in raw_indices:
        sums[idx] = 0.0
        sums2[idx] = 0.0
        counts[idx] = 1.0
    safe_counts = counts.clamp_min(1.0)
    mean = sums / safe_counts
    var = (sums2 / safe_counts - mean ** 2).clamp_min(1e-6)
    for idx in raw_indices:
        mean[idx] = 0.0
        var[idx] = 1.0
    return mean, var.sqrt()


def make_model(cfg: TrainConfig, C_in: int, H: int, W: int) -> FoundationCFDModel:
    if cfg.patch_t != 1:
        raise RuntimeError(
            "Only --patch-t 1 is implemented in this incremental pass. "
            "Temporal tubelets with patch_t>1 require a decoder reshape change."
        )
    boundary_indices = configured_boundary_channel_indices(cfg)
    if cfg.boundary_aware_refine and not boundary_indices:
        raise RuntimeError("--boundary-aware-refine requires --use-boundary-channels.")
    return FoundationCFDModel(
        n_channels=len(TARGET_CHANNEL_NAMES),
        n_input_channels=C_in,
        n_target_channels=len(TARGET_CHANNEL_NAMES),
        H=H,
        W=W,
        patch_size=cfg.patch_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_context=cfg.context_length,
        dropout=cfg.dropout,
        mlp_ratio=cfg.mlp_ratio,
        attention_type=cfg.attention_type,
        pos_encoding=cfg.pos_encoding,
        pde_dim=cfg.pde_dim if cfg.pde_aux_loss else 0,
        use_boundary_channels=cfg.use_boundary_channels,
        boundary_channels=len(BOUNDARY_CLASS_NAMES),
        boundary_aware_refine=cfg.boundary_aware_refine,
        boundary_channel_start=(boundary_indices[0] if boundary_indices else None),
    )


def model_config_dict(cfg: TrainConfig, C_in: int, H: int, W: int) -> Dict:
    return {
        "n_channels": len(TARGET_CHANNEL_NAMES),
        "n_input_channels": C_in,
        "n_target_channels": len(TARGET_CHANNEL_NAMES),
        "H": H,
        "W": W,
        "patch_size": cfg.patch_size,
        "patch_t": cfg.patch_t,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "max_context": cfg.context_length,
        "dropout": cfg.dropout,
        "mlp_ratio": cfg.mlp_ratio,
        "attention_type": cfg.attention_type,
        "pos_encoding": cfg.pos_encoding,
        "pushforward_prob": cfg.pushforward_prob,
        "rollout_train_steps": cfg.rollout_train_steps,
        "rollout_train_weight_decay": cfg.rollout_train_weight_decay,
        "input_channel_names": _configured_input_channel_names(cfg),
        "target_channel_names": TARGET_CHANNEL_NAMES,
        "use_derivatives": cfg.use_derivatives,
        "use_boundary_channels": cfg.use_boundary_channels,
        "boundary_channels": len(BOUNDARY_CLASS_NAMES),
        "boundary_aware_refine": cfg.boundary_aware_refine,
        "boundary_loss_weight": cfg.boundary_loss_weight,
        "boundary_class_names": list(BOUNDARY_CLASS_NAMES),
        "use_mask_channel": cfg.use_mask_channel,
        "mask_loss": cfg.mask_loss,
        "derivative_mode": cfg.derivative_mode,
        "use_physical_derivatives": cfg.use_physical_derivatives,
        "prediction_mode": cfg.prediction_mode,
        "integrator": cfg.integrator,
        "pde_aux_loss": cfg.pde_aux_loss,
        "pde_aux_weight": cfg.pde_aux_weight,
        "pde_normalize": cfg.pde_normalize if cfg.pde_aux_loss else False,
        "pde_log_transport": cfg.pde_log_transport if cfg.pde_aux_loss else False,
        "pde_log_eps": cfg.pde_log_eps,
        "pde_cont_weight": cfg.pde_cont_weight,
        "pde_law_weight": cfg.pde_law_weight,
        "pde_eos_weight": cfg.pde_eos_weight,
        "pde_cont_loss": cfg.pde_cont_loss,
        "pde_huber_beta": cfg.pde_huber_beta,
        "pde_dim": cfg.pde_dim if cfg.pde_aux_loss else 0,
        "strides": parse_strides(cfg.strides),
    }


def attention_complexity_dict(cfg: TrainConfig, H: int, W: int) -> Dict:
    if H % cfg.patch_size != 0 or W % cfg.patch_size != 0:
        raise ValueError(
            f"H={H}, W={W} must be divisible by patch size {cfg.patch_size} "
            "before attention complexity can be computed."
        )
    T = int(cfg.context_length)
    n_patches = (H // cfg.patch_size) * (W // cfg.patch_size)
    total_tokens = T * n_patches
    global_pairs = total_tokens ** 2
    factorized_pairs = T * (n_patches ** 2) + n_patches * (T ** 2)
    reduction = float(global_pairs / factorized_pairs) if factorized_pairs else float("inf")
    return {
        "attention_type": cfg.attention_type,
        "context_length": T,
        "patches_per_frame": n_patches,
        "total_tokens": total_tokens,
        "global_pair_count": global_pairs,
        "factorized_pair_count": factorized_pairs,
        "factorized_reduction_vs_global": reduction,
    }


def print_attention_complexity(complexity: Dict) -> None:
    print(
        "Attention complexity: "
        f"T={complexity['context_length']} "
        f"N={complexity['patches_per_frame']} "
        f"tokens={complexity['total_tokens']}"
    )
    print(
        "  pair counts: "
        f"global={(complexity['global_pair_count']):,} "
        f"factorized={(complexity['factorized_pair_count']):,} "
        f"reduction={complexity['factorized_reduction_vs_global']:.2f}x"
    )


def pde_categorical_class_counts(vectors: np.ndarray, schema: Dict,
                                 indices_key: str, names_key: str) -> Dict:
    indices = [int(i) for i in schema.get(indices_key, [])]
    names = [str(name) for name in schema.get(names_key, [])]
    if not indices:
        return {}
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or any(idx >= values.shape[1] for idx in indices):
        return {}
    classes = values[:, indices].argmax(axis=1)
    counts = {
        name: int(np.sum(classes == i))
        for i, name in enumerate(names)
    }
    present = [name for name, count in counts.items() if count > 0]
    return {
        "indices": indices,
        "names": names,
        "counts": counts,
        "num_classes_present": len(present),
        "present_classes": present,
        "degenerate": len(present) <= 1,
    }


def _warn_if_single_class(label: str, counts: Dict) -> None:
    if not counts or not counts.get("degenerate"):
        return
    present = counts.get("present_classes", [])
    class_name = present[0] if present else "none"
    metric = "law_accuracy" if label == "viscosity-law" else "eos_accuracy"
    print(
        f"WARNING: training set contains only one {label} class: {class_name}. "
        f"{metric} will be trivial and should not be reported as PDE-form "
        "identification. Use ood_mild/mixed data for this metric."
    )


def pde_head_bias_from_normalizer(pde_normalizer: Dict) -> torch.Tensor:
    """Initialise raw-space head bias so transformed bias is near train mean."""
    raw_mean = list(pde_normalizer.get("raw_mean", pde_normalizer.get("mean", [])))
    transformed_mean = pde_normalizer.get("mean", raw_mean)
    log_eps = float(pde_normalizer.get("log_eps", 1e-12))
    for idx in [int(i) for i in pde_normalizer.get("log_indices", [])]:
        if 0 <= idx < len(raw_mean) and idx < len(transformed_mean):
            raw_mean[idx] = max(float(np.exp(float(transformed_mean[idx]))), log_eps)
    return torch.as_tensor(raw_mean, dtype=torch.float32)


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(path: Path, model: FoundationCFDModel, optim, sched,
                    epoch: int, cfg: TrainConfig, history: Dict,
                    train_paths: List[str], val_paths: List[str],
                    model_config: Dict, best_val: float) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": sched.state_dict(),
        "epoch": epoch,
        "config": dc.asdict(cfg),
        "model_config": model_config,
        "history": history,
        "train_paths": train_paths,
        "val_paths": val_paths,
        "best_val_loss": best_val,
        "normalizer": {
            "channel_names": model_config["input_channel_names"],
            "mean": model.normaliser.mean.detach().cpu().tolist(),
            "std": model.normaliser.std.detach().cpu().tolist(),
            "masked_channel_stats": bool(model_config.get("masked_channel_stats", False)),
            "mask_channel_raw": bool(model_config.get("mask_channel_raw", False)),
            "mask_channel_index": (
                model_config["n_input_channels"] - 1
                if model_config.get("mask_channel_raw", False) else None
            ),
            "boundary_channel_raw": bool(model_config.get("boundary_channel_raw", False)),
            "boundary_channel_indices": model_config.get("boundary_channel_indices", []),
            "boundary_class_names": model_config.get("boundary_class_names", []),
        },
        "pde_schema": model_config.get("pde_schema", {}),
        "pde_normalizer": model_config.get("pde_normalizer"),
        "pde_class_counts": model_config.get("pde_class_counts", {}),
    }, path)


def _maybe_apply_pushforward(
    model: FoundationCFDModel,
    features: torch.Tensor,
    context_states: torch.Tensor,
    dt: torch.Tensor,
    cfg: TrainConfig,
    training: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Detached one-step pushforward augmentation for rollout robustness.

    The dataset feature layout always starts with the four physical channels
    [V_x, V_y, rho, T].  Derivative features and the optional mask channel are
    intentionally left unchanged after the replacement; this keeps the
    augmentation cheap and makes it an approximate pushforward-noise regularizer
    rather than a full feature-rebuild rollout.
    """
    if (not training) or cfg.pushforward_prob <= 0.0 or features.shape[1] < 2:
        return features, context_states
    if torch.rand((), device=features.device).item() >= cfg.pushforward_prob:
        return features, context_states

    with torch.no_grad():
        prev_features = features[:, :-1]
        prev_current = context_states[:, -2]
        prev_update, _ = model.predict_update_and_pde(prev_features, normalised=False)
        pseudo_update = prev_update[:, -1]
        pseudo_state = model.integrate_update(
            prev_current,
            pseudo_update,
            prediction_mode=cfg.prediction_mode,
            integrator=cfg.integrator,
            dt=dt,
        ).detach()

    n_phys = min(int(getattr(model, "C", len(TARGET_CHANNEL_NAMES))), features.shape[2], context_states.shape[2])
    augmented_features = features.clone()
    augmented_context = context_states.clone()
    augmented_features[:, -1, :n_phys] = pseudo_state[:, :n_phys]
    augmented_context[:, -1, :n_phys] = pseudo_state[:, :n_phys]
    return augmented_features, augmented_context


def predict_next(model: FoundationCFDModel, batch: Dict, cfg: TrainConfig,
                 device: str, training: bool = False) -> Tuple[
                     torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
                 ]:
    features = batch["input_states"].to(device)
    context_states = batch["context_states"].to(device)
    target = batch["target_states"][:, 0].to(device)
    dt = batch["dt"].to(device)
    if training and cfg.input_noise_std > 0.0:
        channel_scale = model.normaliser.std.view(1, 1, -1, 1, 1).to(
            dtype=features.dtype,
            device=features.device,
        )
        raw_indices = input_raw_channel_indices(cfg, features.shape[2])
        if raw_indices:
            channel_scale = channel_scale.clone()
            channel_scale[:, :, raw_indices, :, :] = 0.0
        features = features + cfg.input_noise_std * channel_scale * torch.randn_like(features)
    features, context_states = _maybe_apply_pushforward(
        model,
        features,
        context_states,
        dt,
        cfg,
        training=training,
    )
    current = context_states[:, -1]
    update, pred_pde = model.predict_update_and_pde(features, normalised=False)
    update = update[:, -1]
    pred_next = model.integrate_update(
        current,
        update,
        prediction_mode=cfg.prediction_mode,
        integrator=cfg.integrator,
        dt=dt,
    )
    true_pde = batch["pde_vec"].to(device) if cfg.pde_aux_loss else None
    return pred_next, target, pred_pde, true_pde


def train(cfg: TrainConfig, device: str | None = None) -> Dict:
    if cfg.prediction_horizon != 1:
        raise ValueError(HORIZON_ERROR)
    if cfg.input_noise_std < 0.0:
        raise ValueError("--input-noise-std must be non-negative.")
    if not 0.0 <= cfg.pushforward_prob <= 1.0:
        raise ValueError("--pushforward-prob must be between 0 and 1.")
    if cfg.rollout_train_steps < 1:
        raise ValueError("--rollout-train-steps must be at least 1.")
    if cfg.rollout_train_steps != 1:
        raise ValueError(
            "--rollout-train-steps > 1 is not implemented in this pass. "
            "Use --pushforward-prob for rollout-stability augmentation."
        )
    if cfg.rollout_train_weight_decay < 0.0:
        raise ValueError("--rollout-train-weight-decay must be non-negative.")
    if cfg.pde_aux_weight < 0.0:
        raise ValueError("--pde-aux-weight must be non-negative.")
    if cfg.pde_cont_weight < 0.0:
        raise ValueError("--pde-cont-weight must be non-negative.")
    if cfg.pde_law_weight < 0.0:
        raise ValueError("--pde-law-weight must be non-negative.")
    if cfg.pde_eos_weight < 0.0:
        raise ValueError("--pde-eos-weight must be non-negative.")
    if cfg.boundary_loss_weight < 0.0:
        raise ValueError("--boundary-loss-weight must be non-negative.")
    if cfg.boundary_aware_refine and not cfg.use_boundary_channels:
        raise ValueError("--boundary-aware-refine requires --use-boundary-channels.")
    if cfg.pde_cont_loss not in {"mse", "huber"}:
        raise ValueError("--pde-cont-loss must be either 'mse' or 'huber'.")
    if cfg.pde_huber_beta <= 0.0:
        raise ValueError("--pde-huber-beta must be positive.")
    if cfg.pde_log_eps <= 0.0:
        raise ValueError("--pde-log-eps must be positive.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    parse_strides(cfg.strides)
    save_json(out_dir / "train_config.json", dc.asdict(cfg))
    resume_ckpt = torch.load(cfg.resume, map_location="cpu") if cfg.resume else None

    train_paths, val_paths = split_paths(cfg.grid_dir, val_frac=cfg.val_frac, seed=cfg.seed)
    print(f"#train sims = {len(train_paths)}   #val sims = {len(val_paths)}")
    if not train_paths or not val_paths:
        raise RuntimeError("Need at least 2 sims (1 train, 1 val).")

    train_ds = CFDWindowDataset(
        train_paths,
        context_length=cfg.context_length,
        prediction_horizon=cfg.prediction_horizon,
        strides=cfg.strides,
        use_derivatives=cfg.use_derivatives,
        derivative_mode=cfg.derivative_mode,
        use_boundary_channels=cfg.use_boundary_channels,
        use_mask_channel=cfg.use_mask_channel,
        start_offset=cfg.start_offset,
        t_start=cfg.t_start,
    )
    val_ds = CFDWindowDataset(
        val_paths,
        context_length=cfg.context_length,
        prediction_horizon=cfg.prediction_horizon,
        strides=cfg.strides,
        use_derivatives=cfg.use_derivatives,
        derivative_mode=cfg.derivative_mode,
        use_boundary_channels=cfg.use_boundary_channels,
        use_mask_channel=cfg.use_mask_channel,
        start_offset=cfg.start_offset,
        t_start=cfg.t_start,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            "No train/val windows. Reduce --context/--horizon/--strides, "
            "lower --start-offset/--t-start, or add snapshots."
        )
    print(f"#train windows = {len(train_ds)}   #val windows = {len(val_ds)}")
    print(f"Window start filter: start_offset={cfg.start_offset}  t_start={cfg.t_start}")
    if cfg.mask_loss and not (train_ds.has_masks and val_ds.has_masks):
        print("WARNING: mask loss requested but one or more grid files lack masks; missing masks use all-fluid defaults.")
    if cfg.use_mask_channel and not (train_ds.has_masks and val_ds.has_masks):
        print("WARNING: mask channel requested but one or more grid files lack masks; missing masks use all-fluid defaults.")
    if cfg.use_boundary_channels and not (train_ds.has_boundary_masks and val_ds.has_boundary_masks):
        print(
            "WARNING: boundary channels requested but one or more grid files lack "
            "boundary_mask; missing masks use a fluid/solid default without inlet/outlet/wall classes."
        )
    if cfg.boundary_loss_weight > 0.0 and not (train_ds.has_boundary_masks and val_ds.has_boundary_masks):
        print(
            "WARNING: boundary-weighted loss requested but one or more grid files lack "
            "boundary_mask; boundary loss may be zero on default fluid/solid masks."
        )
    pde_schema: Dict = {}
    pde_normalizer: Dict | None = None
    pde_class_counts: Dict = {}
    if cfg.pde_aux_loss:
        if not (train_ds.has_pde_vec and val_ds.has_pde_vec):
            raise RuntimeError("--pde-aux-loss requested but at least one grid file is missing pde_vec.")
        inferred_pde_dim = int(train_ds.pde_dim)
        if inferred_pde_dim <= 0 or val_ds.pde_dim != inferred_pde_dim:
            raise RuntimeError(
                "--pde-aux-loss requested but pde_vec lengths are missing or inconsistent "
                "between train/validation grid files."
            )
        if cfg.pde_dim <= 0:
            cfg.pde_dim = inferred_pde_dim
        if cfg.pde_dim != inferred_pde_dim:
            raise RuntimeError(f"--pde-dim={cfg.pde_dim} does not match dataset pde_vec length {inferred_pde_dim}.")
        pde_schema = infer_pde_schema(train_ds.pde_vec_names, cfg.pde_dim)
        pde_vectors = pde_vectors_from_records(train_ds._cache)
        pde_class_counts = {
            "viscosity_law": pde_categorical_class_counts(
                pde_vectors,
                pde_schema,
                "viscosity_law_indices",
                "viscosity_law_names",
            ),
            "eos_type": pde_categorical_class_counts(
                pde_vectors,
                pde_schema,
                "eos_type_indices",
                "eos_type_names",
            ),
        }
        _warn_if_single_class("viscosity-law", pde_class_counts["viscosity_law"])
        _warn_if_single_class("EOS", pde_class_counts["eos_type"])
        if cfg.pde_normalize:
            if resume_ckpt is not None and resume_ckpt.get("pde_normalizer"):
                pde_normalizer = resume_ckpt["pde_normalizer"]
                print("Loaded PDE auxiliary normalizer from resume checkpoint.")
            else:
                pde_normalizer = compute_pde_normalizer(
                    pde_vectors,
                    min_std=1e-6,
                    log_transport=cfg.pde_log_transport,
                    log_eps=cfg.pde_log_eps,
                    names=pde_schema.get("pde_vec_names"),
                )
        save_json(out_dir / "train_config.json", dc.asdict(cfg))
    physical_spacing_used = bool(
        cfg.use_derivatives and train_ds.uses_physical_spacing and val_ds.uses_physical_spacing
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    sample = train_ds[0]
    _, C_in, H, W = sample["input_states"].shape
    print(f"Input shape: C_in={C_in} H={H} W={W}")
    print(
        f"Prediction mode: {cfg.prediction_mode}  integrator: {cfg.integrator}  "
        f"strides={cfg.strides}  attention={cfg.attention_type}"
    )
    print(
        f"Mask channel: {'enabled' if cfg.use_mask_channel else 'disabled'}  "
        f"mask loss: {'enabled' if cfg.mask_loss else 'disabled'}"
    )
    print(f"Boundary one-hot channels: {'enabled' if cfg.use_boundary_channels else 'disabled'}")
    print(
        f"Boundary loss weight: {cfg.boundary_loss_weight:g}  "
        f"boundary-aware refine: {'enabled' if cfg.boundary_aware_refine else 'disabled'}"
    )
    print(
        f"PDE aux loss: {'enabled' if cfg.pde_aux_loss else 'disabled'}"
        + (f"  pde_dim={cfg.pde_dim} weight={cfg.pde_aux_weight:g}" if cfg.pde_aux_loss else "")
    )
    if cfg.pde_aux_loss:
        law_status = "enabled" if pde_schema.get("has_viscosity_law") else "not available"
        eos_status = "enabled" if pde_schema.get("has_eos_type") else "not available"
        print(
            f"PDE aux normalization: {'enabled' if cfg.pde_normalize else 'disabled'}  "
            f"pde_dim={cfg.pde_dim}  law_classification={law_status}  "
            f"eos_classification={eos_status}"
        )
        print(f"  pde_mean: {np.array(pde_normalizer['mean']).round(6).tolist() if pde_normalizer else 'not used'}")
        print(f"  pde_std : {np.array(pde_normalizer['std']).round(6).tolist() if pde_normalizer else 'not used'}")
        if pde_normalizer:
            log_indices = [int(i) for i in pde_normalizer.get("log_indices", [])]
            log_names = [pde_schema.get("pde_vec_names", [])[i] for i in log_indices]
            print(f"  pde_log_indices: {log_indices}  names={log_names}  eps={pde_normalizer.get('log_eps')}")
            print(
                "  pde_active_continuous: "
                f"{pde_normalizer.get('active_continuous_indices', [])}  "
                f"skipped={pde_normalizer.get('skipped_continuous_names', [])}"
            )
        if pde_class_counts:
            print(f"  pde_class_counts: {pde_class_counts}")
    print(f"Input noise std: {cfg.input_noise_std:g} normalizer-std units (training only)")
    pushforward_active = cfg.pushforward_prob > 0.0 and cfg.context_length >= 2
    print(
        f"Pushforward augmentation: {'active' if pushforward_active else 'disabled'}  "
        f"prob={cfg.pushforward_prob:g}  context={cfg.context_length}"
    )
    print(
        "Rollout training loss: disabled "
        f"(steps={cfg.rollout_train_steps}, weight_decay={cfg.rollout_train_weight_decay:g})"
    )
    if cfg.pushforward_prob > 0.0 and cfg.context_length < 2:
        print("WARNING: --pushforward-prob requested but context length < 2; augmentation is disabled.")
    if pushforward_active and cfg.use_derivatives:
        print(
            "NOTE: pushforward replaces only the last-frame physical channels; "
            "derivative features are preserved as an approximate rollout-noise augmentation."
        )
    attention_complexity = attention_complexity_dict(cfg, H, W)
    print_attention_complexity(attention_complexity)

    model = make_model(cfg, C_in, H, W).to(device)
    if cfg.pde_aux_loss and pde_normalizer is not None and model.pde_head is not None and not cfg.resume:
        with torch.no_grad():
            model.pde_head[-1].weight.zero_()
            bias = pde_head_bias_from_normalizer(pde_normalizer).to(device)
            model.pde_head[-1].bias.copy_(bias)
    model_cfg = model_config_dict(cfg, C_in, H, W)
    model_cfg["physical_derivative_spacing"] = "physical" if physical_spacing_used else "index_or_unused"
    model_cfg["attention_complexity"] = attention_complexity
    model_cfg["pde_schema"] = pde_schema
    model_cfg["pde_normalizer"] = pde_normalizer
    model_cfg["pde_class_counts"] = pde_class_counts
    model_cfg["masked_channel_stats"] = bool(train_ds.has_masks)
    model_cfg["mask_channel_raw"] = bool(cfg.use_mask_channel)
    model_cfg["boundary_channel_raw"] = bool(cfg.use_boundary_channels)
    model_cfg["boundary_channel_indices"] = configured_boundary_channel_indices(cfg)
    model_cfg["boundary_channel_start"] = (
        int(model_cfg["boundary_channel_indices"][0])
        if model_cfg["boundary_channel_indices"] else None
    )
    save_json(out_dir / "model_config.json", model_cfg)
    if cfg.pde_aux_loss:
        save_json(out_dir / "pde_normalizer.json", {
            "enabled": bool(cfg.pde_normalize),
            "pde_schema": pde_schema,
            "pde_normalizer": pde_normalizer,
            "pde_log_transport": cfg.pde_log_transport,
            "pde_log_eps": cfg.pde_log_eps,
            "pde_cont_weight": cfg.pde_cont_weight,
            "pde_law_weight": cfg.pde_law_weight,
            "pde_eos_weight": cfg.pde_eos_weight,
            "pde_cont_loss": cfg.pde_cont_loss,
            "pde_huber_beta": cfg.pde_huber_beta,
            "pde_class_counts": pde_class_counts,
        })
    print(f"Model params: {count_params(model):,}")

    mask_channel_index = C_in - 1 if cfg.use_mask_channel else None
    boundary_channel_indices = list(model_cfg.get("boundary_channel_indices", []))
    mean, std = compute_channel_stats(
        train_loader,
        C_in,
        use_fluid_mask=bool(train_ds.has_masks),
        mask_channel_index=mask_channel_index,
        raw_channel_indices=boundary_channel_indices,
    )
    model.normaliser.mean.copy_(mean.to(device))
    model.normaliser.std.copy_(std.to(device))
    model.normaliser._initialised = True
    normalizer = {
        "channel_names": model_cfg["input_channel_names"],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "masked_channel_stats": bool(train_ds.has_masks),
        "mask_channel_raw": bool(cfg.use_mask_channel),
        "mask_channel_index": mask_channel_index,
        "boundary_channel_raw": bool(cfg.use_boundary_channels),
        "boundary_channel_indices": boundary_channel_indices,
        "boundary_class_names": list(BOUNDARY_CLASS_NAMES),
    }
    save_json(out_dir / "normalizer.json", normalizer)
    if train_ds.has_masks:
        print("  channel stats: using fluid-mask cells only for non-mask input channels")
    else:
        print("  channel stats: no saved masks found; using all grid cells")
    if cfg.use_mask_channel:
        print("  mask channel normalisation: raw binary channel (mean=0, std=1)")
    if cfg.use_boundary_channels:
        print("  boundary channel normalisation: raw one-hot channels (mean=0, std=1)")
    print(f"  input means: {mean.numpy().round(3)}")
    print(f"  input stds : {std.numpy().round(3)}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, cfg.n_epochs * len(train_loader))
    )

    history: Dict[str, List] = {
        "train_loss": [],
        "val_loss": [],
        "train_pred_loss": [],
        "val_pred_loss": [],
        "train_boundary_loss": [],
        "val_boundary_loss": [],
        "boundary_loss_weight": [],
        "train_pde_loss": [],
        "val_pde_loss": [],
        "train_pde_cont_loss": [],
        "val_pde_cont_loss": [],
        "train_pde_law_loss": [],
        "val_pde_law_loss": [],
        "train_pde_law_acc": [],
        "val_pde_law_acc": [],
        "train_pde_eos_loss": [],
        "val_pde_eos_loss": [],
        "train_pde_eos_acc": [],
        "val_pde_eos_acc": [],
        "epoch_time": [],
        "lr": [],
    }
    start_epoch = 0
    best_val = float("inf")
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        history = ckpt.get("history", history)
        for key, value in {
            "train_pde_cont_loss": [],
            "val_pde_cont_loss": [],
            "train_boundary_loss": [],
            "val_boundary_loss": [],
            "boundary_loss_weight": [],
            "train_pde_law_loss": [],
            "val_pde_law_loss": [],
            "train_pde_law_acc": [],
            "val_pde_law_acc": [],
            "train_pde_eos_loss": [],
            "val_pde_eos_loss": [],
            "train_pde_eos_acc": [],
            "val_pde_eos_acc": [],
        }.items():
            history.setdefault(key, value)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_val_loss", best_val))
        print(f"Resumed from {cfg.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.n_epochs):
        t0 = time.time()
        model.train()
        running = 0.0
        running_pred = 0.0
        running_boundary = 0.0
        running_pde = 0.0
        running_pde_cont = 0.0
        running_pde_law = 0.0
        running_pde_law_acc = 0.0
        running_pde_eos = 0.0
        running_pde_eos_acc = 0.0
        n_batches = 0
        for batch in train_loader:
            pred, tgt, pred_pde, true_pde = predict_next(model, batch, cfg, device, training=True)
            mask = batch["target_mask"].to(device) if cfg.mask_loss else None
            pred_loss = weighted_masked_mse(pred, tgt, mask=mask)
            if cfg.boundary_loss_weight > 0.0:
                boundary_mask = boundary_fluid_mask(batch, device)
                boundary_loss = weighted_masked_mse(pred, tgt, mask=boundary_mask)
            else:
                boundary_loss = torch.zeros((), dtype=pred_loss.dtype, device=pred_loss.device)
            if cfg.pde_aux_loss:
                pde_comp = pde_loss_components(
                    pred_pde,
                    true_pde,
                    pde_schema,
                    normalizer=pde_normalizer,
                    normalize=cfg.pde_normalize,
                    cont_weight=cfg.pde_cont_weight,
                    law_weight=cfg.pde_law_weight,
                    eos_weight=cfg.pde_eos_weight,
                    cont_loss_mode=cfg.pde_cont_loss,
                    huber_beta=cfg.pde_huber_beta,
                )
                pde_loss = pde_comp["total"]
                pde_cont_loss = pde_comp["continuous"]
                pde_law_loss = pde_comp["law"]
                pde_law_acc = pde_comp["law_accuracy"]
                pde_eos_loss = pde_comp["eos"]
                pde_eos_acc = pde_comp["eos_accuracy"]
                loss = pred_loss + cfg.boundary_loss_weight * boundary_loss + cfg.pde_aux_weight * pde_loss
            else:
                pde_loss = torch.zeros((), dtype=pred_loss.dtype, device=pred_loss.device)
                pde_cont_loss = pde_loss
                pde_law_loss = pde_loss
                pde_law_acc = pde_loss
                pde_eos_loss = pde_loss
                pde_eos_acc = pde_loss
                loss = pred_loss + cfg.boundary_loss_weight * boundary_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += loss.item()
            running_pred += pred_loss.item()
            running_boundary += boundary_loss.item()
            running_pde += pde_loss.item()
            running_pde_cont += pde_cont_loss.item()
            running_pde_law += pde_law_loss.item()
            running_pde_law_acc += pde_law_acc.item()
            running_pde_eos += pde_eos_loss.item()
            running_pde_eos_acc += pde_eos_acc.item()
            n_batches += 1
        train_loss = running / max(1, n_batches)
        train_pred_loss = running_pred / max(1, n_batches)
        train_boundary_loss = running_boundary / max(1, n_batches)
        train_pde_loss = running_pde / max(1, n_batches)
        train_pde_cont_loss = running_pde_cont / max(1, n_batches)
        train_pde_law_loss = running_pde_law / max(1, n_batches)
        train_pde_law_acc = running_pde_law_acc / max(1, n_batches)
        train_pde_eos_loss = running_pde_eos / max(1, n_batches)
        train_pde_eos_acc = running_pde_eos_acc / max(1, n_batches)

        model.eval()
        with torch.no_grad():
            val_running = 0.0
            val_pred_running = 0.0
            val_boundary_running = 0.0
            val_pde_running = 0.0
            val_pde_cont_running = 0.0
            val_pde_law_running = 0.0
            val_pde_law_acc_running = 0.0
            val_pde_eos_running = 0.0
            val_pde_eos_acc_running = 0.0
            val_batches = 0
            for batch in val_loader:
                pred, tgt, pred_pde, true_pde = predict_next(model, batch, cfg, device)
                mask = batch["target_mask"].to(device) if cfg.mask_loss else None
                pred_loss = weighted_masked_mse(pred, tgt, mask=mask)
                if cfg.boundary_loss_weight > 0.0:
                    boundary_mask = boundary_fluid_mask(batch, device)
                    boundary_loss = weighted_masked_mse(pred, tgt, mask=boundary_mask)
                else:
                    boundary_loss = torch.zeros((), dtype=pred_loss.dtype, device=pred_loss.device)
                if cfg.pde_aux_loss:
                    pde_comp = pde_loss_components(
                        pred_pde,
                        true_pde,
                        pde_schema,
                        normalizer=pde_normalizer,
                        normalize=cfg.pde_normalize,
                        cont_weight=cfg.pde_cont_weight,
                        law_weight=cfg.pde_law_weight,
                        eos_weight=cfg.pde_eos_weight,
                        cont_loss_mode=cfg.pde_cont_loss,
                        huber_beta=cfg.pde_huber_beta,
                    )
                    pde_loss = pde_comp["total"]
                    pde_cont_loss = pde_comp["continuous"]
                    pde_law_loss = pde_comp["law"]
                    pde_law_acc = pde_comp["law_accuracy"]
                    pde_eos_loss = pde_comp["eos"]
                    pde_eos_acc = pde_comp["eos_accuracy"]
                    loss = pred_loss + cfg.boundary_loss_weight * boundary_loss + cfg.pde_aux_weight * pde_loss
                else:
                    pde_loss = torch.zeros((), dtype=pred_loss.dtype, device=pred_loss.device)
                    pde_cont_loss = pde_loss
                    pde_law_loss = pde_loss
                    pde_law_acc = pde_loss
                    pde_eos_loss = pde_loss
                    pde_eos_acc = pde_loss
                    loss = pred_loss + cfg.boundary_loss_weight * boundary_loss
                val_running += loss.item()
                val_pred_running += pred_loss.item()
                val_boundary_running += boundary_loss.item()
                val_pde_running += pde_loss.item()
                val_pde_cont_running += pde_cont_loss.item()
                val_pde_law_running += pde_law_loss.item()
                val_pde_law_acc_running += pde_law_acc.item()
                val_pde_eos_running += pde_eos_loss.item()
                val_pde_eos_acc_running += pde_eos_acc.item()
                val_batches += 1
            val_loss = val_running / max(1, val_batches)
            val_pred_loss = val_pred_running / max(1, val_batches)
            val_boundary_loss = val_boundary_running / max(1, val_batches)
            val_pde_loss = val_pde_running / max(1, val_batches)
            val_pde_cont_loss = val_pde_cont_running / max(1, val_batches)
            val_pde_law_loss = val_pde_law_running / max(1, val_batches)
            val_pde_law_acc = val_pde_law_acc_running / max(1, val_batches)
            val_pde_eos_loss = val_pde_eos_running / max(1, val_batches)
            val_pde_eos_acc = val_pde_eos_acc_running / max(1, val_batches)

        epoch_time = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_pred_loss"].append(train_pred_loss)
        history["val_pred_loss"].append(val_pred_loss)
        history["train_boundary_loss"].append(train_boundary_loss)
        history["val_boundary_loss"].append(val_boundary_loss)
        history["boundary_loss_weight"].append(float(cfg.boundary_loss_weight))
        history["train_pde_loss"].append(train_pde_loss)
        history["val_pde_loss"].append(val_pde_loss)
        history["train_pde_cont_loss"].append(train_pde_cont_loss)
        history["val_pde_cont_loss"].append(val_pde_cont_loss)
        history["train_pde_law_loss"].append(train_pde_law_loss)
        history["val_pde_law_loss"].append(val_pde_law_loss)
        history["train_pde_law_acc"].append(train_pde_law_acc)
        history["val_pde_law_acc"].append(val_pde_law_acc)
        history["train_pde_eos_loss"].append(train_pde_eos_loss)
        history["val_pde_eos_loss"].append(val_pde_eos_loss)
        history["train_pde_eos_acc"].append(train_pde_eos_acc)
        history["val_pde_eos_acc"].append(val_pde_eos_acc)
        history["epoch_time"].append(epoch_time)
        history["lr"].append(float(optim.param_groups[0]["lr"]))
        print(
            f"[ep {epoch:3d}] train={train_loss:.4e} val={val_loss:.4e} "
            f"pred={train_pred_loss:.4e}/{val_pred_loss:.4e} "
            f"boundary={train_boundary_loss:.3e}/{val_boundary_loss:.3e} "
            f"pde={train_pde_loss:.4e}/{val_pde_loss:.4e} "
            f"cont={train_pde_cont_loss:.3e}/{val_pde_cont_loss:.3e} "
            f"law={train_pde_law_loss:.3e}/{val_pde_law_loss:.3e} "
            f"law_acc={train_pde_law_acc:.2f}/{val_pde_law_acc:.2f} "
            f"eos={train_pde_eos_loss:.3e}/{val_pde_eos_loss:.3e} "
            f"eos_acc={train_pde_eos_acc:.2f}/{val_pde_eos_acc:.2f} "
            f"lr={history['lr'][-1]:.3e} t={epoch_time:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                out_dir / "best_model.pt", model, optim, sched, epoch, cfg,
                history, train_paths, val_paths, model_cfg, best_val,
            )
        save_checkpoint(
            out_dir / "last_model.pt", model, optim, sched, epoch, cfg,
            history, train_paths, val_paths, model_cfg, best_val,
        )
        if cfg.save_every and (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(
                out_dir / f"epoch_{epoch + 1:04d}.pt", model, optim, sched,
                epoch, cfg, history, train_paths, val_paths, model_cfg, best_val,
            )
        save_json(out_dir / "history.json", history)

    best_path = out_dir / "best_model.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location="cpu")
        torch.save(best_ckpt, out_dir / "model.pt")

    metrics = {
        "best_val_loss": best_val,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_train_pred_loss": history["train_pred_loss"][-1],
        "final_val_pred_loss": history["val_pred_loss"][-1],
        "final_train_boundary_loss": history["train_boundary_loss"][-1],
        "final_val_boundary_loss": history["val_boundary_loss"][-1],
        "final_train_pde_loss": history["train_pde_loss"][-1],
        "final_val_pde_loss": history["val_pde_loss"][-1],
        "final_train_pde_cont_loss": history["train_pde_cont_loss"][-1],
        "final_val_pde_cont_loss": history["val_pde_cont_loss"][-1],
        "final_train_pde_law_loss": history["train_pde_law_loss"][-1],
        "final_val_pde_law_loss": history["val_pde_law_loss"][-1],
        "final_train_pde_law_acc": history["train_pde_law_acc"][-1],
        "final_val_pde_law_acc": history["val_pde_law_acc"][-1],
        "final_train_pde_eos_loss": history["train_pde_eos_loss"][-1],
        "final_val_pde_eos_loss": history["val_pde_eos_loss"][-1],
        "final_train_pde_eos_acc": history["train_pde_eos_acc"][-1],
        "final_val_pde_eos_acc": history["val_pde_eos_acc"][-1],
        "n_train_sims": len(train_paths),
        "n_val_sims": len(val_paths),
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "start_offset": cfg.start_offset,
        "t_start": cfg.t_start,
        "model_params": count_params(model),
        "prediction_mode": cfg.prediction_mode,
        "integrator": cfg.integrator,
        "use_derivatives": cfg.use_derivatives,
        "use_boundary_channels": cfg.use_boundary_channels,
        "boundary_aware_refine": cfg.boundary_aware_refine,
        "boundary_loss_weight": cfg.boundary_loss_weight,
        "boundary_channel_raw": bool(cfg.use_boundary_channels),
        "boundary_channel_indices": boundary_channel_indices,
        "use_mask_channel": cfg.use_mask_channel,
        "mask_loss": cfg.mask_loss,
        "masked_channel_stats": bool(train_ds.has_masks),
        "mask_channel_raw": bool(cfg.use_mask_channel),
        "use_physical_derivatives": cfg.use_physical_derivatives,
        "physical_derivative_spacing": model_cfg["physical_derivative_spacing"],
        "attention_type": cfg.attention_type,
        "attention_complexity": attention_complexity,
        "input_noise_std": cfg.input_noise_std,
        "pushforward_prob": cfg.pushforward_prob,
        "pushforward_active": pushforward_active,
        "rollout_train_steps": cfg.rollout_train_steps,
        "rollout_train_weight_decay": cfg.rollout_train_weight_decay,
        "rollout_train_active": False,
        "pde_aux_loss": cfg.pde_aux_loss,
        "pde_aux_weight": cfg.pde_aux_weight,
        "pde_normalize": cfg.pde_normalize if cfg.pde_aux_loss else False,
        "pde_log_transport": cfg.pde_log_transport if cfg.pde_aux_loss else False,
        "pde_log_eps": cfg.pde_log_eps,
        "pde_cont_weight": cfg.pde_cont_weight,
        "pde_law_weight": cfg.pde_law_weight,
        "pde_eos_weight": cfg.pde_eos_weight,
        "pde_cont_loss": cfg.pde_cont_loss,
        "pde_huber_beta": cfg.pde_huber_beta,
        "pde_dim": cfg.pde_dim if cfg.pde_aux_loss else 0,
        "pde_schema": pde_schema,
        "pde_normalizer": pde_normalizer,
        "pde_class_counts": pde_class_counts,
        "strides": parse_strides(cfg.strides),
    }
    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "history.json", history)
    print(f"Saved best checkpoint to {best_path}")
    print(f"Saved last checkpoint to {out_dir / 'last_model.pt'}")
    return {
        "history": history,
        "model": model,
        "train_paths": train_paths,
        "val_paths": val_paths,
        "metrics": metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="grid-adapter output directory")
    ap.add_argument("--out", required=True, help="checkpoint directory")
    ap.add_argument("--context", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--start-offset", type=int, default=0,
                    help="number of saved snapshots to skip before constructing windows")
    ap.add_argument("--t-start", type=float, default=None,
                    help="physical time threshold for the first context frame")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--patch-t", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--attention-type", choices=["global", "factorized"], default="global")
    ap.add_argument("--pos-encoding", choices=["learned_absolute", "sinusoidal"],
                    default="learned_absolute")
    ap.add_argument("--input-noise-std", type=float, default=0.0,
                    help="training-only Gaussian input noise in model-normalizer std units")
    ap.add_argument("--pushforward-prob", type=float, default=0.0,
                    help="training-only probability of replacing the last context state with a detached one-step model prediction")
    ap.add_argument("--rollout-train-steps", type=int, default=1,
                    help="reserved for future multi-step rollout loss; currently only 1 is supported")
    ap.add_argument("--rollout-train-weight-decay", type=float, default=1.0,
                    help="reserved rollout-loss step weighting; currently recorded but inactive while steps=1")
    ap.add_argument("--use-boundary-channels", action="store_true",
                    help="append six static boundary-type one-hot channels as model inputs")
    ap.add_argument("--boundary-aware-refine", action="store_true",
                    help="concatenate the final boundary one-hot mask into the post-patch refine CNN")
    ap.add_argument("--boundary-loss-weight", type=float, default=0.0,
                    help="extra prediction-loss weight over inlet/outlet/obstacle/channel wall pixels")
    ap.add_argument("--use-mask-channel", action="store_true",
                    help="append the static fluid mask as an input channel")
    ap.add_argument("--mask-loss", action="store_true",
                    help="compute prediction loss only over fluid-mask cells")
    ap.add_argument("--pde-aux-loss", action="store_true",
                    help="enable auxiliary pde_vec identification loss")
    ap.add_argument("--pde-aux-weight", type=float, default=0.01)
    ap.add_argument("--pde-normalize", dest="pde_normalize", action="store_true",
                    help="z-score continuous pde_vec targets using train-set statistics")
    ap.add_argument("--no-pde-normalize", dest="pde_normalize", action="store_false",
                    help="disable pde_vec normalization for the auxiliary loss")
    ap.set_defaults(pde_normalize=True)
    ap.add_argument("--pde-log-transport", dest="pde_log_transport", action="store_true",
                    help="log-transform viscosity, bulk viscosity, and thermal conductivity before PDE normalization")
    ap.add_argument("--no-pde-log-transport", dest="pde_log_transport", action="store_false",
                    help="disable log-space normalization for transport coefficients")
    ap.set_defaults(pde_log_transport=True)
    ap.add_argument("--pde-log-eps", type=float, default=1e-12,
                    help="positive floor before log-transforming PDE transport coefficients")
    ap.add_argument("--pde-cont-weight", type=float, default=1.0,
                    help="weight for continuous PDE regression inside the auxiliary loss")
    ap.add_argument("--pde-law-weight", type=float, default=1.0,
                    help="weight for viscosity-law classification inside the auxiliary loss")
    ap.add_argument("--pde-eos-weight", type=float, default=1.0,
                    help="weight for EOS-type classification inside the auxiliary loss")
    ap.add_argument("--pde-cont-loss", choices=["mse", "huber"], default="huber",
                    help="robust loss for normalized continuous PDE regression")
    ap.add_argument("--pde-huber-beta", type=float, default=1.0,
                    help="Huber beta in normalized PDE space when --pde-cont-loss huber")
    ap.add_argument("--pde-dim", type=int, default=0,
                    help="PDE auxiliary output dimension; inferred from pde_vec when 0")
    ap.add_argument("--use-derivatives", dest="use_derivatives", action="store_true")
    ap.add_argument("--no-derivatives", dest="use_derivatives", action="store_false")
    ap.set_defaults(use_derivatives=False)
    ap.add_argument("--derivative-mode", default="central", choices=["central"])
    ap.add_argument("--prediction-mode", default="delta", choices=["delta", "derivative"])
    ap.add_argument("--integrator", default="euler", choices=["euler"])
    ap.add_argument("--strides", default="1")
    args = ap.parse_args()
    if args.horizon != 1:
        ap.error(HORIZON_ERROR)
    if args.start_offset < 0:
        ap.error("--start-offset must be non-negative.")
    if args.input_noise_std < 0.0:
        ap.error("--input-noise-std must be non-negative.")
    if not 0.0 <= args.pushforward_prob <= 1.0:
        ap.error("--pushforward-prob must be between 0 and 1.")
    if args.rollout_train_steps < 1:
        ap.error("--rollout-train-steps must be at least 1.")
    if args.rollout_train_steps != 1:
        ap.error("--rollout-train-steps > 1 is not implemented yet; use --pushforward-prob for this pass.")
    if args.rollout_train_weight_decay < 0.0:
        ap.error("--rollout-train-weight-decay must be non-negative.")
    if args.pde_aux_weight < 0.0:
        ap.error("--pde-aux-weight must be non-negative.")
    if args.pde_cont_weight < 0.0:
        ap.error("--pde-cont-weight must be non-negative.")
    if args.pde_law_weight < 0.0:
        ap.error("--pde-law-weight must be non-negative.")
    if args.pde_eos_weight < 0.0:
        ap.error("--pde-eos-weight must be non-negative.")
    if args.pde_huber_beta <= 0.0:
        ap.error("--pde-huber-beta must be positive.")
    if args.pde_log_eps <= 0.0:
        ap.error("--pde-log-eps must be positive.")
    if args.boundary_loss_weight < 0.0:
        ap.error("--boundary-loss-weight must be non-negative.")
    if args.boundary_aware_refine and not args.use_boundary_channels:
        ap.error("--boundary-aware-refine requires --use-boundary-channels.")

    cfg = TrainConfig(
        grid_dir=args.grid,
        out_dir=args.out,
        context_length=args.context,
        prediction_horizon=args.horizon,
        start_offset=args.start_offset,
        t_start=args.t_start,
        n_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        save_every=args.save_every,
        resume=args.resume,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        patch_size=args.patch,
        patch_t=args.patch_t,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        attention_type=args.attention_type,
        pos_encoding=args.pos_encoding,
        input_noise_std=args.input_noise_std,
        pushforward_prob=args.pushforward_prob,
        rollout_train_steps=args.rollout_train_steps,
        rollout_train_weight_decay=args.rollout_train_weight_decay,
        use_boundary_channels=args.use_boundary_channels,
        boundary_aware_refine=args.boundary_aware_refine,
        boundary_loss_weight=args.boundary_loss_weight,
        use_mask_channel=args.use_mask_channel,
        mask_loss=args.mask_loss,
        pde_aux_loss=args.pde_aux_loss,
        pde_aux_weight=args.pde_aux_weight,
        pde_normalize=args.pde_normalize,
        pde_log_transport=args.pde_log_transport,
        pde_log_eps=args.pde_log_eps,
        pde_cont_weight=args.pde_cont_weight,
        pde_law_weight=args.pde_law_weight,
        pde_eos_weight=args.pde_eos_weight,
        pde_cont_loss=args.pde_cont_loss,
        pde_huber_beta=args.pde_huber_beta,
        pde_dim=args.pde_dim,
        use_derivatives=args.use_derivatives,
        derivative_mode=args.derivative_mode,
        use_physical_derivatives=True,
        prediction_mode=args.prediction_mode,
        integrator=args.integrator,
        strides=args.strides,
    )
    try:
        train(cfg, device=args.device)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
