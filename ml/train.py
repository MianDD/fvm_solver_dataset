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
    pos_encoding: str = "learned_absolute"
    use_mask_channel: bool = False
    mask_loss: bool = False
    pde_aux_loss: bool = False
    pde_aux_weight: float = 0.01
    pde_normalize: bool = True
    pde_log_transport: bool = True
    pde_log_eps: float = 1e-12
    pde_cont_weight: float = 1.0
    pde_law_weight: float = 1.0
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
    return (err * mask).sum() / denom.clamp_min(1.0)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _input_channel_names(use_derivatives: bool) -> List[str]:
    return input_channel_names(use_derivatives, False, TARGET_CHANNEL_NAMES)


def _configured_input_channel_names(cfg: TrainConfig) -> List[str]:
    return input_channel_names(cfg.use_derivatives, cfg.use_mask_channel, TARGET_CHANNEL_NAMES)


def compute_channel_stats(loader: DataLoader, n_channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
    sums = torch.zeros(n_channels)
    sums2 = torch.zeros(n_channels)
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["input_states"]                    # (B, context, C_in, H, W)
            _, _, C, H, W = x.shape
            xx = x.reshape(-1, C, H * W)
            sums += xx.sum(dim=(0, 2))
            sums2 += (xx ** 2).sum(dim=(0, 2))
            n += xx.shape[0] * xx.shape[2]
    mean = sums / max(n, 1)
    var = (sums2 / max(n, 1) - mean ** 2).clamp_min(1e-6)
    return mean, var.sqrt()


def make_model(cfg: TrainConfig, C_in: int, H: int, W: int) -> FoundationCFDModel:
    if cfg.patch_t != 1:
        raise RuntimeError(
            "Only --patch-t 1 is implemented in this incremental pass. "
            "Temporal tubelets with patch_t>1 require a decoder reshape change."
        )
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
        "input_channel_names": _configured_input_channel_names(cfg),
        "target_channel_names": TARGET_CHANNEL_NAMES,
        "use_derivatives": cfg.use_derivatives,
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
        },
        "pde_schema": model_config.get("pde_schema", {}),
        "pde_normalizer": model_config.get("pde_normalizer"),
    }, path)


def predict_next(model: FoundationCFDModel, batch: Dict, cfg: TrainConfig,
                 device: str, training: bool = False) -> Tuple[
                     torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
                 ]:
    features = batch["input_states"].to(device)
    current = batch["context_states"][:, -1].to(device)
    target = batch["target_states"][:, 0].to(device)
    dt = batch["dt"].to(device)
    if training and cfg.input_noise_std > 0.0:
        channel_scale = model.normaliser.std.view(1, 1, -1, 1, 1).to(
            dtype=features.dtype,
            device=features.device,
        )
        if cfg.use_mask_channel:
            channel_scale = channel_scale.clone()
            channel_scale[:, :, -1:] = 0.0
        features = features + cfg.input_noise_std * channel_scale * torch.randn_like(features)
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
    if cfg.pde_aux_weight < 0.0:
        raise ValueError("--pde-aux-weight must be non-negative.")
    if cfg.pde_cont_weight < 0.0:
        raise ValueError("--pde-cont-weight must be non-negative.")
    if cfg.pde_law_weight < 0.0:
        raise ValueError("--pde-law-weight must be non-negative.")
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
        use_mask_channel=cfg.use_mask_channel,
    )
    val_ds = CFDWindowDataset(
        val_paths,
        context_length=cfg.context_length,
        prediction_horizon=cfg.prediction_horizon,
        strides=cfg.strides,
        use_derivatives=cfg.use_derivatives,
        derivative_mode=cfg.derivative_mode,
        use_mask_channel=cfg.use_mask_channel,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("No train/val windows. Reduce --context/--horizon/--strides or add snapshots.")
    print(f"#train windows = {len(train_ds)}   #val windows = {len(val_ds)}")
    if cfg.mask_loss and not (train_ds.has_masks and val_ds.has_masks):
        print("WARNING: mask loss requested but one or more grid files lack masks; missing masks use all-fluid defaults.")
    if cfg.use_mask_channel and not (train_ds.has_masks and val_ds.has_masks):
        print("WARNING: mask channel requested but one or more grid files lack masks; missing masks use all-fluid defaults.")
    pde_schema: Dict = {}
    pde_normalizer: Dict | None = None
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
        if cfg.pde_normalize:
            if resume_ckpt is not None and resume_ckpt.get("pde_normalizer"):
                pde_normalizer = resume_ckpt["pde_normalizer"]
                print("Loaded PDE auxiliary normalizer from resume checkpoint.")
            else:
                pde_vectors = pde_vectors_from_records(train_ds._cache)
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
    print(
        f"PDE aux loss: {'enabled' if cfg.pde_aux_loss else 'disabled'}"
        + (f"  pde_dim={cfg.pde_dim} weight={cfg.pde_aux_weight:g}" if cfg.pde_aux_loss else "")
    )
    if cfg.pde_aux_loss:
        law_status = "enabled" if pde_schema.get("has_viscosity_law") else "not available"
        print(
            f"PDE aux normalization: {'enabled' if cfg.pde_normalize else 'disabled'}  "
            f"pde_dim={cfg.pde_dim}  law_classification={law_status}"
        )
        print(f"  pde_mean: {np.array(pde_normalizer['mean']).round(6).tolist() if pde_normalizer else 'not used'}")
        print(f"  pde_std : {np.array(pde_normalizer['std']).round(6).tolist() if pde_normalizer else 'not used'}")
        if pde_normalizer:
            log_indices = [int(i) for i in pde_normalizer.get("log_indices", [])]
            log_names = [pde_schema.get("pde_vec_names", [])[i] for i in log_indices]
            print(f"  pde_log_indices: {log_indices}  names={log_names}  eps={pde_normalizer.get('log_eps')}")
    print(f"Input noise std: {cfg.input_noise_std:g} normalizer-std units (training only)")
    attention_complexity = attention_complexity_dict(cfg, H, W)
    print_attention_complexity(attention_complexity)

    model = make_model(cfg, C_in, H, W).to(device)
    if cfg.pde_aux_loss and pde_normalizer is not None and model.pde_head is not None and not cfg.resume:
        with torch.no_grad():
            model.pde_head[-1].weight.zero_()
            bias_values = pde_normalizer.get("raw_mean", pde_normalizer["mean"])
            bias = torch.as_tensor(bias_values, dtype=torch.float32, device=device)
            model.pde_head[-1].bias.copy_(bias)
    model_cfg = model_config_dict(cfg, C_in, H, W)
    model_cfg["physical_derivative_spacing"] = "physical" if physical_spacing_used else "index_or_unused"
    model_cfg["attention_complexity"] = attention_complexity
    model_cfg["pde_schema"] = pde_schema
    model_cfg["pde_normalizer"] = pde_normalizer
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
        })
    print(f"Model params: {count_params(model):,}")

    mean, std = compute_channel_stats(train_loader, C_in)
    model.normaliser.mean.copy_(mean.to(device))
    model.normaliser.std.copy_(std.to(device))
    model.normaliser._initialised = True
    normalizer = {
        "channel_names": model_cfg["input_channel_names"],
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    save_json(out_dir / "normalizer.json", normalizer)
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
        "train_pde_loss": [],
        "val_pde_loss": [],
        "train_pde_cont_loss": [],
        "val_pde_cont_loss": [],
        "train_pde_law_loss": [],
        "val_pde_law_loss": [],
        "train_pde_law_acc": [],
        "val_pde_law_acc": [],
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
            "train_pde_law_loss": [],
            "val_pde_law_loss": [],
            "train_pde_law_acc": [],
            "val_pde_law_acc": [],
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
        running_pde = 0.0
        running_pde_cont = 0.0
        running_pde_law = 0.0
        running_pde_law_acc = 0.0
        n_batches = 0
        for batch in train_loader:
            pred, tgt, pred_pde, true_pde = predict_next(model, batch, cfg, device, training=True)
            mask = batch["target_mask"].to(device) if cfg.mask_loss else None
            pred_loss = weighted_masked_mse(pred, tgt, mask=mask)
            if cfg.pde_aux_loss:
                pde_comp = pde_loss_components(
                    pred_pde,
                    true_pde,
                    pde_schema,
                    normalizer=pde_normalizer,
                    normalize=cfg.pde_normalize,
                    cont_weight=cfg.pde_cont_weight,
                    law_weight=cfg.pde_law_weight,
                )
                pde_loss = pde_comp["total"]
                pde_cont_loss = pde_comp["continuous"]
                pde_law_loss = pde_comp["law"]
                pde_law_acc = pde_comp["law_accuracy"]
                loss = pred_loss + cfg.pde_aux_weight * pde_loss
            else:
                pde_loss = torch.zeros((), dtype=pred_loss.dtype, device=pred_loss.device)
                pde_cont_loss = pde_loss
                pde_law_loss = pde_loss
                pde_law_acc = pde_loss
                loss = pred_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += loss.item()
            running_pred += pred_loss.item()
            running_pde += pde_loss.item()
            running_pde_cont += pde_cont_loss.item()
            running_pde_law += pde_law_loss.item()
            running_pde_law_acc += pde_law_acc.item()
            n_batches += 1
        train_loss = running / max(1, n_batches)
        train_pred_loss = running_pred / max(1, n_batches)
        train_pde_loss = running_pde / max(1, n_batches)
        train_pde_cont_loss = running_pde_cont / max(1, n_batches)
        train_pde_law_loss = running_pde_law / max(1, n_batches)
        train_pde_law_acc = running_pde_law_acc / max(1, n_batches)

        model.eval()
        with torch.no_grad():
            val_running = 0.0
            val_pred_running = 0.0
            val_pde_running = 0.0
            val_pde_cont_running = 0.0
            val_pde_law_running = 0.0
            val_pde_law_acc_running = 0.0
            val_batches = 0
            for batch in val_loader:
                pred, tgt, pred_pde, true_pde = predict_next(model, batch, cfg, device)
                mask = batch["target_mask"].to(device) if cfg.mask_loss else None
                pred_loss = weighted_masked_mse(pred, tgt, mask=mask)
                if cfg.pde_aux_loss:
                    pde_comp = pde_loss_components(
                        pred_pde,
                        true_pde,
                        pde_schema,
                        normalizer=pde_normalizer,
                        normalize=cfg.pde_normalize,
                        cont_weight=cfg.pde_cont_weight,
                        law_weight=cfg.pde_law_weight,
                    )
                    pde_loss = pde_comp["total"]
                    pde_cont_loss = pde_comp["continuous"]
                    pde_law_loss = pde_comp["law"]
                    pde_law_acc = pde_comp["law_accuracy"]
                    loss = pred_loss + cfg.pde_aux_weight * pde_loss
                else:
                    pde_loss = torch.zeros((), dtype=pred_loss.dtype, device=pred_loss.device)
                    pde_cont_loss = pde_loss
                    pde_law_loss = pde_loss
                    pde_law_acc = pde_loss
                    loss = pred_loss
                val_running += loss.item()
                val_pred_running += pred_loss.item()
                val_pde_running += pde_loss.item()
                val_pde_cont_running += pde_cont_loss.item()
                val_pde_law_running += pde_law_loss.item()
                val_pde_law_acc_running += pde_law_acc.item()
                val_batches += 1
            val_loss = val_running / max(1, val_batches)
            val_pred_loss = val_pred_running / max(1, val_batches)
            val_pde_loss = val_pde_running / max(1, val_batches)
            val_pde_cont_loss = val_pde_cont_running / max(1, val_batches)
            val_pde_law_loss = val_pde_law_running / max(1, val_batches)
            val_pde_law_acc = val_pde_law_acc_running / max(1, val_batches)

        epoch_time = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_pred_loss"].append(train_pred_loss)
        history["val_pred_loss"].append(val_pred_loss)
        history["train_pde_loss"].append(train_pde_loss)
        history["val_pde_loss"].append(val_pde_loss)
        history["train_pde_cont_loss"].append(train_pde_cont_loss)
        history["val_pde_cont_loss"].append(val_pde_cont_loss)
        history["train_pde_law_loss"].append(train_pde_law_loss)
        history["val_pde_law_loss"].append(val_pde_law_loss)
        history["train_pde_law_acc"].append(train_pde_law_acc)
        history["val_pde_law_acc"].append(val_pde_law_acc)
        history["epoch_time"].append(epoch_time)
        history["lr"].append(float(optim.param_groups[0]["lr"]))
        print(
            f"[ep {epoch:3d}] train={train_loss:.4e} val={val_loss:.4e} "
            f"pred={train_pred_loss:.4e}/{val_pred_loss:.4e} "
            f"pde={train_pde_loss:.4e}/{val_pde_loss:.4e} "
            f"cont={train_pde_cont_loss:.3e}/{val_pde_cont_loss:.3e} "
            f"law={train_pde_law_loss:.3e}/{val_pde_law_loss:.3e} "
            f"law_acc={train_pde_law_acc:.2f}/{val_pde_law_acc:.2f} "
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
        "final_train_pde_loss": history["train_pde_loss"][-1],
        "final_val_pde_loss": history["val_pde_loss"][-1],
        "final_train_pde_cont_loss": history["train_pde_cont_loss"][-1],
        "final_val_pde_cont_loss": history["val_pde_cont_loss"][-1],
        "final_train_pde_law_loss": history["train_pde_law_loss"][-1],
        "final_val_pde_law_loss": history["val_pde_law_loss"][-1],
        "final_train_pde_law_acc": history["train_pde_law_acc"][-1],
        "final_val_pde_law_acc": history["val_pde_law_acc"][-1],
        "n_train_sims": len(train_paths),
        "n_val_sims": len(val_paths),
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "model_params": count_params(model),
        "prediction_mode": cfg.prediction_mode,
        "integrator": cfg.integrator,
        "use_derivatives": cfg.use_derivatives,
        "use_mask_channel": cfg.use_mask_channel,
        "mask_loss": cfg.mask_loss,
        "use_physical_derivatives": cfg.use_physical_derivatives,
        "physical_derivative_spacing": model_cfg["physical_derivative_spacing"],
        "attention_type": cfg.attention_type,
        "attention_complexity": attention_complexity,
        "input_noise_std": cfg.input_noise_std,
        "pde_aux_loss": cfg.pde_aux_loss,
        "pde_aux_weight": cfg.pde_aux_weight,
        "pde_normalize": cfg.pde_normalize if cfg.pde_aux_loss else False,
        "pde_log_transport": cfg.pde_log_transport if cfg.pde_aux_loss else False,
        "pde_log_eps": cfg.pde_log_eps,
        "pde_cont_weight": cfg.pde_cont_weight,
        "pde_law_weight": cfg.pde_law_weight,
        "pde_dim": cfg.pde_dim if cfg.pde_aux_loss else 0,
        "pde_schema": pde_schema,
        "pde_normalizer": pde_normalizer,
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
    if args.input_noise_std < 0.0:
        ap.error("--input-noise-std must be non-negative.")
    if args.pde_aux_weight < 0.0:
        ap.error("--pde-aux-weight must be non-negative.")
    if args.pde_cont_weight < 0.0:
        ap.error("--pde-cont-weight must be non-negative.")
    if args.pde_law_weight < 0.0:
        ap.error("--pde-law-weight must be non-negative.")
    if args.pde_log_eps <= 0.0:
        ap.error("--pde-log-eps must be positive.")

    cfg = TrainConfig(
        grid_dir=args.grid,
        out_dir=args.out,
        context_length=args.context,
        prediction_horizon=args.horizon,
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
        use_mask_channel=args.use_mask_channel,
        mask_loss=args.mask_loss,
        pde_aux_loss=args.pde_aux_loss,
        pde_aux_weight=args.pde_aux_weight,
        pde_normalize=args.pde_normalize,
        pde_log_transport=args.pde_log_transport,
        pde_log_eps=args.pde_log_eps,
        pde_cont_weight=args.pde_cont_weight,
        pde_law_weight=args.pde_law_weight,
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
