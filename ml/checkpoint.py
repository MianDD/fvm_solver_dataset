"""Checkpoint loading helpers shared by evaluation and plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from .model import FoundationCFDModel


MODEL_KWARGS = {
    "n_channels", "n_input_channels", "n_target_channels", "H", "W",
    "patch_size", "d_model", "n_heads", "n_layers", "max_context",
    "dropout", "mlp_ratio",
}


def load_model_from_checkpoint(path: str | Path, device: str = "cpu") -> Tuple[FoundationCFDModel, dict]:
    ckpt = torch.load(path, map_location=device)
    if "model_config" in ckpt:
        model_cfg = dict(ckpt["model_config"])
    else:
        cfg = ckpt.get("config", {})
        model_cfg = {
            "n_channels": 4,
            "H": cfg.get("H", 64),
            "W": cfg.get("W", 96),
            "patch_size": cfg.get("patch_size", 8),
            "d_model": cfg.get("d_model", 128),
            "n_heads": cfg.get("n_heads", 4),
            "n_layers": cfg.get("n_layers", 4),
            "max_context": cfg.get("context_length", 4) + cfg.get("prediction_horizon", 1),
            "dropout": cfg.get("dropout", 0.0),
            "mlp_ratio": cfg.get("mlp_ratio", 4.0),
        }
    model_kwargs = {k: v for k, v in model_cfg.items() if k in MODEL_KWARGS}
    model = FoundationCFDModel(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
