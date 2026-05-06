"""Smoke tests for boundary-weighted loss and boundary-aware refinement."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .boundary import BOUNDARY_CLASS_NAMES
from .model import FoundationCFDModel
from .smoke_boundary_mask import _boundary_for, _mask, _write_grid
from .train import TrainConfig, boundary_fluid_mask, train, weighted_masked_mse


def _check_model(name: str, C_in: int, *, use_boundary: bool,
                 boundary_aware: bool, boundary_start: int | None = None) -> int:
    torch.manual_seed(7)
    B, T, H, W = 2, 2, 16, 24
    model = FoundationCFDModel(
        n_channels=4,
        n_input_channels=C_in,
        n_target_channels=4,
        H=H,
        W=W,
        patch_size=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        attention_type="factorized",
        pos_encoding="sinusoidal",
        use_boundary_channels=use_boundary,
        boundary_channels=len(BOUNDARY_CLASS_NAMES),
        boundary_aware_refine=boundary_aware,
        boundary_channel_start=boundary_start,
    )
    x = torch.randn(B, T, C_in, H, W)
    if use_boundary:
        if boundary_start is None:
            raise RuntimeError("boundary model smoke requires boundary_start")
        x[:, :, boundary_start:boundary_start + 6] = 0.0
        x[:, :, boundary_start] = 1.0
    out = model.predict_update(x, normalised=False)
    if tuple(out.shape) != (B, 1, 4, H, W):
        raise RuntimeError(f"{name} output shape changed: {tuple(out.shape)}")
    loss = out.square().mean()
    loss.backward()
    grads = [
        p.grad.detach()
        for p in model.parameters()
        if p.grad is not None
    ]
    if not grads or not all(torch.all(torch.isfinite(g)) for g in grads):
        raise RuntimeError(f"{name} gradients are missing or non-finite")
    return sum(p.numel() for p in model.parameters())


def _check_boundary_loss_mask() -> None:
    mask = _mask()
    boundary = _boundary_for(mask)
    batch = {"boundary_mask": torch.from_numpy(boundary[None])}
    region = boundary_fluid_mask(batch, "cpu")
    if region is None or float(region.sum()) <= 0.0:
        raise RuntimeError("boundary loss mask should include wall/inlet/outlet fluid pixels")
    if bool(torch.any(region.bool() & torch.from_numpy(mask <= 0.5))):
        raise RuntimeError("boundary loss mask must not include solid pixels")
    pred = torch.ones(1, 4, *mask.shape)
    true = torch.zeros_like(pred)
    loss = weighted_masked_mse(pred, true, mask=region)
    if not torch.isfinite(loss) or float(loss) <= 0.0:
        raise RuntimeError(f"boundary loss should be finite and positive, got {loss}")


def main() -> None:
    root = Path("datasets") / "_codex_smoke_boundary_refine"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    grid = root / "grid"
    try:
        _check_boundary_loss_mask()
        params = {
            "standard": _check_model("standard", 4, use_boundary=False, boundary_aware=False),
            "boundary_channels": _check_model(
                "boundary_channels", 23, use_boundary=True,
                boundary_aware=False, boundary_start=16,
            ),
            "boundary_refine": _check_model(
                "boundary_refine", 23, use_boundary=True,
                boundary_aware=True, boundary_start=16,
            ),
        }

        grid.mkdir(parents=True, exist_ok=True)
        mask = _mask()
        boundary = _boundary_for(mask)
        for sim_idx in range(2):
            _write_grid(grid / f"sim_{sim_idx:04d}.npz", sim_idx, mask, boundary)

        result = train(
            TrainConfig(
                grid_dir=str(grid),
                out_dir=str(root / "train"),
                context_length=2,
                prediction_horizon=1,
                n_epochs=1,
                batch_size=2,
                val_frac=0.5,
                seed=31,
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
                boundary_loss_weight=0.5,
                boundary_aware_refine=True,
            ),
            device="cpu",
        )
        metrics = result["metrics"]
        if not np.isfinite(metrics["final_train_loss"]):
            raise RuntimeError("boundary-refine training loss is not finite")
        if metrics["final_train_boundary_loss"] <= 0.0:
            raise RuntimeError("boundary-weighted loss should be positive in the smoke dataset")
        model_cfg = json.loads((root / "train" / "model_config.json").read_text(encoding="utf-8"))
        if not model_cfg.get("boundary_aware_refine"):
            raise RuntimeError("model_config should record boundary_aware_refine=true")

        print(
            "OK boundary refine smoke: "
            f"params={params} boundary_loss={metrics['final_train_boundary_loss']:.4e} "
            f"train_loss={metrics['final_train_loss']:.4e}"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
