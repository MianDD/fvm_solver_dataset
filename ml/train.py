"""Training script for the foundation CFD model."""

from __future__ import annotations

import argparse
import dataclasses as dc
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import CFDWindowDataset, split_paths
from .model import FoundationCFDModel, count_params


# --------------------------------------------------------------------------
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
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4


def weighted_mse(pred: torch.Tensor, true: torch.Tensor,
                 weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)) -> torch.Tensor:
    w = torch.as_tensor(weights, dtype=pred.dtype, device=pred.device)
    w = w.view(*([1] * (pred.dim() - 3)), -1, 1, 1)
    return (w * (pred - true) ** 2).mean()


# --------------------------------------------------------------------------
def train(cfg: TrainConfig, device: str | None = None) -> Dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    train_paths, val_paths = split_paths(cfg.grid_dir, val_frac=cfg.val_frac,
                                         seed=cfg.seed)
    print(f"#train sims = {len(train_paths)}   #val sims = {len(val_paths)}")
    if not train_paths or not val_paths:
        raise RuntimeError("Need at least 2 sims (1 train, 1 val).")

    train_ds = CFDWindowDataset(train_paths,
                                context_length=cfg.context_length,
                                prediction_horizon=cfg.prediction_horizon)
    val_ds = CFDWindowDataset(val_paths,
                              context_length=cfg.context_length,
                              prediction_horizon=cfg.prediction_horizon)
    print(f"#train windows = {len(train_ds)}   #val windows = {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0)

    sample = train_ds[0]
    _, C, H, W = sample["states"].shape
    print(f"State shape: C={C} H={H} W={W}")

    model = FoundationCFDModel(
        n_channels=C, H=H, W=W, patch_size=cfg.patch_size,
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
        max_context=cfg.context_length + cfg.prediction_horizon,
    ).to(device)
    print(f"Model params: {count_params(model):,}")

    # Warm-start the channel normaliser from training data
    print("Initialising channel normaliser…")
    sums = torch.zeros(C); sums2 = torch.zeros(C); n = 0
    with torch.no_grad():
        for batch in train_loader:
            x = batch["states"]                           # (B, tau+1, C, H, W)
            xx = x.reshape(-1, C, H * W)
            sums  += xx.sum(dim=(0, 2))
            sums2 += (xx ** 2).sum(dim=(0, 2))
            n += xx.shape[0] * xx.shape[2]
        mean = sums / max(n, 1)
        var = (sums2 / max(n, 1) - mean ** 2).clamp_min(1e-6)
        std = var.sqrt()
        model.normaliser.mean.copy_(mean.to(device))
        model.normaliser.std.copy_(std.to(device))
        model.normaliser._initialised = True
    print(f"  channel means: {model.normaliser.mean.cpu().numpy().round(3)}")
    print(f"  channel stds : {model.normaliser.std.cpu().numpy().round(3)}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cfg.n_epochs * max(1, len(train_loader)))

    history: Dict[str, List] = {"train_loss": [], "val_loss": [], "epoch_time": []}

    for epoch in range(cfg.n_epochs):
        t0 = time.time()
        model.train()
        running = 0.0; n_batches = 0
        for batch in train_loader:
            x = batch["states"].to(device)               # (B, tau+1, C, H, W)
            inp = x[:, :-1]; tgt = x[:, 1:]
            pred = model(inp, normalised=False)
            loss = weighted_mse(pred, tgt)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += loss.item(); n_batches += 1
        train_loss = running / max(1, n_batches)

        model.eval()
        with torch.no_grad():
            v = 0.0; nb = 0
            for batch in val_loader:
                x = batch["states"].to(device)
                inp = x[:, :-1]; tgt = x[:, 1:]
                pred = model(inp, normalised=False)
                v += weighted_mse(pred, tgt).item(); nb += 1
            val_loss = v / max(1, nb)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time"].append(time.time() - t0)
        print(f"[ep {epoch:3d}] train={train_loss:.4e}  val={val_loss:.4e}  "
              f"t={history['epoch_time'][-1]:.1f}s")

    ckpt = {
        "model": model.state_dict(),
        "config": dc.asdict(cfg),
        "history": history,
        "train_paths": train_paths, "val_paths": val_paths,
    }
    torch.save(ckpt, out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved {out_dir / 'model.pt'}")

    return {"history": history, "model": model,
            "train_paths": train_paths, "val_paths": val_paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="grid-adapter output directory")
    ap.add_argument("--out", required=True, help="checkpoint directory")
    ap.add_argument("--context", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = TrainConfig(
        grid_dir=args.grid, out_dir=args.out,
        context_length=args.context, n_epochs=args.epochs,
        batch_size=args.batch, lr=args.lr,
        d_model=args.d_model, patch_size=args.patch,
    )
    train(cfg, device=args.device)


if __name__ == "__main__":
    main()
