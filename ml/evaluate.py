"""Evaluate a trained CFD surrogate on grid-adapter outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import load_model_from_checkpoint
from .dataset import (
    CFDWindowDataset,
    TARGET_CHANNEL_NAMES,
    build_input_features,
    load_grid_record,
    parse_strides,
)


CHANNEL_NAMES = TARGET_CHANNEL_NAMES
HORIZON_ERROR = "Only --horizon 1 is currently supported. Multi-horizon training is not implemented yet."


def grid_paths(grid_dir: str | Path) -> List[str]:
    paths = sorted(str(p) for p in Path(grid_dir).iterdir() if p.suffix == ".npz")
    if not paths:
        raise RuntimeError(f"No .npz grid files found in {grid_dir}")
    return paths


def _init_accumulators(C: int) -> Dict:
    return {
        "mse_sum": torch.zeros(C),
        "mae_sum": torch.zeros(C),
        "sq_true_sum": torch.zeros(C),
        "count": 0,
    }


def _update(acc: Dict, pred: torch.Tensor, true: torch.Tensor) -> None:
    diff = pred - true
    C = diff.shape[-3]
    flat_diff = diff.reshape(-1, C, diff.shape[-2] * diff.shape[-1])
    flat_true = true.reshape(-1, C, true.shape[-2] * true.shape[-1])
    acc["mse_sum"] += (flat_diff ** 2).sum(dim=(0, 2)).detach().cpu()
    acc["mae_sum"] += flat_diff.abs().sum(dim=(0, 2)).detach().cpu()
    acc["sq_true_sum"] += (flat_true ** 2).sum(dim=(0, 2)).detach().cpu()
    acc["count"] += flat_diff.shape[0] * flat_diff.shape[2]


def _finalize(acc: Dict) -> Dict:
    count = max(1, int(acc["count"]))
    mse = acc["mse_sum"] / count
    mae = acc["mae_sum"] / count
    rel_l2 = torch.sqrt(acc["mse_sum"].sum() / acc["sq_true_sum"].sum().clamp_min(1e-12))
    return {
        "mse": float(mse.mean()),
        "mae": float(mae.mean()),
        "relative_l2": float(rel_l2),
        "per_channel": {
            name: {"mse": float(mse[i]), "mae": float(mae[i])}
            for i, name in enumerate(CHANNEL_NAMES[:len(mse)])
        },
        "count": count,
    }


def _cfg_value(ckpt: Dict, key: str, default):
    cfg = ckpt.get("config", {})
    model_cfg = ckpt.get("model_config", {})
    return cfg.get(key, model_cfg.get(key, default))


def _prediction_settings(ckpt: Dict, overrides: argparse.Namespace) -> Dict:
    ckpt_strides = _cfg_value(ckpt, "strides", [1])
    if isinstance(ckpt_strides, str):
        stride_text = ckpt_strides
    else:
        stride_text = ",".join(str(s) for s in ckpt_strides)
    return {
        "use_derivatives": (
            overrides.use_derivatives
            if overrides.use_derivatives is not None
            else bool(_cfg_value(ckpt, "use_derivatives", False))
        ),
        "derivative_mode": overrides.derivative_mode or _cfg_value(ckpt, "derivative_mode", "central"),
        "use_physical_derivatives": bool(_cfg_value(ckpt, "use_physical_derivatives", True)),
        "prediction_mode": overrides.prediction_mode or _cfg_value(ckpt, "prediction_mode", "delta"),
        "integrator": overrides.integrator or _cfg_value(ckpt, "integrator", "euler"),
        "strides": overrides.strides or stride_text,
    }


def _predict_next(model, batch: Dict, settings: Dict, device: str):
    features = batch["input_states"].to(device)
    current = batch["context_states"][:, -1].to(device)
    target = batch["target_states"][:, 0].to(device)
    dt = batch["dt"].to(device)
    update = model.predict_update(features, normalised=False)[:, -1]
    pred = model.integrate_update(
        current,
        update,
        prediction_mode=settings["prediction_mode"],
        integrator=settings["integrator"],
        dt=dt,
    )
    return pred, target


def _rollout_one_file(model, path: str | Path, context: int, steps: int,
                      temporal_stride: int, settings: Dict, device: str):
    rec = load_grid_record(path)
    states = rec["states"]
    times = rec["times"]
    required_last = (context + steps - 1) * temporal_stride
    if states.shape[0] <= required_last:
        return None

    context_idx = np.arange(context, dtype=np.int64) * temporal_stride
    context_states = states[context_idx].copy()
    context_times = times[context_idx].copy()
    preds = []
    true = []
    for step in range(steps):
        target_idx = (context + step) * temporal_stride
        target_time = float(times[target_idx])
        dt = target_time - float(context_times[-1])
        if not np.isfinite(dt) or dt <= 0:
            print(f"WARNING: {Path(path).name} has missing/non-monotonic times; using dt=1.0")
            dt = 1.0
        features = build_input_features(
            context_states[-context:],
            times=context_times[-context:],
            dx_spacing=rec["dx"],
            dy_spacing=rec["dy"],
            use_derivatives=settings["use_derivatives"],
            derivative_mode=settings["derivative_mode"],
        )
        features_t = torch.from_numpy(features).unsqueeze(0).to(device)
        current = torch.from_numpy(context_states[-1]).unsqueeze(0).to(device)
        with torch.no_grad():
            update = model.predict_update(features_t, normalised=False)[:, -1]
            pred = model.integrate_update(
                current,
                update,
                prediction_mode=settings["prediction_mode"],
                integrator=settings["integrator"],
                dt=torch.tensor([dt], dtype=torch.float32, device=device),
            )
        pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
        preds.append(pred_np)
        true.append(states[target_idx])
        context_states = np.concatenate([context_states, pred_np[None, ...]], axis=0)
        context_times = np.concatenate([context_times, np.array([target_time], dtype=np.float32)])
    return torch.from_numpy(np.stack(preds)), torch.from_numpy(np.stack(true))


def evaluate(grid_dir: str | Path, ckpt_path: str | Path, out_dir: str | Path,
             context: int | None = None, horizon: int | None = None,
             batch_size: int = 4, device: str = "cpu", num_workers: int = 0,
             use_derivatives: bool | None = None, derivative_mode: str | None = None,
             prediction_mode: str | None = None, integrator: str | None = None,
             strides: str | None = None, rollout_steps: str = "4,8,16") -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_model_from_checkpoint(ckpt_path, device=device)
    cfg = ckpt.get("config", {})
    context = int(context or cfg.get("context_length", 4))
    horizon = int(horizon or cfg.get("prediction_horizon", 1))
    if horizon != 1:
        raise ValueError(HORIZON_ERROR)
    overrides = argparse.Namespace(
        use_derivatives=use_derivatives,
        derivative_mode=derivative_mode,
        prediction_mode=prediction_mode,
        integrator=integrator,
        strides=strides,
    )
    settings = _prediction_settings(ckpt, overrides)
    paths = grid_paths(grid_dir)
    ds = CFDWindowDataset(
        paths,
        context_length=context,
        prediction_horizon=horizon,
        strides=settings["strides"],
        use_derivatives=settings["use_derivatives"],
        derivative_mode=settings["derivative_mode"],
    )
    if len(ds) == 0:
        raise RuntimeError("No evaluation windows. Reduce --context/--horizon/--strides or add snapshots.")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    C = model.C
    overall = _init_accumulators(C)
    by_stride: Dict[str, Dict] = {}
    by_family: Dict[str, Dict] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred, target = _predict_next(model, batch, settings, device)
            _update(overall, pred, target)
            strides_b = batch["stride"].detach().cpu().tolist()
            families = batch["family"]
            for i, stride_i in enumerate(strides_b):
                key = str(int(stride_i))
                by_stride.setdefault(key, _init_accumulators(C))
                _update(by_stride[key], pred[i:i + 1], target[i:i + 1])
            for i, family in enumerate(families):
                key = str(family)
                by_family.setdefault(key, _init_accumulators(C))
                _update(by_family[key], pred[i:i + 1], target[i:i + 1])

    rollout_metrics: Dict[str, Dict] = {}
    rollout_targets = parse_strides(rollout_steps) if rollout_steps else []
    eval_strides = parse_strides(settings["strides"])
    with torch.no_grad():
        for steps in rollout_targets:
            for temporal_stride in eval_strides:
                acc = _init_accumulators(C)
                used = 0
                for path in paths:
                    rolled = _rollout_one_file(
                        model, path, context, steps, temporal_stride,
                        settings, device,
                    )
                    if rolled is None:
                        continue
                    pred_r, true_r = rolled
                    _update(acc, pred_r, true_r)
                    used += 1
                if used:
                    rollout_metrics[f"steps_{steps}_stride_{temporal_stride}"] = {
                        **_finalize(acc),
                        "n_files": used,
                    }

    metrics = {
        "grid_dir": str(Path(grid_dir).resolve()),
        "checkpoint": str(Path(ckpt_path).resolve()),
        "n_grid_files": len(paths),
        "n_windows": len(ds),
        "context_length": context,
        "prediction_horizon": horizon,
        "settings": settings,
        "one_step": _finalize(overall),
        "by_stride": {k: _finalize(v) for k, v in sorted(by_stride.items())},
        "by_family": {k: _finalize(v) for k, v in sorted(by_family.items())},
        "rollout": rollout_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = CHANNEL_NAMES[:C]
        mse = [metrics["one_step"]["per_channel"][name]["mse"] for name in names]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(names, mse)
        ax.set_ylabel("one-step MSE")
        ax.set_title("Per-channel one-step error")
        fig.tight_layout()
        fig.savefig(out_dir / "one_step_channel_mse.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        metrics["plot_warning"] = str(exc)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--use-derivatives", dest="use_derivatives", action="store_true")
    ap.add_argument("--no-derivatives", dest="use_derivatives", action="store_false")
    ap.set_defaults(use_derivatives=None)
    ap.add_argument("--derivative-mode", default=None, choices=["central"])
    ap.add_argument("--prediction-mode", default=None, choices=["delta", "derivative"])
    ap.add_argument("--integrator", default=None, choices=["euler"])
    ap.add_argument("--strides", default=None)
    ap.add_argument("--rollout-steps", default="4,8,16")
    args = ap.parse_args()
    if args.horizon is not None and args.horizon != 1:
        ap.error(HORIZON_ERROR)
    try:
        metrics = evaluate(
            args.grid, args.ckpt, args.out, context=args.context,
            horizon=args.horizon, batch_size=args.batch,
            device=args.device, num_workers=args.num_workers,
            use_derivatives=args.use_derivatives,
            derivative_mode=args.derivative_mode,
            prediction_mode=args.prediction_mode,
            integrator=args.integrator,
            strides=args.strides,
            rollout_steps=args.rollout_steps,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
