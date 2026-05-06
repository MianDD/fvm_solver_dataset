"""Smoke test for later-stage window filtering with start_offset/t_start."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .dataset import CFDWindowDataset
from .evaluate import evaluate
from .pde import DEFAULT_PDE_VEC_NAMES
from .plot_predictions import plot_predictions
from .train import TrainConfig, train


def _pde_vec(sim_idx: int) -> np.ndarray:
    base = np.array([
        1.35 + 0.01 * sim_idx,
        1.0e-3,
        0.0,
        1.35 * 2.5 * 1.0e-3 / 0.71,
        2.5,
        100.0,
        1.0,
        100.0,
        1.0,
    ], dtype=np.float32)
    law = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    power_law_n = np.array([0.75], dtype=np.float32)
    eos = np.array([1.0, 0.0], dtype=np.float32)
    p_inf = np.array([0.0, 0.0], dtype=np.float32)
    return np.concatenate([base, law, power_law_n, eos, p_inf]).astype(np.float32)


def _write_grid(path: Path, sim_idx: int, with_times: bool = True) -> None:
    T, C, H, W = 8, 4, 16, 24
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, H, dtype=np.float32),
        np.linspace(0.0, 1.0, W, dtype=np.float32),
        indexing="ij",
    )
    states = np.zeros((T, C, H, W), dtype=np.float32)
    for t in range(T):
        states[t, 0] = xx + 0.05 * t + 0.01 * sim_idx
        states[t, 1] = yy - 0.02 * t
        states[t, 2] = 1.0 + 0.01 * t + 0.02 * xx
        states[t, 3] = 100.0 + 0.1 * t + 0.5 * yy
    mask = np.ones((H, W), dtype=np.float32)
    mask[5:8, 10:14] = 0.0
    states[:, 0, mask < 0.5] = 0.0
    states[:, 1, mask < 0.5] = 0.0
    states[:, 2, mask < 0.5] = 1.0
    states[:, 3, mask < 0.5] = 100.0
    metadata = {
        "sim_id": f"sim_{sim_idx:04d}",
        "family": "synthetic_start_filter",
        "status": "success",
        "number_of_snapshots": T,
        "channel_names": ["V_x", "V_y", "rho", "T"],
        "mask_semantics": {"0": "solid_or_invalid", "1": "fluid"},
    }
    payload = {
        "states": states,
        "snapshots": states,
        "channel_names": np.array(["V_x", "V_y", "rho", "T"]),
        "mask": mask,
        "dx": np.float32(1.0 / (W - 1)),
        "dy": np.float32(1.0 / (H - 1)),
        "metadata_json": json.dumps(metadata),
        "cfg_json": json.dumps({"sim_id": metadata["sim_id"], "family": metadata["family"]}),
        "pde_vec": _pde_vec(sim_idx),
        "pde_vec_names": np.array(DEFAULT_PDE_VEC_NAMES),
    }
    if with_times:
        payload["times"] = np.arange(T, dtype=np.float32) * 0.3
    np.savez_compressed(path, **payload)


def _check_window_counts(grid: Path) -> None:
    paths = sorted(grid.glob("*.npz"))
    kwargs = {"context_length": 3, "prediction_horizon": 1, "strides": 1}
    base = CFDWindowDataset(paths[:1], **kwargs)
    offset = CFDWindowDataset(paths[:1], start_offset=3, **kwargs)
    time_filtered = CFDWindowDataset(paths[:1], t_start=0.9, **kwargs)
    stricter_time = CFDWindowDataset(paths[:1], start_offset=2, t_start=1.2, **kwargs)
    stricter_offset = CFDWindowDataset(paths[:1], start_offset=4, t_start=0.6, **kwargs)
    if len(base) != 5:
        raise RuntimeError(f"start_offset=0 should preserve 5 windows, got {len(base)}")
    if len(offset) != 2:
        raise RuntimeError(f"start_offset=3 should leave 2 windows, got {len(offset)}")
    if len(time_filtered) != 2:
        raise RuntimeError(f"t_start=0.9 should leave 2 windows, got {len(time_filtered)}")
    if len(stricter_time) != 1:
        raise RuntimeError(f"start_offset=2,t_start=1.2 should leave 1 window, got {len(stricter_time)}")
    if len(stricter_offset) != 1:
        raise RuntimeError(f"start_offset=4,t_start=0.6 should leave 1 window, got {len(stricter_offset)}")
    if base.windows[0][1] != 0 or offset.windows[0][1] != 3:
        raise RuntimeError("window start indices do not match requested offsets")


def _check_missing_times_fallback(grid_missing_times: Path) -> None:
    ds = CFDWindowDataset(
        sorted(grid_missing_times.glob("*.npz")),
        context_length=3,
        prediction_horizon=1,
        start_offset=2,
        t_start=0.9,
    )
    if len(ds) != 3:
        raise RuntimeError(f"missing-times fallback should use start_offset=2, got {len(ds)} windows")
    if ds.windows[0][1] != 2:
        raise RuntimeError("missing-times fallback did not start at start_offset")


def _check_train_eval_plot(grid: Path, root: Path) -> None:
    train_out = root / "checkpoints"
    result = train(
        TrainConfig(
            grid_dir=str(grid),
            out_dir=str(train_out),
            context_length=3,
            prediction_horizon=1,
            start_offset=2,
            t_start=0.6,
            n_epochs=1,
            batch_size=1,
            val_frac=0.5,
            seed=7,
            patch_size=8,
            d_model=16,
            n_heads=4,
            n_layers=1,
            attention_type="factorized",
            pos_encoding="sinusoidal",
            use_derivatives=True,
            use_mask_channel=True,
            mask_loss=True,
        ),
        device="cpu",
    )
    metrics = result["metrics"]
    if not np.isfinite(metrics["final_train_loss"]):
        raise RuntimeError("start-filter train loss is not finite")
    if metrics["start_offset"] != 2 or metrics["t_start"] != 0.6:
        raise RuntimeError("train metrics did not record start filter settings")
    if not (train_out / "best_model.pt").exists():
        raise RuntimeError("training did not write best_model.pt")

    eval_metrics = evaluate(
        grid,
        train_out / "best_model.pt",
        root / "eval",
        context=3,
        horizon=1,
        batch_size=1,
        device="cpu",
        rollout_steps="",
        start_offset=2,
        t_start=0.6,
    )
    if eval_metrics["start_offset"] != 2 or eval_metrics["t_start"] != 0.6:
        raise RuntimeError("evaluation metrics did not record start filter settings")
    if eval_metrics["n_windows"] <= 0:
        raise RuntimeError("evaluation should have at least one filtered window")

    fig_dir = root / "figures"
    plot_predictions(
        grid,
        train_out / "best_model.pt",
        fig_dir,
        context=3,
        horizon=1,
        num_examples=1,
        stride=1,
        device="cpu",
        start_offset=2,
        t_start=0.6,
    )
    metadata = json.loads((fig_dir / "plot_metadata.json").read_text(encoding="utf-8"))
    if metadata["start_offset"] != 2 or metadata["t_start"] != 0.6:
        raise RuntimeError("plot metadata did not record start filter settings")
    if metadata["examples"][0]["start_index"] < 2:
        raise RuntimeError("plot example ignored start filtering")


def main() -> None:
    root = Path("datasets") / "_codex_smoke_start_filter"
    shutil.rmtree(root, ignore_errors=True)
    grid = root / "grid"
    missing_times = root / "grid_missing_times"
    try:
        grid.mkdir(parents=True)
        missing_times.mkdir(parents=True)
        for sim_idx in range(2):
            _write_grid(grid / f"sim_{sim_idx:04d}.npz", sim_idx, with_times=True)
        _write_grid(missing_times / "sim_0000.npz", 0, with_times=False)

        _check_window_counts(grid)
        _check_missing_times_fallback(missing_times)
        _check_train_eval_plot(grid, root)

        print("OK start filter smoke: start_offset/t_start window filtering, train, evaluate, and plot")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
