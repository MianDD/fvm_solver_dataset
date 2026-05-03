"""PyTorch datasets for regular-grid CFD trajectories.

The grid files are produced by :mod:`ml.grid_adapter`.  New files contain
``states`` plus metadata; older files contain ``snapshots`` only.  Both are
supported here so existing smoke-test datasets still train.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


TARGET_CHANNEL_NAMES = ["V_x", "V_y", "rho", "T"]


def parse_strides(value: str | Sequence[int] | int) -> List[int]:
    """Parse CLI stride values such as ``"1,2,4"``."""
    if isinstance(value, int):
        strides = [value]
    elif isinstance(value, str):
        strides = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        strides = [int(v) for v in value]
    strides = sorted(set(strides))
    if not strides or any(s <= 0 for s in strides):
        raise ValueError("strides must be positive integers")
    return strides


def derivative_channel_names(base_names: Sequence[str] = TARGET_CHANNEL_NAMES) -> List[str]:
    names = list(base_names)
    names += [f"d{name}_dx" for name in base_names]
    names += [f"d{name}_dy" for name in base_names]
    names += [f"d{name}_dt" for name in base_names]
    return names


def _json_from_npz_value(value) -> Dict:
    if value is None:
        return {}
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def load_grid_record(path: str | Path) -> Dict:
    """Load one packed grid trajectory with backward-compatible keys."""
    path = Path(path)
    z = np.load(path, allow_pickle=True)
    if "states" in z:
        states = z["states"].astype(np.float32)
    elif "snapshots" in z:
        states = z["snapshots"].astype(np.float32)
    else:
        raise KeyError(f"{path} has neither 'states' nor 'snapshots'")

    if "times" in z:
        times = z["times"].astype(np.float32)
    else:
        print(f"WARNING: {path.name} is missing times; using unit-spaced indices.")
        times = np.arange(states.shape[0], dtype=np.float32)
    if "channel_names" in z:
        channel_names = [str(x) for x in z["channel_names"].tolist()]
    else:
        channel_names = list(TARGET_CHANNEL_NAMES)
    metadata = _json_from_npz_value(z["metadata_json"] if "metadata_json" in z else None)
    cfg = _json_from_npz_value(z["cfg_json"] if "cfg_json" in z else None)
    if not metadata:
        metadata = {
            "sim_id": cfg.get("sim_id", path.stem),
            "seed": cfg.get("seed"),
            "family": cfg.get("family", "unknown"),
            "status": "success",
            "number_of_snapshots": int(states.shape[0]),
            "channel_names": channel_names,
        }
    x_coords = z["x_coords"].astype(np.float32) if "x_coords" in z else None
    y_coords = z["y_coords"].astype(np.float32) if "y_coords" in z else None
    dx = float(z["dx"]) if "dx" in z else None
    dy = float(z["dy"]) if "dy" in z else None
    physical_spacing = bool(dx is not None and dy is not None)
    return {
        "path": str(path),
        "states": states,
        "times": times,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "dx": dx,
        "dy": dy,
        "physical_spacing": physical_spacing,
        "channel_names": channel_names,
        "metadata": metadata,
        "pde_vec": z["pde_vec"].astype(np.float32) if "pde_vec" in z else np.zeros(9, dtype=np.float32),
    }


def _safe_dt(a: float, b: float, fallback: float = 1.0) -> float:
    dt = float(b - a)
    if not np.isfinite(dt) or dt <= 0:
        return float(fallback)
    return dt


def compute_derivative_features(frames: np.ndarray, times: np.ndarray | None = None,
                                dx_spacing: float | None = None,
                                dy_spacing: float | None = None,
                                mode: str = "central") -> np.ndarray:
    """Concatenate physical channels with dx, dy, and dt features.

    Parameters
    ----------
    frames:
        Array shaped ``(T, C, H, W)``.
    times:
        Optional time stamps for the T frames.  Missing or non-monotonic time
        stamps fall back to unit spacing for the temporal derivative.
    mode:
        Currently ``central``.  Boundaries use one-sided differences.
    """
    if mode != "central":
        raise ValueError(f"Unsupported derivative mode: {mode}")
    x = np.asarray(frames, dtype=np.float32)
    dx = np.zeros_like(x)
    dy = np.zeros_like(x)
    dt = np.zeros_like(x)
    sx = float(dx_spacing) if dx_spacing is not None and dx_spacing > 0 else 1.0
    sy = float(dy_spacing) if dy_spacing is not None and dy_spacing > 0 else 1.0

    if x.shape[-1] > 1:
        dx[..., 1:-1] = 0.5 * (x[..., 2:] - x[..., :-2]) / sx
        dx[..., 0] = (x[..., 1] - x[..., 0]) / sx
        dx[..., -1] = (x[..., -1] - x[..., -2]) / sx
    if x.shape[-2] > 1:
        dy[..., 1:-1, :] = 0.5 * (x[..., 2:, :] - x[..., :-2, :]) / sy
        dy[..., 0, :] = (x[..., 1, :] - x[..., 0, :]) / sy
        dy[..., -1, :] = (x[..., -1, :] - x[..., -2, :]) / sy

    if x.shape[0] > 1:
        if times is None or len(times) != x.shape[0]:
            times_arr = np.arange(x.shape[0], dtype=np.float32)
        else:
            times_arr = np.asarray(times, dtype=np.float32)
        for i in range(x.shape[0]):
            if i == 0:
                denom = _safe_dt(times_arr[0], times_arr[1])
                dt[i] = (x[1] - x[0]) / denom
            elif i == x.shape[0] - 1:
                denom = _safe_dt(times_arr[-2], times_arr[-1])
                dt[i] = (x[-1] - x[-2]) / denom
            else:
                denom = _safe_dt(times_arr[i - 1], times_arr[i + 1], fallback=2.0)
                dt[i] = (x[i + 1] - x[i - 1]) / denom

    return np.concatenate([x, dx, dy, dt], axis=1).astype(np.float32)


def build_input_features(frames: np.ndarray, times: np.ndarray | None = None,
                         dx_spacing: float | None = None,
                         dy_spacing: float | None = None,
                         use_derivatives: bool = False,
                         derivative_mode: str = "central") -> np.ndarray:
    if not use_derivatives:
        return np.asarray(frames, dtype=np.float32)
    if dx_spacing is None or dy_spacing is None:
        print("WARNING: grid spacing missing; derivative features use index spacing.")
    return compute_derivative_features(
        frames,
        times=times,
        dx_spacing=dx_spacing,
        dy_spacing=dy_spacing,
        mode=derivative_mode,
    )


class CFDWindowDataset(Dataset):
    """One item is a temporal context and its future target.

    ``strides`` controls temporal subsampling inside each trajectory:
    context indices are ``t, t+s, t+2s, ...`` and the first target is
    ``t+context*s``.  Targets remain the physical channels
    ``[V_x, V_y, rho, T]`` even when derivative features are enabled.
    """

    def __init__(self, paths: List[str | Path],
                 context_length: int = 4, prediction_horizon: int = 1,
                 strides: str | Sequence[int] | int = 1,
                 stride: int | None = None,
                 use_derivatives: bool = False,
                 derivative_mode: str = "central"):
        self.paths = [Path(p) for p in paths]
        self.context_length = int(context_length)
        self.prediction_horizon = int(prediction_horizon)
        self.strides = parse_strides(stride if stride is not None else strides)
        self.use_derivatives = bool(use_derivatives)
        self.derivative_mode = derivative_mode

        self._cache: List[Dict] = []
        self.windows: List[Tuple[int, int, int]] = []  # (sim_idx, start, temporal_stride)
        span = self.context_length + self.prediction_horizon - 1
        for sim_idx, p in enumerate(self.paths):
            rec = load_grid_record(p)
            self._cache.append(rec)
            T = rec["states"].shape[0]
            for temporal_stride in self.strides:
                last_offset = span * temporal_stride
                if T <= last_offset:
                    continue
                for start in range(0, T - last_offset):
                    self.windows.append((sim_idx, start, temporal_stride))
        self.uses_physical_spacing = (
            all(bool(rec["physical_spacing"]) for rec in self._cache)
            if self._cache else False
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        sim_idx, start, temporal_stride = self.windows[idx]
        rec = self._cache[sim_idx]
        offsets = np.arange(
            self.context_length + self.prediction_horizon,
            dtype=np.int64,
        ) * temporal_stride
        indices = start + offsets
        states = rec["states"][indices].astype(np.float32)
        times = rec["times"][indices].astype(np.float32)
        context = states[:self.context_length]
        future = states[self.context_length:]
        context_times = times[:self.context_length]
        if len(times) > self.context_length:
            eff_dt = _safe_dt(
                times[self.context_length - 1],
                times[self.context_length],
                fallback=float(temporal_stride),
            )
        else:
            eff_dt = float(temporal_stride)
        features = build_input_features(
            context,
            times=context_times,
            dx_spacing=rec["dx"],
            dy_spacing=rec["dy"],
            use_derivatives=self.use_derivatives,
            derivative_mode=self.derivative_mode,
        )
        metadata = rec["metadata"]
        return {
            "input_states": torch.from_numpy(features.copy()).float(),
            "target_states": torch.from_numpy(future.copy()).float(),
            "context_states": torch.from_numpy(context.copy()).float(),
            "states": torch.from_numpy(states.copy()).float(),
            "times": torch.from_numpy(times.copy()).float(),
            "dt": torch.tensor(eff_dt, dtype=torch.float32),
            "dx": torch.tensor(rec["dx"] if rec["dx"] is not None else 1.0, dtype=torch.float32),
            "dy": torch.tensor(rec["dy"] if rec["dy"] is not None else 1.0, dtype=torch.float32),
            "uses_physical_spacing": torch.tensor(bool(rec["physical_spacing"]), dtype=torch.bool),
            "stride": torch.tensor(temporal_stride, dtype=torch.long),
            "pde_vec": torch.from_numpy(rec["pde_vec"].copy()).float(),
            "sim_idx": torch.tensor(sim_idx, dtype=torch.long),
            "sim_id": str(metadata.get("sim_id", Path(rec["path"]).stem)),
            "family": str(metadata.get("family", "unknown")),
            "path": rec["path"],
        }

    @property
    def channel_names(self) -> List[str]:
        return (
            derivative_channel_names(TARGET_CHANNEL_NAMES)
            if self.use_derivatives else list(TARGET_CHANNEL_NAMES)
        )

    @property
    def target_channel_names(self) -> List[str]:
        return list(TARGET_CHANNEL_NAMES)


def split_paths(grid_dir: str | Path, val_frac: float = 0.25, seed: int = 0
                ) -> Tuple[List[str], List[str]]:
    """Split simulations, not windows, into train and validation."""
    grid_dir = Path(grid_dir)
    paths = sorted(str(p) for p in grid_dir.iterdir() if p.suffix == ".npz")
    if not paths:
        return [], []
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * len(paths)))) if len(paths) > 1 else 0
    n_val = min(n_val, max(0, len(paths) - 1))
    val = sorted(paths[i] for i in idx[:n_val])
    train = sorted(paths[i] for i in idx[n_val:])
    return train, val


def iter_input_tensors(dataset: CFDWindowDataset) -> Iterable[torch.Tensor]:
    for item in dataset:
        yield item["input_states"]
