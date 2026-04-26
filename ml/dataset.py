"""PyTorch Dataset that streams sliding-window contexts from regular-grid
.npz files produced by :mod:`ml.grid_adapter`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class CFDWindowDataset(Dataset):
    """Each item is one (context + target) window from one simulation.

    The window is ``(tau + horizon)`` consecutive snapshots; given the
    first ``tau`` the model is asked to predict the next ``horizon``.

    .. note::
        We also keep the data in memory (one .npz per sim, all loaded
        eagerly) because a typical sweep produces only ~30 sims of
        ~20 snapshots each at 64x96, well under 1 GB total.
    """

    def __init__(self, paths: List[str | Path],
                 context_length: int = 4, prediction_horizon: int = 1,
                 stride: int = 1):
        self.paths = [Path(p) for p in paths]
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride

        self._cache = []
        self.windows: List[Tuple[int, int]] = []      # (sim_idx, start_step)
        win = context_length + prediction_horizon
        for sim_idx, p in enumerate(self.paths):
            z = np.load(p, allow_pickle=True)
            snaps = z["snapshots"]                    # (T, C, H, W)
            self._cache.append({
                "snapshots": snaps,
                "pde_vec": z["pde_vec"],
            })
            T = snaps.shape[0]
            for s in range(0, T - win + 1, stride):
                self.windows.append((sim_idx, s))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        sim_idx, start = self.windows[idx]
        rec = self._cache[sim_idx]
        end = start + self.context_length + self.prediction_horizon
        win = rec["snapshots"][start:end]             # (tau+1, C, H, W)
        return {
            "states": torch.from_numpy(win.copy()).float(),
            "pde_vec": torch.from_numpy(rec["pde_vec"].copy()).float(),
            "sim_idx": sim_idx,
        }

    @property
    def channel_names(self) -> List[str]:
        return ["V_x", "V_y", "rho", "T"]


def split_paths(grid_dir: str | Path, val_frac: float = 0.25, seed: int = 0
                ) -> Tuple[List[str], List[str]]:
    """Split simulations (not windows) into train and validation.

    Splitting at the simulation level prevents leakage: every PDE config is
    seen entirely in train OR entirely in val.
    """
    grid_dir = Path(grid_dir)
    paths = sorted(str(p) for p in grid_dir.iterdir() if p.suffix == ".npz")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths)); rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * len(paths))))
    val = sorted(paths[i] for i in idx[:n_val])
    train = sorted(paths[i] for i in idx[n_val:])
    return train, val
