"""Smoke test for high-quality physical field plotting."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .plot_fields import plot_fields


def _write_grid(path: Path) -> None:
    T, C, H, W = 4, 4, 32, 40
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, H, dtype=np.float32),
        np.linspace(-1.5, 1.5, W, dtype=np.float32),
        indexing="ij",
    )
    states = np.zeros((T, C, H, W), dtype=np.float32)
    for t in range(T):
        phase = 0.2 * t
        states[t, 0] = -yy + 0.1 * np.sin(3.0 * xx + phase)
        states[t, 1] = xx + 0.1 * np.cos(2.0 * yy - phase)
        states[t, 2] = 1.0 + 0.25 * np.exp(-((xx - 0.2 * t) ** 2 + yy ** 2) / 0.22)
        states[t, 3] = 100.0 + 2.0 * states[t, 2]
    mask = np.ones((H, W), dtype=np.float32)
    mask[(xx ** 2 / 0.18 ** 2 + yy ** 2 / 0.28 ** 2) < 1.0] = 0.0
    states[:, 0, mask < 0.5] = 0.0
    states[:, 1, mask < 0.5] = 0.0
    states[:, 2, mask < 0.5] = 1.0
    states[:, 3, mask < 0.5] = 100.0
    metadata = {
        "sim_id": "sim_0042",
        "family": "synthetic_plot",
        "status": "success",
        "number_of_snapshots": T,
        "channel_names": ["V_x", "V_y", "rho", "T"],
    }
    np.savez_compressed(
        path,
        states=states,
        snapshots=states,
        times=np.arange(T, dtype=np.float32) * 0.3,
        channel_names=np.array(["V_x", "V_y", "rho", "T"]),
        mask=mask,
        dx=np.float32(3.0 / (W - 1)),
        dy=np.float32(2.0 / (H - 1)),
        metadata_json=json.dumps(metadata),
        cfg_json=json.dumps({"sim_id": "sim_0042", "family": "synthetic_plot"}),
    )


def _assert_written(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"expected plot was not written: {path}")


def main() -> None:
    root = Path("datasets") / "_codex_smoke_plot_fields"
    shutil.rmtree(root, ignore_errors=True)
    try:
        grid = root / "grid"
        grid.mkdir(parents=True)
        _write_grid(grid / "sim_0042.npz")
        out = root / "figures"

        plot_fields(grid, out / "rho", field="rho", sim_ids="42", time_indices="1", dpi=120)
        plot_fields(grid, out / "vorticity", field="vorticity", sim_ids="sim_0042", last=True, dpi=120)
        plot_fields(grid, out / "schlieren", field="schlieren", sim_ids="42", last=True, dpi=120, save_pdf=True)

        _assert_written(out / "rho" / "sim_sim_0042_step_0001_rho.png")
        _assert_written(out / "vorticity" / "sim_sim_0042_step_0003_vorticity.png")
        _assert_written(out / "schlieren" / "sim_sim_0042_step_0003_schlieren.png")
        _assert_written(out / "schlieren" / "sim_sim_0042_step_0003_schlieren.pdf")
        print("OK plot fields smoke: rho, vorticity, schlieren PNG/PDF written")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
