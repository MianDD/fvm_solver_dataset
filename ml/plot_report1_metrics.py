"""Create lightweight Report 1 figures from saved evaluation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def _load_metrics(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pde_section(metrics: Dict) -> Dict | None:
    if "pde_identification" in metrics:
        return metrics["pde_identification"]
    if "continuous" in metrics and "overall" in metrics:
        return metrics
    return None


def _plot_continuous_bars(pde: Dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping PDE bar plots: matplotlib unavailable ({exc})")
        return

    per_dim = pde.get("continuous", {}).get("per_dimension", {})
    if not per_dim:
        print("Skipping PDE bar plots: no continuous per-dimension metrics.")
        return
    names = list(per_dim.keys())
    maes = [float(per_dim[name]["mae"]) for name in names]
    r2_pairs = [(name, per_dim[name].get("r2")) for name in names if per_dim[name].get("r2") is not None]

    fig_w = max(7.0, 0.55 * len(names))
    fig, ax = plt.subplots(figsize=(fig_w, 3.4))
    ax.bar(names, maes)
    ax.set_ylabel("MAE")
    ax.set_title("PDE parameter MAE")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "pde_mae_bar.png", dpi=180)
    plt.close(fig)

    if r2_pairs:
        r2_names = [name for name, _ in r2_pairs]
        r2_values = [float(value) for _, value in r2_pairs]
        fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(r2_names)), 3.4))
        ax.bar(r2_names, r2_values)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("R^2")
        ax.set_title("PDE parameter R^2")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(out_dir / "pde_r2_bar.png", dpi=180)
        plt.close(fig)
    else:
        print("Skipping PDE R^2 plot: all R^2 values are undefined.")


def _plot_confusion(pde: Dict, out_dir: Path) -> None:
    law = pde.get("viscosity_law")
    if not law or not law.get("confusion_matrix"):
        print("Skipping viscosity-law confusion plot: metrics not available.")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping confusion plot: matplotlib unavailable ({exc})")
        return

    matrix = np.asarray(law["confusion_matrix"], dtype=np.int64)
    names = law.get("names") or [str(i) for i in range(matrix.shape[0])]
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Viscosity-law confusion")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_dir / "viscosity_law_confusion.png", dpi=180)
    plt.close(fig)


def _plot_rollout(metrics: Dict, out_dir: Path) -> None:
    rollout = metrics.get("rollout", {})
    if not rollout:
        print("Skipping rollout plot: no rollout metrics in input JSON.")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping rollout plot: matplotlib unavailable ({exc})")
        return

    xs = []
    ys = []
    labels = []
    for key, payload in sorted(rollout.items()):
        parts = key.split("_")
        try:
            step = int(parts[1])
        except Exception:
            continue
        xs.append(step)
        ys.append(float(payload["mse"]))
        labels.append(key)
    if not xs:
        print("Skipping rollout plot: rollout keys did not contain step counts.")
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(xs, ys, marker="o")
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("rollout steps")
    ax.set_ylabel("MSE")
    ax.set_title("Rollout error")
    fig.tight_layout()
    fig.savefig(out_dir / "rollout_error.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="metrics.json or pde_metrics.json from ml.evaluate")
    ap.add_argument("--out", required=True, help="output directory for PNG figures")
    args = ap.parse_args()

    metrics = _load_metrics(args.metrics)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pde = _pde_section(metrics)
    if pde is None:
        print("No PDE identification metrics found; PDE plots skipped.")
    else:
        _plot_continuous_bars(pde, out_dir)
        _plot_confusion(pde, out_dir)
    _plot_rollout(metrics, out_dir)
    print(f"Wrote Report 1 figures to {out_dir}")


if __name__ == "__main__":
    main()
