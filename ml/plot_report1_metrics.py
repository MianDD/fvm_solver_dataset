"""Create lightweight Report 1 figures from saved evaluation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def _parse_metrics_list(value: str) -> List[Tuple[str, Path]]:
    entries: List[Tuple[str, Path]] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"metrics-list entry must be label:path, got {part!r}")
        label, path = part.split(":", 1)
        entries.append((label.strip(), Path(path.strip())))
    if not entries:
        raise ValueError("--metrics-list did not contain any label:path entries")
    return entries


def _mean_rollout_mse(metrics: Dict) -> float | None:
    rollout = metrics.get("rollout", {})
    values = [
        float(payload["mse"])
        for payload in rollout.values()
        if isinstance(payload, dict) and "mse" in payload
    ]
    if not values:
        return None
    return float(np.mean(values))


def _grouped_records_from_metrics_list(value: str) -> List[Dict]:
    records = []
    for label, path in _parse_metrics_list(value):
        metrics = _load_metrics(path)
        pde = _pde_section(metrics)
        records.append({
            "label": label,
            "metrics": metrics,
            "pde": pde,
            "mean_r2": None if pde is None else pde.get("overall", {}).get("mean_continuous_r2"),
            "law_accuracy": None if pde is None else pde.get("overall", {}).get("law_accuracy"),
            "rollout_mse": _mean_rollout_mse(metrics),
        })
    return records


def _grouped_records_from_by_family(pde: Dict) -> List[Dict]:
    records = []
    for label, fam_pde in sorted(pde.get("by_family", {}).items()):
        records.append({
            "label": label,
            "metrics": {},
            "pde": fam_pde,
            "mean_r2": fam_pde.get("overall", {}).get("mean_continuous_r2"),
            "law_accuracy": fam_pde.get("overall", {}).get("law_accuracy"),
            "rollout_mse": None,
        })
    return records


def _plot_grouped_bar(records: List[Dict], key: str, filename: str,
                      ylabel: str, title: str, out_dir: Path) -> None:
    pairs = [
        (rec["label"], rec.get(key))
        for rec in records
        if rec.get(key) is not None
    ]
    if not pairs:
        print(f"Skipping {filename}: no values available.")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping {filename}: matplotlib unavailable ({exc})")
        return
    labels = [label for label, _ in pairs]
    values = [float(value) for _, value in pairs]
    fig, ax = plt.subplots(figsize=(max(5.0, 1.2 * len(labels)), 3.4))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def _plot_grouped(records: List[Dict], out_dir: Path) -> None:
    if not records:
        return
    _plot_grouped_bar(
        records,
        "law_accuracy",
        "pde_law_accuracy_by_family.png",
        "accuracy",
        "Viscosity-law accuracy by family",
        out_dir,
    )
    _plot_grouped_bar(
        records,
        "mean_r2",
        "pde_mean_r2_by_family.png",
        "mean R^2",
        "Mean PDE continuous R^2 by family",
        out_dir,
    )
    _plot_grouped_bar(
        records,
        "rollout_mse",
        "rollout_mse_by_family.png",
        "mean rollout MSE",
        "Rollout MSE by family",
        out_dir,
    )


def _context_pairs(payload: Dict, key: str) -> List[Tuple[int, float]]:
    pairs: List[Tuple[int, float]] = []
    for context_text, result in payload.get("results", {}).items():
        if result.get("status") != "success":
            continue
        value = result.get(key)
        if value is None:
            continue
        pairs.append((int(context_text), float(value)))
    return sorted(pairs)


def _plot_context_scaling(payload: Dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping context-scaling plots: matplotlib unavailable ({exc})")
        return

    specs = [
        ("mse", "context_scaling_mse.png", "one-step MSE", "Context scaling: prediction MSE"),
        ("pde_mean_r2", "context_scaling_pde_r2.png", "mean PDE R^2", "Context scaling: PDE R^2"),
        ("pde_law_accuracy", "context_scaling_law_accuracy.png", "law accuracy", "Context scaling: viscosity-law accuracy"),
    ]
    for key, filename, ylabel, title in specs:
        pairs = _context_pairs(payload, key)
        if not pairs:
            print(f"Skipping {filename}: no successful context entries with {key}.")
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("context length")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(xs)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", help="metrics.json or pde_metrics.json from ml.evaluate")
    ap.add_argument("--context-scaling", help="context_scaling_metrics.json from ml.evaluate_context_scaling")
    ap.add_argument(
        "--metrics-list",
        default=None,
        help="comma-separated label:path entries, e.g. id:eval/id/metrics.json,ood_mild:eval/ood/metrics.json",
    )
    ap.add_argument("--out", required=True, help="output directory for PNG figures")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.metrics_list:
        _plot_grouped(_grouped_records_from_metrics_list(args.metrics_list), out_dir)
    if args.context_scaling:
        _plot_context_scaling(_load_metrics(args.context_scaling), out_dir)
    if args.metrics:
        metrics = _load_metrics(args.metrics)
        pde = _pde_section(metrics)
        if pde is None:
            print("No PDE identification metrics found; PDE plots skipped.")
        else:
            _plot_continuous_bars(pde, out_dir)
            _plot_confusion(pde, out_dir)
            _plot_grouped(_grouped_records_from_by_family(pde), out_dir)
        _plot_rollout(metrics, out_dir)
    if not args.metrics and not args.metrics_list and not args.context_scaling:
        raise SystemExit("ERROR: provide --metrics, --metrics-list, --context-scaling, or a combination")
    print(f"Wrote Report 1 figures to {out_dir}")


if __name__ == "__main__":
    main()
