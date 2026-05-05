"""Collect Report 1 tables and figures from existing run outputs.

This module is intentionally read-only with respect to experiments: it scans
training checkpoint folders and evaluation folders, then writes lightweight
CSV summaries and PNG figures for the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def _warn(message: str) -> None:
    print(f"WARNING: {message}")


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        _warn(f"could not parse JSON {path}: {exc}")
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _last_float(values: Any) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    return _safe_float(values[-1])


def _min_float(values: Any) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    finite = [_safe_float(v) for v in values]
    finite = [v for v in finite if v is not None]
    return min(finite) if finite else None


def _argmin(values: Any) -> int | None:
    if not isinstance(values, list) or not values:
        return None
    pairs = [(i, _safe_float(v)) for i, v in enumerate(values)]
    pairs = [(i, v) for i, v in pairs if v is not None]
    if not pairs:
        return None
    return min(pairs, key=lambda pair: pair[1])[0] + 1


def _rel_label(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    text = rel.as_posix().strip("/")
    return text or path.name


def _run_label(run_dir: Path, root: Path) -> str:
    return _rel_label(run_dir, root)


def _eval_label(metrics_path: Path, root: Path) -> str:
    return _rel_label(metrics_path.parent, root)


def _is_ablation(label: str, payload: Dict[str, Any] | None = None) -> bool:
    text = label.lower()
    if payload:
        text += " " + str(payload.get("checkpoint", "")).lower()
    return "ablat" in text or "ablate" in text


def _clean_key(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        _warn(f"skipping {path.name}: no rows available")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})
    print(f"Wrote {path}")


def _training_rows(runs_root: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not runs_root.exists():
        _warn(f"runs root does not exist: {runs_root}")
        return [], []
    rows: List[Dict[str, Any]] = []
    history_records: List[Dict[str, Any]] = []
    for history_path in sorted(runs_root.rglob("history.json")):
        run_dir = history_path.parent
        history = _load_json(history_path)
        if history is None:
            continue
        metrics = _load_json(run_dir / "metrics.json") or {}
        train_config = _load_json(run_dir / "train_config.json") or {}
        model_config = _load_json(run_dir / "model_config.json") or {}
        label = _run_label(run_dir, runs_root)
        val_loss = history.get("val_loss", [])
        row: Dict[str, Any] = {
            "run": label,
            "path": str(run_dir),
            "epochs": len(history.get("train_loss", [])),
            "best_epoch": _argmin(val_loss),
            "best_val_loss": metrics.get("best_val_loss", _min_float(val_loss)),
            "final_train_loss": metrics.get("final_train_loss", _last_float(history.get("train_loss"))),
            "final_val_loss": metrics.get("final_val_loss", _last_float(val_loss)),
            "final_train_pred_loss": metrics.get("final_train_pred_loss", _last_float(history.get("train_pred_loss"))),
            "final_val_pred_loss": metrics.get("final_val_pred_loss", _last_float(history.get("val_pred_loss"))),
            "final_train_pde_loss": metrics.get("final_train_pde_loss", _last_float(history.get("train_pde_loss"))),
            "final_val_pde_loss": metrics.get("final_val_pde_loss", _last_float(history.get("val_pde_loss"))),
            "final_train_pde_cont_loss": metrics.get("final_train_pde_cont_loss", _last_float(history.get("train_pde_cont_loss"))),
            "final_val_pde_cont_loss": metrics.get("final_val_pde_cont_loss", _last_float(history.get("val_pde_cont_loss"))),
            "final_train_pde_law_acc": metrics.get("final_train_pde_law_acc", _last_float(history.get("train_pde_law_acc"))),
            "final_val_pde_law_acc": metrics.get("final_val_pde_law_acc", _last_float(history.get("val_pde_law_acc"))),
            "final_train_pde_eos_acc": metrics.get("final_train_pde_eos_acc", _last_float(history.get("train_pde_eos_acc"))),
            "final_val_pde_eos_acc": metrics.get("final_val_pde_eos_acc", _last_float(history.get("val_pde_eos_acc"))),
            "n_train_sims": metrics.get("n_train_sims"),
            "n_val_sims": metrics.get("n_val_sims"),
            "n_train_windows": metrics.get("n_train_windows"),
            "n_val_windows": metrics.get("n_val_windows"),
            "model_params": metrics.get("model_params"),
            "attention_type": metrics.get("attention_type", model_config.get("attention_type", train_config.get("attention_type"))),
            "prediction_mode": metrics.get("prediction_mode", model_config.get("prediction_mode", train_config.get("prediction_mode"))),
            "use_derivatives": metrics.get("use_derivatives", train_config.get("use_derivatives")),
            "use_mask_channel": metrics.get("use_mask_channel", train_config.get("use_mask_channel")),
            "mask_loss": metrics.get("mask_loss", train_config.get("mask_loss")),
            "pde_aux_loss": metrics.get("pde_aux_loss", train_config.get("pde_aux_loss")),
            "context_length": train_config.get("context_length", model_config.get("max_context")),
            "batch_size": train_config.get("batch_size"),
            "d_model": train_config.get("d_model", model_config.get("d_model")),
            "n_layers": train_config.get("n_layers", model_config.get("n_layers")),
            "n_heads": train_config.get("n_heads", model_config.get("n_heads")),
            "patch_size": train_config.get("patch_size", model_config.get("patch_size")),
        }
        rows.append(row)
        history_records.append({"label": label, "path": history_path, "history": history})
    if not rows:
        _warn(f"no history.json files found under {runs_root}")
    return rows, history_records


def _pde_section(metrics: Dict[str, Any]) -> Dict[str, Any] | None:
    if "pde_identification" in metrics:
        return metrics["pde_identification"]
    if "continuous" in metrics and "overall" in metrics:
        return metrics
    return None


def _mean_rollout_mse(metrics: Dict[str, Any]) -> float | None:
    rollout = metrics.get("rollout", {})
    if not isinstance(rollout, dict):
        return None
    values = []
    for payload in rollout.values():
        if isinstance(payload, dict):
            value = _safe_float(payload.get("mse"))
            if value is not None:
                values.append(value)
    return sum(values) / len(values) if values else None


def _eval_records(eval_root: Path) -> List[Dict[str, Any]]:
    if not eval_root.exists():
        _warn(f"eval root does not exist: {eval_root}")
        return []
    records: List[Dict[str, Any]] = []
    for metrics_path in sorted(eval_root.rglob("metrics.json")):
        metrics = _load_json(metrics_path)
        if metrics is None:
            continue
        pde = _pde_section(metrics)
        if pde is None:
            pde = _load_json(metrics_path.parent / "pde_metrics.json")
        label = _eval_label(metrics_path, eval_root)
        records.append({
            "label": label,
            "metrics_path": metrics_path,
            "metrics": metrics,
            "pde": pde,
            "is_ablation": _is_ablation(label, metrics),
        })
    metric_dirs = {record["metrics_path"].parent for record in records}
    for pde_path in sorted(eval_root.rglob("pde_metrics.json")):
        if pde_path.parent in metric_dirs:
            continue
        pde = _load_json(pde_path)
        if pde is None:
            continue
        label = _eval_label(pde_path, eval_root)
        records.append({
            "label": label,
            "metrics_path": pde_path,
            "metrics": {},
            "pde": pde,
            "is_ablation": _is_ablation(label, {}),
        })
    if not records:
        _warn(f"no evaluation metrics found under {eval_root}")
    return records


def _eval_row(record: Dict[str, Any]) -> Dict[str, Any]:
    metrics = record["metrics"]
    one_step = metrics.get("one_step", {}) if isinstance(metrics, dict) else {}
    row: Dict[str, Any] = {
        "eval": record["label"],
        "metrics_path": str(record["metrics_path"]),
        "grid_dir": metrics.get("grid_dir"),
        "checkpoint": metrics.get("checkpoint"),
        "n_grid_files": metrics.get("n_grid_files"),
        "n_windows": metrics.get("n_windows"),
        "context_length": metrics.get("context_length"),
        "prediction_horizon": metrics.get("prediction_horizon"),
        "mse": one_step.get("mse"),
        "mae": one_step.get("mae"),
        "relative_l2": one_step.get("relative_l2"),
        "rollout_mean_mse": _mean_rollout_mse(metrics),
    }
    per_channel = one_step.get("per_channel", {})
    if isinstance(per_channel, dict):
        for name, payload in per_channel.items():
            if not isinstance(payload, dict):
                continue
            clean = _clean_key(str(name))
            row[f"mse_{clean}"] = payload.get("mse")
            row[f"mae_{clean}"] = payload.get("mae")
    settings = metrics.get("settings", {})
    if isinstance(settings, dict):
        row.update({
            "use_derivatives": settings.get("use_derivatives"),
            "use_mask_channel": settings.get("use_mask_channel"),
            "prediction_mode": settings.get("prediction_mode"),
            "strides": settings.get("strides"),
        })
    return row


def _pde_rows(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    pde = record.get("pde")
    if not isinstance(pde, dict):
        return []

    def make_row(section: Dict[str, Any], family: str) -> Dict[str, Any]:
        overall = section.get("overall", {})
        continuous = section.get("continuous", {})
        law = section.get("viscosity_law") or {}
        eos = section.get("eos_type") or {}
        return {
            "eval": record["label"],
            "family": family,
            "metrics_path": str(record["metrics_path"]),
            "n_samples": section.get("n_samples"),
            "mean_continuous_r2": overall.get("mean_continuous_r2", continuous.get("mean_r2")),
            "mean_continuous_mae": overall.get("mean_continuous_mae", continuous.get("mean_mae")),
            "mean_normalized_mae": continuous.get("mean_normalized_mae"),
            "law_accuracy": overall.get("law_accuracy", law.get("accuracy")),
            "law_degenerate": law.get("degenerate"),
            "law_num_classes_present": law.get("num_classes_present"),
            "eos_accuracy": overall.get("eos_accuracy", eos.get("accuracy")),
            "eos_degenerate": eos.get("degenerate"),
            "eos_num_classes_present": eos.get("num_classes_present"),
        }

    rows = [make_row(pde, "overall")]
    by_family = pde.get("by_family", {})
    if isinstance(by_family, dict):
        for family, section in sorted(by_family.items()):
            if isinstance(section, dict):
                rows.append(make_row(section, str(family)))
    return rows


def _plot_loss_curves(history_records: List[Dict[str, Any]], out_path: Path) -> None:
    if not history_records:
        _warn("skipping loss_curves.png: no history records")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        _warn(f"skipping loss_curves.png: matplotlib unavailable ({exc})")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    plotted = 0
    for record in history_records:
        history = record["history"]
        label = record["label"]
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        if train_loss:
            xs = list(range(1, len(train_loss) + 1))
            ax.plot(xs, train_loss, linestyle="--", linewidth=1.2, label=f"{label} train")
            plotted += 1
        if val_loss:
            xs = list(range(1, len(val_loss) + 1))
            ax.plot(xs, val_loss, linewidth=1.5, label=f"{label} val")
            plotted += 1
    if plotted == 0:
        _warn("skipping loss_curves.png: histories did not contain train_loss or val_loss")
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training and validation loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _plot_eval_mse(records: List[Dict[str, Any]], out_path: Path) -> None:
    pairs = []
    for record in records:
        if record["is_ablation"]:
            continue
        value = _safe_float(record["metrics"].get("one_step", {}).get("mse"))
        if value is not None:
            pairs.append((record["label"], value))
    if not pairs:
        _warn("skipping id_vs_ood_mse.png: no non-ablation one-step MSE values")
        return
    _plot_simple_bar(pairs, out_path, ylabel="one-step MSE", title="ID/OOD one-step MSE")


def _plot_channel_mse(records: List[Dict[str, Any]], out_path: Path) -> None:
    rows = []
    channels: List[str] = []
    for record in records:
        if record["is_ablation"]:
            continue
        per_channel = record["metrics"].get("one_step", {}).get("per_channel", {})
        if not isinstance(per_channel, dict):
            continue
        values = {}
        for name, payload in per_channel.items():
            if isinstance(payload, dict):
                value = _safe_float(payload.get("mse"))
                if value is not None:
                    values[str(name)] = value
                    if str(name) not in channels:
                        channels.append(str(name))
        if values:
            rows.append((record["label"], values))
    if not rows or not channels:
        _warn("skipping channel_mse.png: no per-channel MSE values")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        _warn(f"skipping channel_mse.png: matplotlib unavailable ({exc})")
        return
    labels = [label for label, _ in rows]
    x = np.arange(len(labels))
    width = min(0.8 / max(1, len(channels)), 0.2)
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(labels)), 4.0))
    for i, channel in enumerate(channels):
        offsets = x + (i - (len(channels) - 1) / 2.0) * width
        values = [payload.get(channel, 0.0) for _, payload in rows]
        ax.bar(offsets, values, width, label=channel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("one-step MSE")
    ax.set_title("Per-channel one-step MSE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _plot_pde_metrics(pde_rows: List[Dict[str, Any]], out_path: Path) -> None:
    rows = [row for row in pde_rows if row.get("family") == "overall"]
    if not rows:
        rows = pde_rows
    if not rows:
        _warn("skipping pde_aux_metrics.png: no PDE auxiliary metrics")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        _warn(f"skipping pde_aux_metrics.png: matplotlib unavailable ({exc})")
        return
    labels = [row["eval"] if row.get("family") == "overall" else f"{row['eval']}:{row['family']}" for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(max(8.0, 1.3 * len(labels)), 3.8))
    r2_values = [_safe_float(row.get("mean_continuous_r2")) for row in rows]
    if any(value is not None for value in r2_values):
        axes[0].bar(x, [0.0 if value is None else value for value in r2_values])
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].set_ylabel("mean R^2")
        axes[0].set_title("Continuous PDE identification")
    else:
        axes[0].text(0.5, 0.5, "No continuous R^2", ha="center", va="center")
        axes[0].set_axis_off()
    width = 0.35
    law_values = [_safe_float(row.get("law_accuracy")) for row in rows]
    eos_values = [_safe_float(row.get("eos_accuracy")) for row in rows]
    if any(value is not None for value in law_values + eos_values):
        axes[1].bar(x - width / 2, [0.0 if value is None else value for value in law_values], width, label="viscosity law")
        axes[1].bar(x + width / 2, [0.0 if value is None else value for value in eos_values], width, label="EOS")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].set_ylabel("accuracy")
        axes[1].set_title("Categorical PDE identification")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No categorical metrics", ha="center", va="center")
        axes[1].set_axis_off()
    for ax in axes:
        if ax.has_data():
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _plot_simple_bar(pairs: Sequence[tuple[str, float]], out_path: Path,
                     ylabel: str, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        _warn(f"skipping {out_path.name}: matplotlib unavailable ({exc})")
        return
    labels = [label for label, _ in pairs]
    values = [value for _, value in pairs]
    fig, ax = plt.subplots(figsize=(max(5.5, 1.1 * len(labels)), 3.6))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {out_path}")


def collect_report1_results(runs_root: str | Path,
                            eval_root: str | Path,
                            out_figures: str | Path,
                            out_tables: str | Path) -> Dict[str, Any]:
    """Collect Report 1 CSV tables and PNG figures.

    Missing inputs are warnings, not errors, so this can be run while only part
    of the experiment matrix has finished.
    """
    runs_root = Path(runs_root)
    eval_root = Path(eval_root)
    out_figures = Path(out_figures)
    out_tables = Path(out_tables)
    out_figures.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    train_rows, history_records = _training_rows(runs_root)
    eval_records = _eval_records(eval_root)
    eval_pairs = [(record, _eval_row(record)) for record in eval_records if record.get("metrics")]
    main_rows = [row for record, row in eval_pairs if not record["is_ablation"]]
    ablation_rows = [row for record, row in eval_pairs if record["is_ablation"]]
    pde_rows: List[Dict[str, Any]] = []
    for record in eval_records:
        pde_rows.extend(_pde_rows(record))

    _write_csv(out_tables / "training_history_summary.csv", train_rows)
    _write_csv(out_tables / "main_results.csv", main_rows)
    _write_csv(out_tables / "ablation_results.csv", ablation_rows)
    _write_csv(out_tables / "pde_aux_metrics.csv", pde_rows)

    _plot_loss_curves(history_records, out_figures / "loss_curves.png")
    _plot_eval_mse(eval_records, out_figures / "id_vs_ood_mse.png")
    _plot_channel_mse(eval_records, out_figures / "channel_mse.png")
    _plot_pde_metrics(pde_rows, out_figures / "pde_aux_metrics.png")

    return {
        "n_training_runs": len(train_rows),
        "n_eval_records": len(eval_records),
        "n_main_eval_rows": len(main_rows),
        "n_ablation_eval_rows": len(ablation_rows),
        "n_pde_rows": len(pde_rows),
        "out_figures": str(out_figures),
        "out_tables": str(out_tables),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect Report 1 CSV tables and PNG figures from saved training/evaluation outputs."
    )
    ap.add_argument("--runs", required=True, help="root containing training run folders with history.json")
    ap.add_argument("--eval-root", required=True, help="root containing evaluation folders with metrics.json")
    ap.add_argument("--out-figures", required=True, help="directory for Report 1 PNG figures")
    ap.add_argument("--out-tables", required=True, help="directory for Report 1 CSV tables")
    args = ap.parse_args()
    summary = collect_report1_results(
        runs_root=args.runs,
        eval_root=args.eval_root,
        out_figures=args.out_figures,
        out_tables=args.out_tables,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
