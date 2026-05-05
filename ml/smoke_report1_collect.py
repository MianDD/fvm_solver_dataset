"""Smoke test for Report 1 result collection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .report1_collect_results import collect_report1_results


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _history_payload(scale: float = 1.0) -> dict:
    return {
        "train_loss": [0.4 * scale, 0.2 * scale],
        "val_loss": [0.45 * scale, 0.25 * scale],
        "train_pred_loss": [0.04 * scale, 0.02 * scale],
        "val_pred_loss": [0.05 * scale, 0.025 * scale],
        "train_pde_loss": [1.2 * scale, 0.9 * scale],
        "val_pde_loss": [1.3 * scale, 1.0 * scale],
        "train_pde_law_acc": [0.5, 0.8],
        "val_pde_law_acc": [0.4, 0.7],
        "train_pde_eos_acc": [0.6, 0.85],
        "val_pde_eos_acc": [0.55, 0.75],
    }


def _training_metrics_payload(scale: float = 1.0) -> dict:
    return {
        "best_val_loss": 0.25 * scale,
        "final_train_loss": 0.2 * scale,
        "final_val_loss": 0.25 * scale,
        "final_train_pred_loss": 0.02 * scale,
        "final_val_pred_loss": 0.025 * scale,
        "final_train_pde_loss": 0.9 * scale,
        "final_val_pde_loss": 1.0 * scale,
        "final_train_pde_law_acc": 0.8,
        "final_val_pde_law_acc": 0.7,
        "final_train_pde_eos_acc": 0.85,
        "final_val_pde_eos_acc": 0.75,
        "n_train_sims": 2,
        "n_val_sims": 1,
        "model_params": 1234,
        "attention_type": "factorized",
        "prediction_mode": "derivative",
        "use_derivatives": True,
        "use_mask_channel": True,
        "mask_loss": True,
        "pde_aux_loss": True,
    }


def _eval_metrics_payload(label: str, mse: float, law_acc: float, eos_acc: float) -> dict:
    pde = {
        "n_samples": 8,
        "continuous": {
            "mean_mae": 0.1,
            "mean_r2": 0.6,
            "mean_normalized_mae": 0.2,
            "per_dimension": {
                "gamma": {"mae": 0.01, "rmse": 0.02, "r2": 0.8},
                "p_inf_ratio": {"mae": 0.03, "rmse": 0.04, "r2": 0.4},
            },
        },
        "viscosity_law": {
            "names": ["sutherland", "constant", "power_law"],
            "accuracy": law_acc,
            "confusion_matrix": [[3, 0, 0], [0, 2, 1], [0, 1, 1]],
            "degenerate": False,
            "num_classes_present": 3,
        },
        "eos_type": {
            "names": ["ideal", "stiffened_gas"],
            "accuracy": eos_acc,
            "confusion_matrix": [[4, 1], [0, 3]],
            "degenerate": False,
            "num_classes_present": 2,
        },
        "overall": {
            "mean_continuous_r2": 0.6,
            "mean_continuous_mae": 0.1,
            "law_accuracy": law_acc,
            "eos_accuracy": eos_acc,
        },
        "by_family": {
            label: {
                "n_samples": 8,
                "continuous": {"mean_mae": 0.1, "mean_r2": 0.6, "mean_normalized_mae": 0.2},
                "viscosity_law": {"accuracy": law_acc, "degenerate": False, "num_classes_present": 3},
                "eos_type": {"accuracy": eos_acc, "degenerate": False, "num_classes_present": 2},
                "overall": {
                    "mean_continuous_r2": 0.6,
                    "mean_continuous_mae": 0.1,
                    "law_accuracy": law_acc,
                    "eos_accuracy": eos_acc,
                },
            }
        },
    }
    return {
        "grid_dir": f"datasets/gridded/{label}",
        "checkpoint": "checkpoints/report1/main/best_model.pt",
        "n_grid_files": 2,
        "n_windows": 16,
        "context_length": 4,
        "prediction_horizon": 1,
        "settings": {
            "use_derivatives": True,
            "use_mask_channel": True,
            "prediction_mode": "derivative",
            "strides": "1",
        },
        "one_step": {
            "mse": mse,
            "mae": mse ** 0.5,
            "relative_l2": 0.01,
            "per_channel": {
                "V_x": {"mse": mse * 0.5, "mae": 0.01},
                "V_y": {"mse": mse * 0.7, "mae": 0.01},
                "rho": {"mse": mse * 0.2, "mae": 0.01},
                "T": {"mse": mse * 1.2, "mae": 0.01},
            },
        },
        "rollout": {"steps_4_stride_1": {"mse": mse * 2.0}},
        "pde_identification": pde,
    }


def main() -> None:
    root = Path.cwd() / "datasets" / "_codex_report1_collect_smoke"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    try:
        root.mkdir(parents=True, exist_ok=True)
        runs = root / "checkpoints" / "report1"
        eval_root = root / "eval" / "report1"
        figures = root / "figures" / "report1"
        tables = root / "tables" / "report1"

        _write_json(runs / "main" / "history.json", _history_payload())
        _write_json(runs / "main" / "metrics.json", _training_metrics_payload())
        _write_json(runs / "main" / "train_config.json", {"context_length": 4, "batch_size": 2})
        _write_json(runs / "main" / "model_config.json", {"d_model": 32, "n_layers": 1, "n_heads": 4, "patch_size": 8})
        _write_json(runs / "ablate_global" / "history.json", _history_payload(scale=1.2))
        _write_json(runs / "ablate_global" / "metrics.json", _training_metrics_payload(scale=1.2))

        _write_json(eval_root / "id" / "metrics.json", _eval_metrics_payload("id", mse=0.01, law_acc=0.9, eos_acc=0.85))
        _write_json(eval_root / "ood_mild" / "metrics.json", _eval_metrics_payload("ood_mild", mse=0.02, law_acc=0.75, eos_acc=0.7))
        _write_json(eval_root / "ablate_global_on_id" / "metrics.json", _eval_metrics_payload("id", mse=0.03, law_acc=0.6, eos_acc=0.65))

        summary = collect_report1_results(runs, eval_root, figures, tables)
        assert summary["n_training_runs"] == 2
        assert summary["n_main_eval_rows"] == 2
        assert summary["n_ablation_eval_rows"] == 1
        assert summary["n_pde_rows"] >= 3

        expected_tables = [
            "main_results.csv",
            "ablation_results.csv",
            "pde_aux_metrics.csv",
            "training_history_summary.csv",
        ]
        expected_figures = [
            "loss_curves.png",
            "id_vs_ood_mse.png",
            "channel_mse.png",
            "pde_aux_metrics.png",
        ]
        for name in expected_tables:
            path = tables / name
            assert path.exists() and path.stat().st_size > 0, f"missing {path}"
        for name in expected_figures:
            path = figures / name
            assert path.exists() and path.stat().st_size > 0, f"missing {path}"
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("smoke_report1_collect OK")


if __name__ == "__main__":
    main()
