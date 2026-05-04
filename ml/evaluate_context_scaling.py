"""Evaluate one checkpoint over multiple temporal context lengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from .evaluate import evaluate


def _parse_contexts(values: Iterable[int]) -> List[int]:
    contexts = sorted({int(v) for v in values})
    if not contexts or any(v <= 0 for v in contexts):
        raise ValueError("--contexts must contain positive integers")
    return contexts


def _checkpoint_context_limit(ckpt_path: str | Path) -> tuple[str, int | None]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model_cfg = ckpt.get("model_config", {})
    pos_encoding = str(model_cfg.get("pos_encoding") or cfg.get("pos_encoding") or "learned_absolute")
    max_context = model_cfg.get("max_context", cfg.get("context_length"))
    return pos_encoding, int(max_context) if max_context is not None else None


def _summarize_context_metrics(metrics: Dict) -> Dict:
    one_step = metrics.get("one_step", {})
    pde = metrics.get("pde_identification")
    return {
        "status": "success",
        "mse": one_step.get("mse"),
        "mae": one_step.get("mae"),
        "relative_l2": one_step.get("relative_l2"),
        "pde_mean_r2": None if pde is None else pde.get("overall", {}).get("mean_continuous_r2"),
        "pde_mean_mae": None if pde is None else pde.get("overall", {}).get("mean_continuous_mae"),
        "pde_law_accuracy": None if pde is None else pde.get("overall", {}).get("law_accuracy"),
        "pde_eos_accuracy": None if pde is None else pde.get("overall", {}).get("eos_accuracy"),
        "num_windows": metrics.get("n_windows"),
        "metrics_path": metrics.get("metrics_path"),
    }


def evaluate_context_scaling(grid_dir: str | Path, ckpt_path: str | Path,
                             out_dir: str | Path, contexts: Iterable[int],
                             batch_size: int = 4, device: str = "cpu",
                             num_workers: int = 0, strides: str | None = None,
                             rollout_steps: str = "") -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contexts = _parse_contexts(contexts)
    pos_encoding, max_context = _checkpoint_context_limit(ckpt_path)

    results: Dict[str, Dict] = {}
    for context in contexts:
        key = str(context)
        if pos_encoding == "learned_absolute" and max_context is not None and context > max_context:
            reason = (
                f"context={context} exceeds learned_absolute max_context={max_context}; "
                "use a sinusoidal-position checkpoint for longer contexts"
            )
            print(f"SKIP context {context}: {reason}")
            results[key] = {"status": "skipped", "reason": reason}
            continue
        ctx_out = out_dir / f"context_{context}"
        try:
            metrics = evaluate(
                grid_dir,
                ckpt_path,
                ctx_out,
                context=context,
                horizon=1,
                batch_size=batch_size,
                device=device,
                num_workers=num_workers,
                strides=strides,
                rollout_steps=rollout_steps,
            )
            metrics_path = ctx_out / "metrics.json"
            metrics["metrics_path"] = str(metrics_path)
            summary = _summarize_context_metrics(metrics)
            summary["metrics_path"] = str(metrics_path)
            results[key] = summary
            print(
                f"OK context {context}: mse={summary['mse']} "
                f"pde_r2={summary['pde_mean_r2']} law_acc={summary['pde_law_accuracy']} "
                f"eos_acc={summary['pde_eos_accuracy']}"
            )
        except (RuntimeError, ValueError) as exc:
            reason = str(exc)
            print(f"SKIP context {context}: {reason}")
            results[key] = {"status": "skipped", "reason": reason}

    payload = {
        "grid_dir": str(Path(grid_dir).resolve()),
        "checkpoint": str(Path(ckpt_path).resolve()),
        "pos_encoding": pos_encoding,
        "checkpoint_max_context": max_context,
        "contexts": contexts,
        "results": results,
    }
    out_path = out_dir / "context_scaling_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--contexts", nargs="+", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--strides", default=None)
    ap.add_argument("--rollout-steps", default="")
    args = ap.parse_args()
    try:
        payload = evaluate_context_scaling(
            args.grid,
            args.ckpt,
            args.out,
            contexts=args.contexts,
            batch_size=args.batch,
            device=args.device,
            num_workers=args.num_workers,
            strides=args.strides,
            rollout_steps=args.rollout_steps,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
