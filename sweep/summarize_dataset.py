"""Summarise robust FVM sweep outputs.

Usage:
    python -m sweep.summarize_dataset --sweep datasets/family_v1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List


PARAM_KEYS = [
    "gamma", "viscosity", "visc_bulk", "thermal_cond", "S_const", "C_v",
    "T_0", "rho_inf", "T_inf", "v_n_inf", "lnscale", "min_A", "max_A",
    "power_law_n", "p_inf", "p_inf_ratio",
]
CATEGORICAL_KEYS = ["family", "viscosity_law", "eos_type", "eos_version"]


def _snapshot_count(sim_dir: Path) -> int:
    return len([p for p in sim_dir.iterdir()
                if p.name.startswith("t_") and p.name.endswith(".npz")])


def _read_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "failed", "reason": f"invalid json: {exc}"}


def _legacy_status(sim_dir: Path) -> Dict:
    snapshots = _snapshot_count(sim_dir)
    has_mesh = (sim_dir / "mesh_props.npz").exists()
    if has_mesh and snapshots > 0:
        status = "success"
        reason = None
    else:
        status = "failed"
        reason = "missing mesh_props.npz" if not has_mesh else "no t_*.npz snapshots"
    cfg = _read_json(sim_dir / "config.json") if (sim_dir / "config.json").exists() else {}
    return {
        "status": status,
        "reason": reason,
        "number_of_snapshots_saved": snapshots,
        **{k: cfg.get(k) for k in PARAM_KEYS if k in cfg},
        **{k: cfg.get(k) for k in CATEGORICAL_KEYS if k in cfg},
    }


def collect_simulations(sweep_dir: Path) -> List[Dict]:
    rows = []
    for sim_dir in sorted(p for p in sweep_dir.iterdir()
                          if p.is_dir() and p.name.startswith("sim_")):
        status_path = sim_dir / "status.json"
        row = _read_json(status_path) if status_path.exists() else _legacy_status(sim_dir)
        row["sim_dir"] = str(sim_dir)
        row.setdefault("number_of_snapshots_saved", _snapshot_count(sim_dir))
        rows.append(row)
    return rows


def _numeric_summary(values: Iterable[float]) -> Dict:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {}
    return {
        "min": min(vals),
        "mean": mean(vals),
        "max": max(vals),
    }


def summarize(sweep_dir: str | Path) -> Dict:
    sweep_dir = Path(sweep_dir).resolve()
    rows = collect_simulations(sweep_dir)
    counts = Counter(row.get("status", "unknown") for row in rows)
    reasons = Counter()
    params = defaultdict(list)
    categories = {key: Counter() for key in CATEGORICAL_KEYS}
    snapshots = []
    for row in rows:
        status = row.get("status", "unknown")
        if status != "success":
            reason = row.get("reason") or row.get("failure_reason") or "unknown"
            stage = row.get("failed_stage") or row.get("invalid_stage") or "unknown"
            reasons[f"{stage}: {reason}"] += 1
        snapshots.append(row.get("number_of_snapshots_saved", 0))
        for key in PARAM_KEYS:
            if row.get(key) is not None:
                params[key].append(row[key])
        for key in CATEGORICAL_KEYS:
            if row.get(key) is not None:
                categories[key][str(row[key])] += 1

    return {
        "sweep_dir": str(sweep_dir),
        "n_total": len(rows),
        "n_success": counts.get("success", 0),
        "n_failed": counts.get("failed", 0),
        "n_invalid": counts.get("invalid", 0),
        "n_usable": counts.get("success", 0),
        "status_counts": dict(counts),
        "snapshot_count": _numeric_summary(snapshots),
        "parameter_summary": {k: _numeric_summary(v) for k, v in params.items()},
        "categorical_summary": {k: dict(v) for k, v in categories.items() if v},
        "common_failure_reasons": dict(reasons.most_common(10)),
        "ready_for_grid_adapter": counts.get("success", 0) > 0,
    }


def print_summary(summary: Dict) -> None:
    print(f"Sweep: {summary['sweep_dir']}")
    print(
        "Counts: "
        f"success={summary['n_success']} "
        f"failed={summary['n_failed']} "
        f"invalid={summary['n_invalid']} "
        f"usable={summary['n_usable']} "
        f"total={summary['n_total']}"
    )
    snap = summary["snapshot_count"]
    if snap:
        print(
            "Snapshots: "
            f"min={snap['min']:.0f} mean={snap['mean']:.2f} max={snap['max']:.0f}"
        )
    print("Parameter ranges:")
    for key, stats in summary["parameter_summary"].items():
        if stats:
            print(
                f"  {key}: min={stats['min']:.4g} "
                f"mean={stats['mean']:.4g} max={stats['max']:.4g}"
            )
    if summary.get("categorical_summary"):
        print("Categorical counts:")
        for key, counts in summary["categorical_summary"].items():
            joined = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
            print(f"  {key}: {joined}")
    if summary["common_failure_reasons"]:
        print("Common failure/invalid reasons:")
        for reason, count in summary["common_failure_reasons"].items():
            print(f"  {count}x {reason}")
    print(
        "Grid adapter readiness: "
        + ("ready" if summary["ready_for_grid_adapter"] else "not ready")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="sweep directory")
    ap.add_argument("--write-json", action="store_true",
                    help="write summary.json next to MANIFEST.json")
    args = ap.parse_args()
    summary = summarize(args.sweep)
    print_summary(summary)
    if args.write_json:
        out = Path(args.sweep).resolve() / "summary.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
