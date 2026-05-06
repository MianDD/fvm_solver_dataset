"""Plot physical fields directly from gridded CFD trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import load_grid_record
from .field_visualization import (
    FIELD_CHOICES,
    field_slug,
    parse_sim_ids,
    plot_single_field,
    scalar_field,
    sim_id_matches,
)


def _grid_files(grid_dir: str | Path):
    return sorted(p for p in Path(grid_dir).iterdir() if p.suffix == ".npz")


def _parse_time_indices(value: str | None) -> list[int] | None:
    if value is None or not str(value).strip():
        return None
    indices = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not indices:
        return None
    return indices


def plot_fields(grid_dir: str | Path, out_dir: str | Path, field: str = "rho",
                sim_ids: str | None = None, time_indices: str | list[int] | None = None,
                last: bool = False, dpi: int = 250, save_pdf: bool = False,
                fig_scale: float = 1.0, max_plots: int | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = parse_sim_ids(sim_ids)
    files = _grid_files(grid_dir)
    if not files:
        raise RuntimeError(f"No .npz grid files found in {grid_dir}")
    if isinstance(time_indices, str):
        requested_indices = _parse_time_indices(time_indices)
    else:
        requested_indices = time_indices

    plots = []
    for path in files:
        if max_plots is not None and len(plots) >= max_plots:
            break
        rec = load_grid_record(path)
        if not sim_id_matches(path, rec["metadata"], selected_ids):
            continue
        states = rec["states"]
        times = rec["times"]
        if requested_indices is not None:
            indices = [int(i) for i in requested_indices]
        else:
            indices = [states.shape[0] - 1] if last or requested_indices is None else []
        for idx in indices:
            if max_plots is not None and len(plots) >= max_plots:
                break
            if idx < 0:
                idx = states.shape[0] + idx
            if idx < 0 or idx >= states.shape[0]:
                print(f"SKIP {path.name}: time index {idx} is outside [0, {states.shape[0] - 1}]")
                continue
            sim_id = rec["metadata"].get("sim_id", path.stem)
            image = scalar_field(states[idx], field, rec["dx"], rec["dy"], rec["mask"])
            slug = field_slug(field)
            out_name = f"sim_{sim_id}_step_{idx:04d}_{slug}.png"
            plot_single_field(
                out_dir / out_name,
                field,
                f"{path.stem} {field} step {idx}",
                image,
                dpi=dpi,
                fig_scale=fig_scale,
                save_pdf=save_pdf,
            )
            plots.append({
                "file": str(path),
                "sim_id": str(sim_id),
                "time_index": int(idx),
                "time": float(times[idx]) if idx < len(times) else None,
                "field": field,
                "png": str(out_dir / out_name),
                "pdf": str((out_dir / out_name).with_suffix(".pdf")) if save_pdf else None,
            })
    if not plots:
        raise RuntimeError("No field plots were written; check --sim-ids and --time-indices.")
    metadata = {
        "grid_dir": str(Path(grid_dir).resolve()),
        "out_dir": str(out_dir.resolve()),
        "field": field,
        "sim_ids": sim_ids,
        "time_indices": requested_indices,
        "last": bool(last),
        "dpi": int(dpi),
        "save_pdf": bool(save_pdf),
        "fig_scale": float(fig_scale),
        "max_plots": max_plots,
        "plots": plots,
    }
    (out_dir / "plot_fields_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim-ids", default=None,
                    help='comma-separated sim IDs such as "42,1157,1614"')
    ap.add_argument("--field", choices=FIELD_CHOICES, default="rho")
    ap.add_argument("--time-indices", default=None,
                    help='comma-separated saved-frame indices such as "7,10,15"; negative indices are allowed')
    ap.add_argument("--last", action="store_true",
                    help="plot the last saved frame; this is the default when --time-indices is omitted")
    ap.add_argument("--out-dir", dest="out_dir", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--save-pdf", action="store_true")
    ap.add_argument("--fig-scale", type=float, default=1.0)
    ap.add_argument("--max-plots", type=int, default=None)
    args = ap.parse_args()
    if args.dpi <= 0:
        ap.error("--dpi must be positive.")
    if args.fig_scale <= 0.0:
        ap.error("--fig-scale must be positive.")
    if args.max_plots is not None and args.max_plots <= 0:
        ap.error("--max-plots must be positive.")
    try:
        metadata = plot_fields(
            args.grid,
            args.out,
            field=args.field,
            sim_ids=args.sim_ids,
            time_indices=args.time_indices,
            last=args.last,
            dpi=args.dpi,
            save_pdf=args.save_pdf,
            fig_scale=args.fig_scale,
            max_plots=args.max_plots,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(f"Wrote {len(metadata['plots'])} field plot(s) to {args.out}")


if __name__ == "__main__":
    main()
