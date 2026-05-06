"""Plot qualitative CFD surrogate predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_model_from_checkpoint
from .dataset import build_input_features, load_grid_record, parse_strides, resolve_start_index
from .field_visualization import (
    FIELD_CHOICES,
    LEGACY_VELOCITY_MAG_FIELD,
    field_slug,
    parse_sim_ids,
    plot_prediction_panels,
    scalar_field,
    sim_id_matches,
)


HORIZON_ERROR = "Only --horizon 1 is currently supported. Multi-horizon training is not implemented yet."


def _grid_files(grid_dir: str | Path):
    return sorted(p for p in Path(grid_dir).iterdir() if p.suffix == ".npz")


def _cfg_value(ckpt: dict, key: str, default):
    cfg = ckpt.get("config", {})
    model_cfg = ckpt.get("model_config", {})
    return cfg.get(key, model_cfg.get(key, default))


def _settings(ckpt: dict, use_derivatives, use_boundary_channels,
              derivative_mode, prediction_mode, integrator, strides):
    ckpt_strides = _cfg_value(ckpt, "strides", [1])
    stride_text = ckpt_strides if isinstance(ckpt_strides, str) else ",".join(str(s) for s in ckpt_strides)
    return {
        "use_derivatives": use_derivatives if use_derivatives is not None else bool(_cfg_value(ckpt, "use_derivatives", False)),
        "use_mask_channel": bool(_cfg_value(ckpt, "use_mask_channel", False)),
        "use_boundary_channels": (
            use_boundary_channels
            if use_boundary_channels is not None
            else bool(_cfg_value(ckpt, "use_boundary_channels", False))
        ),
        "derivative_mode": derivative_mode or _cfg_value(ckpt, "derivative_mode", "central"),
        "prediction_mode": prediction_mode or _cfg_value(ckpt, "prediction_mode", "delta"),
        "integrator": integrator or _cfg_value(ckpt, "integrator", "euler"),
        "strides": strides or stride_text,
    }


def plot_predictions(grid_dir: str | Path, ckpt_path: str | Path, out_dir: str | Path,
                     context: int | None = None, horizon: int | None = None,
                     num_examples: int = 2, stride: int | None = None,
                     device: str = "cpu", use_derivatives: bool | None = None,
                     use_boundary_channels: bool | None = None,
                     derivative_mode: str | None = None,
                     prediction_mode: str | None = None,
                     integrator: str | None = None,
                     strides: str | None = None,
                     start_offset: int = 0,
                     t_start: float | None = None,
                     dpi: int = 250,
                     save_pdf: bool = False,
                     fig_scale: float = 1.0,
                     sim_ids: str | None = None,
                     max_plots: int | None = None,
                     field: str | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_model_from_checkpoint(ckpt_path, device=device)
    cfg = ckpt.get("config", {})
    context = int(context or cfg.get("context_length", 4))
    horizon = int(horizon or cfg.get("prediction_horizon", 1))
    if horizon != 1:
        raise ValueError(HORIZON_ERROR)
    settings = _settings(
        ckpt,
        use_derivatives,
        use_boundary_channels,
        derivative_mode,
        prediction_mode,
        integrator,
        strides,
    )
    temporal_stride = int(stride) if stride is not None else parse_strides(settings["strides"])[0]
    files = _grid_files(grid_dir)
    if not files:
        raise RuntimeError(f"No .npz grid files found in {grid_dir}")

    fields = [field] if field else ["rho", LEGACY_VELOCITY_MAG_FIELD]
    selected_ids = parse_sim_ids(sim_ids)
    plots_written = 0
    examples_written = 0
    examples = []
    for path in files:
        if max_plots is not None and plots_written >= max_plots:
            break
        if max_plots is None and examples_written >= num_examples:
            break
        rec = load_grid_record(path)
        if not sim_id_matches(path, rec["metadata"], selected_ids):
            continue
        states = rec["states"]
        times = rec["times"]
        start = resolve_start_index(
            times,
            times_available=bool(rec.get("times_available", True)),
            start_offset=start_offset,
            t_start=t_start,
            label=path.name,
        )
        target_idx = start + context * temporal_stride
        if states.shape[0] <= target_idx:
            print(
                f"SKIP {path.name}: not enough frames for context={context}, "
                f"stride={temporal_stride}, start={start}"
            )
            continue
        context_idx = start + np.arange(context, dtype=np.int64) * temporal_stride
        context_states = states[context_idx].astype(np.float32)
        context_times = times[context_idx].astype(np.float32)
        target = states[target_idx].astype(np.float32)
        dt = float(times[target_idx] - context_times[-1])
        if not np.isfinite(dt) or dt <= 0:
            print(f"WARNING: {path.name} has missing/non-monotonic times; using dt=1.0")
            dt = 1.0
        features = build_input_features(
            context_states,
            times=context_times,
            dx_spacing=rec["dx"],
            dy_spacing=rec["dy"],
            use_derivatives=settings["use_derivatives"],
            derivative_mode=settings["derivative_mode"],
            mask=rec["mask"],
            boundary_mask=rec["boundary_mask"],
            use_boundary_channels=settings["use_boundary_channels"],
            use_mask_channel=settings["use_mask_channel"],
        )
        states_t = torch.from_numpy(features).unsqueeze(0).to(device)
        current_t = torch.from_numpy(context_states[-1]).unsqueeze(0).to(device)
        with torch.no_grad():
            update = model.predict_update(states_t, normalised=False)[:, -1]
            pred = model.integrate_update(
                current_t,
                update,
                prediction_mode=settings["prediction_mode"],
                integrator=settings["integrator"],
                dt=torch.tensor([dt], dtype=torch.float32, device=device),
            ).squeeze(0).cpu().numpy()
        context_last = context_states[-1]
        sim_id = rec["metadata"].get("sim_id", path.stem)
        step_label = f"{target_idx:04d}"
        plotted_fields = []
        for plot_field in fields:
            if max_plots is not None and plots_written >= max_plots:
                break
            context_img = scalar_field(context_last, plot_field, rec["dx"], rec["dy"], rec["mask"])
            target_img = scalar_field(target, plot_field, rec["dx"], rec["dy"], rec["mask"])
            pred_img = scalar_field(pred, plot_field, rec["dx"], rec["dy"], rec["mask"])
            slug = field_slug(plot_field)
            out_name = f"sim_{sim_id}_stride_{temporal_stride}_step_{step_label}_{slug}.png"
            plot_prediction_panels(
                out_dir / out_name,
                plot_field,
                f"{path.stem} {plot_field}",
                context_img,
                target_img,
                pred_img,
                dpi=dpi,
                fig_scale=fig_scale,
                save_pdf=save_pdf,
            )
            plotted_fields.append(plot_field)
            plots_written += 1
        if not plotted_fields:
            continue
        examples.append({
            "file": str(path),
            "sim_id": str(sim_id),
            "start_index": int(start),
            "context_indices": [int(i) for i in context_idx.tolist()],
            "target_index": int(target_idx),
            "target_time": float(times[target_idx]) if target_idx < len(times) else None,
            "stride": int(temporal_stride),
            "fields": plotted_fields,
        })
        examples_written += 1
    if plots_written == 0:
        raise RuntimeError("No examples matched the filters or had enough frames to plot.")
    metadata = {
        "grid_dir": str(Path(grid_dir).resolve()),
        "checkpoint": str(Path(ckpt_path).resolve()),
        "context_length": context,
        "prediction_horizon": horizon,
        "start_offset": int(start_offset),
        "t_start": t_start,
        "stride": int(temporal_stride),
        "fields": fields,
        "dpi": int(dpi),
        "save_pdf": bool(save_pdf),
        "fig_scale": float(fig_scale),
        "sim_ids": sim_ids,
        "max_plots": max_plots,
        "plots_written": int(plots_written),
        "settings": settings,
        "examples": examples,
    }
    (out_dir / "plot_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--start-offset", type=int, default=0,
                    help="number of saved snapshots to skip before plotting")
    ap.add_argument("--t-start", type=float, default=None,
                    help="physical time threshold for the first context frame")
    ap.add_argument("--stride", type=int, default=None,
                    help="temporal stride to plot; defaults to the first stride in the checkpoint")
    ap.add_argument("--num-examples", type=int, default=2)
    ap.add_argument("--max-plots", type=int, default=None,
                    help="maximum number of figure files to write")
    ap.add_argument("--field", choices=FIELD_CHOICES, default=None,
                    help="physical field to plot; omit to preserve legacy rho and velocity-magnitude output")
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--save-pdf", action="store_true")
    ap.add_argument("--fig-scale", type=float, default=1.0)
    ap.add_argument("--sim-ids", default=None,
                    help='comma-separated sim IDs such as "42,1157,1614"')
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--use-derivatives", dest="use_derivatives", action="store_true")
    ap.add_argument("--no-derivatives", dest="use_derivatives", action="store_false")
    ap.set_defaults(use_derivatives=None)
    ap.add_argument("--use-boundary-channels", dest="use_boundary_channels", action="store_true")
    ap.add_argument("--no-boundary-channels", dest="use_boundary_channels", action="store_false")
    ap.set_defaults(use_boundary_channels=None)
    ap.add_argument("--derivative-mode", default=None, choices=["central"])
    ap.add_argument("--prediction-mode", default=None, choices=["delta", "derivative"])
    ap.add_argument("--integrator", default=None, choices=["euler"])
    ap.add_argument("--strides", default=None)
    args = ap.parse_args()
    if args.horizon is not None and args.horizon != 1:
        ap.error(HORIZON_ERROR)
    if args.start_offset < 0:
        ap.error("--start-offset must be non-negative.")
    if args.dpi <= 0:
        ap.error("--dpi must be positive.")
    if args.fig_scale <= 0.0:
        ap.error("--fig-scale must be positive.")
    if args.max_plots is not None and args.max_plots <= 0:
        ap.error("--max-plots must be positive.")
    try:
        plot_predictions(
            args.grid, args.ckpt, args.out, context=args.context,
            horizon=args.horizon, num_examples=args.num_examples,
            stride=args.stride, device=args.device,
            use_derivatives=args.use_derivatives,
            use_boundary_channels=args.use_boundary_channels,
            derivative_mode=args.derivative_mode,
            prediction_mode=args.prediction_mode,
            integrator=args.integrator,
            strides=args.strides,
            start_offset=args.start_offset,
            t_start=args.t_start,
            dpi=args.dpi,
            save_pdf=args.save_pdf,
            fig_scale=args.fig_scale,
            sim_ids=args.sim_ids,
            max_plots=args.max_plots,
            field=args.field,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(f"Wrote prediction plots to {args.out}")


if __name__ == "__main__":
    main()
