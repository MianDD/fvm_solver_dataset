"""Plot qualitative CFD surrogate predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_model_from_checkpoint
from .dataset import build_input_features, load_grid_record, parse_strides


HORIZON_ERROR = "Only --horizon 1 is currently supported. Multi-horizon training is not implemented yet."


def _grid_files(grid_dir: str | Path):
    return sorted(p for p in Path(grid_dir).iterdir() if p.suffix == ".npz")


def _velocity_mag(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(arr[0] ** 2 + arr[1] ** 2)


def _cfg_value(ckpt: dict, key: str, default):
    cfg = ckpt.get("config", {})
    model_cfg = ckpt.get("model_config", {})
    return cfg.get(key, model_cfg.get(key, default))


def _settings(ckpt: dict, use_derivatives, derivative_mode, prediction_mode, integrator, strides):
    ckpt_strides = _cfg_value(ckpt, "strides", [1])
    stride_text = ckpt_strides if isinstance(ckpt_strides, str) else ",".join(str(s) for s in ckpt_strides)
    return {
        "use_derivatives": use_derivatives if use_derivatives is not None else bool(_cfg_value(ckpt, "use_derivatives", False)),
        "derivative_mode": derivative_mode or _cfg_value(ckpt, "derivative_mode", "central"),
        "prediction_mode": prediction_mode or _cfg_value(ckpt, "prediction_mode", "delta"),
        "integrator": integrator or _cfg_value(ckpt, "integrator", "euler"),
        "strides": strides or stride_text,
    }


def _plot_channel(fig_path: Path, title: str, context_img: np.ndarray,
                  target_img: np.ndarray, pred_img: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("context", context_img),
        ("target", target_img),
        ("prediction", pred_img),
        ("absolute error", np.abs(pred_img - target_img)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    for ax, (name, image) in zip(axes.ravel(), panels):
        im = ax.imshow(image, origin="lower", cmap="viridis")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(title)
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def plot_predictions(grid_dir: str | Path, ckpt_path: str | Path, out_dir: str | Path,
                     context: int | None = None, horizon: int | None = None,
                     num_examples: int = 2, stride: int | None = None,
                     device: str = "cpu", use_derivatives: bool | None = None,
                     derivative_mode: str | None = None,
                     prediction_mode: str | None = None,
                     integrator: str | None = None,
                     strides: str | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_model_from_checkpoint(ckpt_path, device=device)
    cfg = ckpt.get("config", {})
    context = int(context or cfg.get("context_length", 4))
    horizon = int(horizon or cfg.get("prediction_horizon", 1))
    if horizon != 1:
        raise ValueError(HORIZON_ERROR)
    settings = _settings(ckpt, use_derivatives, derivative_mode, prediction_mode, integrator, strides)
    temporal_stride = int(stride) if stride is not None else parse_strides(settings["strides"])[0]
    files = _grid_files(grid_dir)
    if not files:
        raise RuntimeError(f"No .npz grid files found in {grid_dir}")

    written = 0
    for path in files:
        if written >= num_examples:
            break
        rec = load_grid_record(path)
        states = rec["states"]
        times = rec["times"]
        target_idx = context * temporal_stride
        if states.shape[0] <= target_idx:
            print(f"SKIP {path.name}: not enough frames for context={context}, stride={temporal_stride}")
            continue
        context_idx = np.arange(context, dtype=np.int64) * temporal_stride
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

        rho_name = f"sim_{sim_id}_stride_{temporal_stride}_step_{step_label}_rho.png"
        _plot_channel(
            out_dir / rho_name,
            f"{path.stem} rho",
            context_last[2],
            target[2],
            pred[2],
        )
        vel_name = f"sim_{sim_id}_stride_{temporal_stride}_step_{step_label}_velmag.png"
        _plot_channel(
            out_dir / vel_name,
            f"{path.stem} velocity magnitude",
            _velocity_mag(context_last),
            _velocity_mag(target),
            _velocity_mag(pred),
        )
        written += 1
    if written == 0:
        raise RuntimeError("No examples had enough frames to plot.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None,
                    help="temporal stride to plot; defaults to the first stride in the checkpoint")
    ap.add_argument("--num-examples", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--use-derivatives", dest="use_derivatives", action="store_true")
    ap.add_argument("--no-derivatives", dest="use_derivatives", action="store_false")
    ap.set_defaults(use_derivatives=None)
    ap.add_argument("--derivative-mode", default=None, choices=["central"])
    ap.add_argument("--prediction-mode", default=None, choices=["delta", "derivative"])
    ap.add_argument("--integrator", default=None, choices=["euler"])
    ap.add_argument("--strides", default=None)
    args = ap.parse_args()
    if args.horizon is not None and args.horizon != 1:
        ap.error(HORIZON_ERROR)
    try:
        plot_predictions(
            args.grid, args.ckpt, args.out, context=args.context,
            horizon=args.horizon, num_examples=args.num_examples,
            stride=args.stride, device=args.device,
            use_derivatives=args.use_derivatives,
            derivative_mode=args.derivative_mode,
            prediction_mode=args.prediction_mode,
            integrator=args.integrator,
            strides=args.strides,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(f"Wrote prediction plots to {args.out}")


if __name__ == "__main__":
    main()
