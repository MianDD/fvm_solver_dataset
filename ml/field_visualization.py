"""Physical field helpers for Report 1 visualisation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np


FIELD_CHOICES = ("rho", "T", "V_x", "V_y", "vorticity", "grad_rho", "schlieren")
LEGACY_VELOCITY_MAG_FIELD = "velocity_magnitude"


def parse_sim_ids(value: str | None) -> set[str] | None:
    if value is None or not str(value).strip():
        return None
    ids = {part.strip().lower() for part in str(value).split(",") if part.strip()}
    expanded = set(ids)
    for item in ids:
        digits = _trailing_digits(item)
        if digits is not None:
            expanded.add(str(int(digits)))
    return expanded


def sim_id_matches(path: str | Path, metadata: dict, selected: set[str] | None) -> bool:
    if selected is None:
        return True
    candidates = set()
    for value in (metadata.get("sim_id"), Path(path).stem):
        if value is None:
            continue
        text = str(value).lower()
        candidates.add(text)
        digits = _trailing_digits(text)
        if digits is not None:
            candidates.add(str(int(digits)))
    return bool(candidates & selected)


def field_slug(field: str) -> str:
    return "velmag" if field == LEGACY_VELOCITY_MAG_FIELD else field.replace("_", "")


def scalar_field(state: np.ndarray, field: str, dx: float | None = None,
                 dy: float | None = None, mask: np.ndarray | None = None,
                 schlieren_alpha: float = 20.0) -> np.ndarray:
    """Return a 2D visualisation field from primitive state [V_x, V_y, rho, T]."""
    arr = np.asarray(state, dtype=np.float32)
    if arr.shape[0] < 4:
        raise ValueError(f"state must have at least 4 primitive channels, got {arr.shape}")
    sx = _safe_spacing(dx)
    sy = _safe_spacing(dy)
    if field == "V_x":
        image = arr[0]
    elif field == "V_y":
        image = arr[1]
    elif field == "rho":
        image = arr[2]
    elif field == "T":
        image = arr[3]
    elif field == LEGACY_VELOCITY_MAG_FIELD:
        image = np.sqrt(arr[0] ** 2 + arr[1] ** 2)
    elif field == "vorticity":
        dvy_dx, _ = _gradient_xy(arr[1], sx, sy)
        _, dvx_dy = _gradient_xy(arr[0], sx, sy)
        image = dvy_dx - dvx_dy
    elif field == "grad_rho":
        drho_dx, drho_dy = _gradient_xy(arr[2], sx, sy)
        image = np.sqrt(drho_dx ** 2 + drho_dy ** 2)
    elif field == "schlieren":
        drho_dx, drho_dy = _gradient_xy(arr[2], sx, sy)
        grad_rho = np.sqrt(drho_dx ** 2 + drho_dy ** 2)
        image = np.log1p(float(schlieren_alpha) * np.maximum(grad_rho, 0.0))
    else:
        raise ValueError(f"Unsupported field {field!r}")
    image = np.asarray(image, dtype=np.float32).copy()
    if mask is not None and np.asarray(mask).shape == image.shape:
        image[np.asarray(mask) <= 0.5] = np.nan
    return image


def plot_prediction_panels(out_path: str | Path, field: str, title: str,
                           context_img: np.ndarray, target_img: np.ndarray,
                           pred_img: np.ndarray, dpi: int = 250,
                           fig_scale: float = 1.0, save_pdf: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    cmap = _cmap_for_field(field)
    vmin, vmax = shared_limits([context_img, target_img, pred_img], field)
    error = np.abs(pred_img - target_img)
    err_vmin, err_vmax = error_limits(error)
    panels = [
        ("context", context_img, vmin, vmax),
        ("target", target_img, vmin, vmax),
        ("prediction", pred_img, vmin, vmax),
        ("absolute error", error, err_vmin, err_vmax),
    ]
    fig, axes = plt.subplots(
        1, 4, figsize=(14.0 * fig_scale, 3.6 * fig_scale),
        constrained_layout=True,
    )
    for ax, (name, image, lo, hi) in zip(axes.ravel(), panels):
        im = ax.imshow(image, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.76)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=int(dpi))
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_single_field(out_path: str | Path, field: str, title: str, image: np.ndarray,
                      dpi: int = 250, fig_scale: float = 1.0,
                      save_pdf: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    cmap = _cmap_for_field(field)
    vmin, vmax = shared_limits([image], field)
    fig, ax = plt.subplots(
        1, 1, figsize=(5.4 * fig_scale, 4.2 * fig_scale),
        constrained_layout=True,
    )
    im = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.82)
    fig.savefig(out_path, dpi=int(dpi))
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def shared_limits(images: Iterable[np.ndarray], field: str) -> tuple[float, float]:
    values = _finite_values(images)
    if values.size == 0:
        return 0.0, 1.0
    if field == "vorticity":
        max_abs = float(np.nanpercentile(np.abs(values), 99.0))
        if not np.isfinite(max_abs) or max_abs <= 0.0:
            max_abs = 1.0
        return -max_abs, max_abs
    if field in {"grad_rho", "schlieren"}:
        lo, hi = np.nanpercentile(values, [1.0, 99.0])
    else:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    return _expand_limits(float(lo), float(hi))


def error_limits(error: np.ndarray) -> tuple[float, float]:
    values = _finite_values([error])
    if values.size == 0:
        return 0.0, 1.0
    hi = float(np.nanpercentile(values, 99.0))
    if not np.isfinite(hi) or hi <= 0.0:
        hi = float(np.nanmax(values)) if values.size else 1.0
    if not np.isfinite(hi) or hi <= 0.0:
        hi = 1.0
    return 0.0, hi


def _safe_spacing(value: float | None) -> float:
    if value is None:
        return 1.0
    value = float(value)
    return value if np.isfinite(value) and value > 0.0 else 1.0


def _gradient_xy(image: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(image, dtype=np.float32)
    gx = np.zeros_like(arr, dtype=np.float32)
    gy = np.zeros_like(arr, dtype=np.float32)
    if arr.shape[1] > 1:
        gx[:, 1:-1] = 0.5 * (arr[:, 2:] - arr[:, :-2]) / dx
        gx[:, 0] = (arr[:, 1] - arr[:, 0]) / dx
        gx[:, -1] = (arr[:, -1] - arr[:, -2]) / dx
    if arr.shape[0] > 1:
        gy[1:-1, :] = 0.5 * (arr[2:, :] - arr[:-2, :]) / dy
        gy[0, :] = (arr[1, :] - arr[0, :]) / dy
        gy[-1, :] = (arr[-1, :] - arr[-2, :]) / dy
    return gx, gy


def _cmap_for_field(field: str):
    import matplotlib.pyplot as plt

    name = "coolwarm" if field == "vorticity" else "magma" if field in {"grad_rho", "schlieren"} else "viridis"
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("#d9d9d9")
    return cmap


def _finite_values(images: Iterable[np.ndarray]) -> np.ndarray:
    chunks = []
    for image in images:
        arr = np.asarray(image, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            chunks.append(finite.ravel())
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def _expand_limits(lo: float, hi: float) -> tuple[float, float]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if hi > lo:
        return lo, hi
    pad = max(1.0, abs(lo)) * 0.05
    return lo - pad, hi + pad


def _trailing_digits(value: str) -> str | None:
    match = re.search(r"(\d+)$", str(value))
    return match.group(1) if match else None
