"""Boundary-channel constants shared by grid adapter and datasets."""

from __future__ import annotations

import numpy as np


BOUNDARY_CLASS_NAMES = [
    "fluid_interior",
    "inlet",
    "outlet",
    "obstacle_wall",
    "channel_wall",
    "solid",
]
BOUNDARY_MASK_VERSION = "boundary_mask_v1"
N_BOUNDARY_CLASSES = len(BOUNDARY_CLASS_NAMES)


def default_boundary_mask(mask: np.ndarray) -> np.ndarray:
    """Return a minimal one-hot boundary mask from a binary fluid mask.

    Old grid datasets do not contain detailed boundary classes.  For backward
    compatibility, treat all fluid pixels as fluid interior and all non-fluid
    pixels as solid.
    """
    fluid = np.asarray(mask, dtype=np.float32) > 0.5
    out = np.zeros((N_BOUNDARY_CLASSES, *fluid.shape), dtype=np.float32)
    out[0, fluid] = 1.0
    out[5, ~fluid] = 1.0
    return out


def boundary_channel_names(prefix: str = "boundary") -> list[str]:
    return [f"{prefix}_{name}" for name in BOUNDARY_CLASS_NAMES]
