"""Smoke checks for Report 1 family sampling.

This stays deliberately solver-free: it only exercises the parameter-family
sampler and representative stiffened-gas safety checks.
"""

from __future__ import annotations

import numpy as np

from .sweep_fvm import (
    FAMILY_SPECS,
    eos_reference_state,
    p_inf_from_ratio,
    stiffened_reference_safe,
)


def _samples(family: str, n: int = 128) -> list[dict]:
    rng = np.random.default_rng(20260505)
    spec = FAMILY_SPECS[family]
    return [spec.sample_physics(rng) for _ in range(n)]


def _check_metadata(params: dict) -> None:
    required = {
        "gamma", "C_v", "rho_inf", "T_inf",
        "eos_type", "p_inf", "p_inf_ratio",
    }
    missing = sorted(required - set(params))
    if missing:
        raise RuntimeError(f"sample is missing required EOS metadata: {missing}")


def _check_stiffened(params: dict, lo: float, hi: float) -> None:
    _check_metadata(params)
    ratio = float(params["p_inf_ratio"])
    if not (lo <= ratio <= hi):
        raise RuntimeError(f"p_inf_ratio={ratio} outside expected range [{lo}, {hi}]")
    expected_p_inf = p_inf_from_ratio(params, ratio)
    if not np.isclose(float(params["p_inf"]), expected_p_inf, rtol=1e-6, atol=1e-9):
        raise RuntimeError(
            f"p_inf={params['p_inf']} is inconsistent with p_inf_ratio={ratio}; "
            f"expected {expected_p_inf}"
        )
    if not stiffened_reference_safe(params):
        raise RuntimeError(f"unsafe stiffened-gas sample: {eos_reference_state(params)}")
    ref = eos_reference_state(params)
    if ref["P_ref"] <= 0.0 or ref["c2_arg_ref"] <= 0.0:
        raise RuntimeError(f"non-positive representative EOS state: {ref}")


def main() -> None:
    id_samples = _samples("id")
    if not all(p["eos_type"] == "ideal" for p in id_samples):
        raise RuntimeError("ID family should sample ideal gas only")
    if not all(float(p["p_inf"]) == 0.0 and float(p["p_inf_ratio"]) == 0.0 for p in id_samples):
        raise RuntimeError("ID family should keep p_inf and p_inf_ratio at zero")

    mild_samples = _samples("ood_mild")
    mild_ideal = [p for p in mild_samples if p["eos_type"] == "ideal"]
    mild_stiffened = [p for p in mild_samples if p["eos_type"] == "stiffened_gas"]
    if not mild_ideal or not mild_stiffened:
        raise RuntimeError("OOD-mild should sample a mixture of ideal and stiffened_gas")
    for params in mild_ideal:
        _check_metadata(params)
        if float(params["p_inf"]) != 0.0 or float(params["p_inf_ratio"]) != 0.0:
            raise RuntimeError("ideal OOD-mild samples should keep p_inf fields at zero")
    for params in mild_stiffened:
        _check_stiffened(params, 0.02, 0.08)

    hard_samples = _samples("ood_hard")
    if not all(p["eos_type"] == "stiffened_gas" for p in hard_samples):
        raise RuntimeError("OOD-hard should sample stiffened_gas only")
    for params in hard_samples:
        _check_stiffened(params, 0.08, 0.20)

    mild_ratios = np.array([float(p["p_inf_ratio"]) for p in mild_stiffened], dtype=np.float64)
    hard_ratios = np.array([float(p["p_inf_ratio"]) for p in hard_samples], dtype=np.float64)
    if not float(hard_ratios.mean()) > float(mild_ratios.mean()):
        raise RuntimeError("OOD-hard should use larger p_inf_ratio values than OOD-mild")

    print(
        "OK family sampling: "
        f"id={len(id_samples)} ideal-only, "
        f"ood_mild ideal={len(mild_ideal)} stiffened={len(mild_stiffened)} "
        f"ratio_mean={float(mild_ratios.mean()):.3f}, "
        f"ood_hard stiffened={len(hard_samples)} ratio_mean={float(hard_ratios.mean()):.3f}"
    )


if __name__ == "__main__":
    main()
