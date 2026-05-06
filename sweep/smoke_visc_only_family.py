"""Smoke checks for viscosity-only Report 1 families."""

from __future__ import annotations

import numpy as np

from .sweep_fvm import FAMILY_SPECS, eos_reference_state


FIXED_KEYS = ("gamma", "C_v", "T_0", "rho_inf", "T_inf", "v_n_inf")
EXPECTED_FIXED = {
    "gamma": 1.4,
    "C_v": 2.5,
    "T_0": 100.0,
    "rho_inf": 1.0,
    "T_inf": 100.0,
    "v_n_inf": 4.0,
}
PR = 0.71


def _samples(family: str, n: int = 128) -> list[dict]:
    rng = np.random.default_rng(20260506)
    spec = FAMILY_SPECS[family]
    return [spec.sample_physics(rng) for _ in range(n)]


def _check_common(samples: list[dict], lo: float, hi: float) -> np.ndarray:
    viscosities = np.array([float(p["viscosity"]) for p in samples], dtype=np.float64)
    if not np.all((lo <= viscosities) & (viscosities <= hi)):
        raise RuntimeError(f"viscosity samples outside [{lo}, {hi}]: {viscosities.min()}..{viscosities.max()}")
    if not float(viscosities.max()) > float(viscosities.min()):
        raise RuntimeError("viscosity-only family should vary shear viscosity")
    for params in samples:
        if params["eos_type"] != "ideal":
            raise RuntimeError("viscosity-only families should use ideal gas only")
        if params["viscosity_law"] != "constant":
            raise RuntimeError("viscosity-only families should use constant viscosity law")
        if float(params["p_inf"]) != 0.0 or float(params["p_inf_ratio"]) != 0.0:
            raise RuntimeError("viscosity-only families should keep p_inf fields at zero")
        if float(params["visc_bulk"]) != 0.0:
            raise RuntimeError("viscosity-only families should use zero bulk viscosity")
        for key in FIXED_KEYS:
            if not np.isclose(float(params[key]), EXPECTED_FIXED[key], rtol=0.0, atol=1e-12):
                raise RuntimeError(f"{key} should be fixed at {EXPECTED_FIXED[key]}, got {params[key]}")
        cp = float(params["gamma"]) * float(params["C_v"])
        expected_k = float(params["viscosity"]) * cp / PR
        if not np.isclose(float(params["thermal_cond"]), expected_k, rtol=1e-7, atol=1e-12):
            raise RuntimeError(
                f"thermal_cond={params['thermal_cond']} inconsistent with mu*cp/Pr={expected_k}"
            )
        if params.get("thermal_cond_policy") != "fixed_prandtl":
            raise RuntimeError("thermal_cond_policy should be fixed_prandtl")
        if not np.isclose(float(params.get("Pr", -1.0)), PR, rtol=0.0, atol=1e-12):
            raise RuntimeError("Pr metadata should be recorded as 0.71")
        ref = eos_reference_state(params)
        if ref["P_ref"] <= 0.0 or ref["c2_arg_ref"] <= 0.0:
            raise RuntimeError(f"representative ideal-gas state should be safe: {ref}")
    return viscosities


def main() -> None:
    id_samples = _samples("id_visc_only")
    ood_samples = _samples("ood_visc_only")
    id_mu = _check_common(id_samples, 1e-3, 5e-3)
    ood_mu = _check_common(ood_samples, 5e-3, 2e-2)
    if not float(ood_mu.min()) >= float(id_mu.max()):
        raise RuntimeError(
            "OOD viscosity-only range should be disjoint from and higher than ID: "
            f"id max={id_mu.max()} ood min={ood_mu.min()}"
        )
    print(
        "OK viscosity-only family sampling: "
        f"id_mu=[{id_mu.min():.3e}, {id_mu.max():.3e}] "
        f"ood_mu=[{ood_mu.min():.3e}, {ood_mu.max():.3e}] "
        f"Pr={PR}"
    )


if __name__ == "__main__":
    main()
