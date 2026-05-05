"""Smoke checks for ideal and stiffened-gas EOS formulas."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


TIME_FVM_DIR = Path(__file__).resolve().parent
if str(TIME_FVM_DIR) not in sys.path:
    sys.path.insert(0, str(TIME_FVM_DIR))

from config_fvm import ConfigEllipse  # noqa: E402
from fvm_equation import PhysicalSetup  # noqa: E402


def _setup(eos_type: str, p_inf: float) -> PhysicalSetup:
    cfg = ConfigEllipse()
    cfg.device = "cpu"
    cfg.gamma = 1.4
    cfg.C_v = 2.5
    cfg.T_0 = 100.0
    cfg.viscosity = 1e-3
    cfg.viscosity_law = "sutherland"
    cfg.power_law_n = 0.75
    cfg.visc_bulk = 1e-2
    cfg.thermal_cond = 1e-6
    cfg.S_const = 110.4
    cfg.eos_type = eos_type
    cfg.p_inf = float(p_inf)
    cfg.eos_version = "stiffened_gas_gamma_pinf_v1"
    return PhysicalSetup(cfg)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual, expected, rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"{name} mismatch: actual={actual} expected={expected}")


def main() -> None:
    rho = torch.tensor([[2.0], [1.3]], dtype=torch.float32)
    T = torch.tensor([[10.0], [14.0]], dtype=torch.float32)
    p_inf = 3.0

    ideal = _setup("ideal", p_inf=0.0)
    stiff = _setup("stiffened_gas", p_inf=p_inf)
    R = ideal.R

    P_ideal = ideal.eos_P(rho, T)
    P_stiff = stiff.eos_P(rho, T)
    _assert_close("ideal pressure", P_ideal, R * rho * T)
    _assert_close("stiffened pressure", P_stiff, R * rho * T - stiff.gamma * p_inf)
    if torch.allclose(P_ideal, P_stiff):
        raise RuntimeError("ideal and stiffened_gas pressures should differ when p_inf > 0")

    T_back = stiff.eos_T(rho, P_stiff)
    rho_back = stiff.eos_rho(P_stiff, T)
    _assert_close("stiffened inverse T", T_back, T)
    _assert_close("stiffened inverse rho", rho_back, rho)

    c_ideal = ideal.eos_c(rho, T)
    c_stiff = stiff.eos_c(rho, T)
    if torch.allclose(c_ideal, c_stiff):
        raise RuntimeError("ideal and stiffened_gas sound speeds should differ when p_inf > 0")
    if not torch.all(torch.isfinite(c_ideal)) or not torch.all(c_ideal > 0):
        raise RuntimeError(f"ideal sound speed invalid: {c_ideal}")
    if not torch.all(torch.isfinite(c_stiff)) or not torch.all(c_stiff > 0):
        raise RuntimeError(f"stiffened sound speed invalid: {c_stiff}")
    expected_c2 = stiff.gamma * (P_stiff + p_inf).clamp_min(1e-12) / rho.clamp_min(1e-12)
    _assert_close("stiffened sound speed", c_stiff, torch.sqrt(expected_c2.clamp_min(1e-12)))

    if stiff.eos_version != "stiffened_gas_gamma_pinf_v1":
        raise RuntimeError(f"unexpected eos_version={stiff.eos_version!r}")

    print(
        "OK EOS smoke: "
        f"P_ideal={P_ideal.flatten().tolist()} "
        f"P_stiff={P_stiff.flatten().tolist()} "
        f"c_ideal={c_ideal.flatten().tolist()} "
        f"c_stiff={c_stiff.flatten().tolist()}"
    )


if __name__ == "__main__":
    main()
