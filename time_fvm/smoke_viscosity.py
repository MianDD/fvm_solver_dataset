"""Smoke checks for shear and bulk viscosity constitutive laws."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


TIME_FVM_DIR = Path(__file__).resolve().parent
if str(TIME_FVM_DIR) not in sys.path:
    sys.path.insert(0, str(TIME_FVM_DIR))

from config_fvm import ConfigEllipse  # noqa: E402
from fvm_equation import PhysicalSetup  # noqa: E402


def _setup(law: str) -> PhysicalSetup:
    cfg = ConfigEllipse()
    cfg.device = "cpu"
    cfg.T_0 = 100.0
    cfg.viscosity = 2e-3
    cfg.visc_bulk = 8e-3
    cfg.viscosity_law = law
    cfg.power_law_n = 0.7
    cfg.S_const = 110.4
    cfg.gamma = 1.4
    cfg.C_v = 2.5
    cfg.eos_type = "ideal"
    cfg.p_inf = 0.0
    return PhysicalSetup(cfg)


def _assert_finite_nonnegative(name: str, values: torch.Tensor) -> None:
    if not torch.all(torch.isfinite(values)):
        raise RuntimeError(f"{name} produced non-finite values: {values}")
    if not torch.all(values >= 0):
        raise RuntimeError(f"{name} produced negative values: {values}")


def main() -> None:
    T = torch.tensor([[[50.0]], [[100.0]], [[200.0]]], dtype=torch.float32)

    constant = _setup("constant")
    mu_b_const = constant.bulk_viscosity(T)
    if not torch.allclose(mu_b_const, torch.ones_like(T) * constant.mu_b):
        raise RuntimeError(f"constant bulk viscosity should not vary with T: {mu_b_const}")

    power = _setup("power_law")
    mu_b_power = power.bulk_viscosity(T)
    if torch.allclose(mu_b_power[0], mu_b_power[-1]):
        raise RuntimeError(f"power-law bulk viscosity should vary with T: {mu_b_power}")

    sutherland = _setup("sutherland")
    mu_b_suth = sutherland.bulk_viscosity(T)
    if torch.allclose(mu_b_suth[0], mu_b_suth[-1]):
        raise RuntimeError(f"Sutherland bulk viscosity should vary with T: {mu_b_suth}")

    for law, setup in {
        "constant": constant,
        "power_law": power,
        "sutherland": sutherland,
    }.items():
        shear = setup.shear_viscosity(T)
        bulk = setup.bulk_viscosity(T)
        _assert_finite_nonnegative(f"{law} shear viscosity", shear)
        _assert_finite_nonnegative(f"{law} bulk viscosity", bulk)

    e_props = SimpleNamespace(
        T_faces=torch.ones((2, 2, 1), dtype=torch.float32) * 100.0,
        grad_V=torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[-0.2, 0.1], [0.0, 0.3]],
            ],
            dtype=torch.float32,
        ),
    )
    sutherland._tau(e_props)
    if not torch.all(torch.isfinite(sutherland.tau)):
        raise RuntimeError(f"_tau produced non-finite values: {sutherland.tau}")

    print(
        "OK viscosity smoke: "
        f"constant_bulk={mu_b_const.flatten().tolist()} "
        f"power_bulk={mu_b_power.flatten().tolist()} "
        f"sutherland_bulk={mu_b_suth.flatten().tolist()} "
        f"tau_shape={tuple(sutherland.tau.shape)}"
    )


if __name__ == "__main__":
    main()
