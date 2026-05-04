from __future__ import annotations
import torch
from abc import ABC
from cprint import c_print

from sparse_utils import plot_points, plot_edges, plot_interp_cell
from fvm_mesh import FVMMesh
from edge_process import FVMEdgeInfo
from t_solvers import FVMCells
from integrators import get_solver
from config_fvm import ConfigFVM
from sparse_utils import to_csr


class PhysicalSetup:
    """ Set physical properties of fluid. """
    tau: torch.Tensor       # shape = [n_edges, 2, 2]
    P_face: torch.Tensor    # shape = [n_edges, 2, 1]
    c: torch.Tensor         # shape = [n_edges, 2, 1]

    def __init__(self, cfg: ConfigFVM):
        self.device = cfg.device

        self.T_0 = cfg.T_0
        self.mu = cfg.viscosity
        self.viscosity_law = getattr(cfg, "viscosity_law", "sutherland")
        self.power_law_n = getattr(cfg, "power_law_n", 0.75)
        self.mu_b = cfg.visc_bulk
        self.S_const = cfg.S_const
        self.C_v = cfg.C_v
        self.gamma = cfg.gamma
        self.R = self.C_v * (cfg.gamma - 1)
        self.eos_type = getattr(cfg, "eos_type", "ideal")
        self.p_inf = float(getattr(cfg, "p_inf", 0.0))
        if self.viscosity_law not in {"sutherland", "constant", "power_law"}:
            raise ValueError(f"Unknown viscosity_law={self.viscosity_law!r}")
        if self.eos_type not in {"ideal", "stiffened_gas"}:
            raise ValueError(f"Unknown eos_type={self.eos_type!r}")

    def state_to_primative(self, state: torch.Tensor):
        """ Convert from conserved quantities (momentum, rho, energy) to primitives (velocity, rho, T) """
        momentum, rho, Q = state[:, [0, 1]], state[:,[2]], state[:,[3]]

        V = momentum / rho
        T = 1 / self.C_v * (Q / rho - 0.5 * V.square().sum(dim=1, keepdim=True))
        primatives = torch.cat([V, rho, T], dim=-1)

        return primatives, state

    def primatives_to_state(self, V, rho, T):
        """ Convert from primitives (velocity, rho, T) to conserved quantities (momentum, rho, energy)
            V.shape = [..., 2]
            rho.shape = [..., 1]
            T.shape = [..., 1]
         """

        momentum = V * rho
        Q = rho * (self.C_v * T + 0.5 * V.norm(dim=-1, keepdim=True) ** 2)
        return momentum, rho, Q

    def _tau(self, E_props: FVMEdgeInfo):
        """ Compute stress tensor:
                tau = mu * (grad(V) + grad(V).T) + mu_b * div(V) * I
         """
        T = E_props.T_faces       # shape = [n_edges, edges=2, n_comp=1]

        # Strain and invariants
        D, I1, I2 = self._strain_values(E_props)        # shape = [n_edges, 2, 2]
        I1 = I1.view(-1, 1, 1)   # [n_edges, 1, 1]. Trace of D.

        mu = self.shear_viscosity(T)  # shape = [n_edges, edges=2, n_comp=1]
        # Bulk viscosity: Proportional to T^2
        mu_b = self.mu_b * T ** 2 / self.T_0 ** 2

        eye = torch.eye(2, device=self.device)
        self.tau = -2 * mu * D - mu_b * I1 * eye  # shape = [n_edges, 2, 2]

    def shear_viscosity(self, T: torch.Tensor) -> torch.Tensor:
        """Dynamic shear viscosity for the selected constitutive law."""
        if self.viscosity_law == "sutherland":
            return self.mu * (T / self.T_0) ** 1.5 * (self.T_0 + self.S_const) / (T + self.S_const)
        if self.viscosity_law == "constant":
            return torch.ones_like(T) * self.mu
        if self.viscosity_law == "power_law":
            return self.mu * (T / self.T_0) ** self.power_law_n
        raise ValueError(f"Unknown viscosity_law={self.viscosity_law!r}")

    def _strain_values(self, E_props: FVMEdgeInfo):
        """ Compute strain tensor:
                epsilon = 0.5 * (grad(V) + grad(V).T)
            Then compute the 2D invariants:
                I1 = tr(D)  (divergence)
                I2 = tr(D^2) (Magnitude of deformation)
         """
        grad_V_t = E_props.grad_V  # shape = [n_edges, dim=2, n_comp=2]
        D = 0.5 * (grad_V_t + grad_V_t.permute(0, 2, 1))  # shape = [n_edges, 2, 2]

        # Invariants
        I1 = D[:, 0, 0] + D[:, 1, 1]  # shape = [n_edges]
        I2 = (D**2).sum(dim=(-1, -2))       # Since D is symmetric, this is faster.

        return D, I1, I2

    def _pressure(self, E_props: FVMEdgeInfo):
        """ Pressure force:
                P = rho * C_v * (gamma - 1) * T = R * rho * T
        """
        rho_faces = E_props.rho_faces               # shape = [n_edges, edges=2, n_comp=1]
        T_faces = E_props.T_faces                   # shape = [n_edges, edges=2, n_comp=1]

        self.P_face = self.eos_P(rho_faces, T_faces)
        self.c = self.eos_c(rho_faces, T_faces)     # shape = [n_edges, edges=2, n_comp=1]

    # General gas parameters.
    def eos_c(self, rho, T):
        """ Speed of sound, c^2 = dp/drho | s"""
        if self.eos_type == "stiffened_gas":
            P = self.eos_P(rho, T)
            rho_safe = rho.clamp_min(1e-12)
            return torch.sqrt((self.gamma * (P + self.p_inf).clamp_min(1e-12) / rho_safe).clamp_min(1e-12))
        return torch.sqrt(self.gamma * self.R * T)

    def eos_P(self, rho, T):
        """ Pressure EOS """
        if self.eos_type == "stiffened_gas":
            return self.R * rho * T - self.p_inf
        return self.R * rho * T

    def eos_T(self, rho, P):
        """ Inverse of eos_P """
        if self.eos_type == "stiffened_gas":
            return (P + self.p_inf) / (self.R * rho)
        return P / (self.R * rho)

    def eos_rho(self, P, T):
        """ Inverse of eos_P """
        if self.eos_type == "stiffened_gas":
            return (P + self.p_inf) / (self.R * T)
        return P / (self.R * T)

    def update(self, E_props: FVMEdgeInfo):
        # E_props = self.E_props
        #E_props.T_faces = E_props.T_faces.clamp(min=10, max=2000)
        self._tau(E_props)
        self._pressure(E_props)


class FVMEdgeFunc(ABC):
    device: str

    #@abstractmethod
    def edge_fluxes(self, fluxes=None):
        """ Compute flux for each edge
        """
        pass


class Adevction(FVMEdgeFunc):
    """ out_i = div(rho V V_i) for velocity V, i = {x, y}
        dims: Which dimensions of Us are advected.
    """
    E_props: FVMEdgeInfo

    def __init__(self, E_props: FVMEdgeInfo, phy_setup: PhysicalSetup, cfg: ConfigFVM, device="cpu"):
        self.device = device
        self.E_props = E_props
        self.phy_setup = phy_setup

    def edge_fluxes(self, fluxes=None):
        """ rho * U @ V.T @ n = rho V * phi
            f_x = rho V_x * phi
            f_y = rho V_y * phi
            f_rho = rho * phi
            f_E = (Q+p) * phi
        """
        E_props = self.E_props
        rho_faces = E_props.rho_faces # shape = [n_edges, edges=2, n_comp=1]
        phi = E_props.phi           # Linear interpolation of convection vector = (v_faces dot normal). shape = [n_edges, edges=2]
        mom_f = E_props.mom_faces
        Q_faces = E_props.Q_faces # = rho_faces * (1/2  * V_faces.norm(dim=-1, keepdim=True) ** 2 + self.C_v * T_faces)

        Q_p_P = Q_faces + self.phy_setup.P_face
        Us_f = torch.cat([mom_f, rho_faces, Q_p_P], dim=-1)  # shape = [n_edges, edges=2, n_comp=3]
        advec_flux = Us_f * phi.unsqueeze(-1)           # shape = [n_edges, edges=2, n_comp=3]
        advec_flux = advec_flux.mean(dim=1)              # shape = [n_edges, n_comp=3]

        if fluxes is None:
            return advec_flux
        else:
            fluxes += advec_flux


class Viscosity(FVMEdgeFunc):
    """ Viscous forces:
            Shear viscosity: div(mu grad(V)) = sum_f grad(V) * mu_f * l_f
            Bulk viscosity: k * grad(div(V))
    """
    E_props: FVMEdgeInfo
    def __init__(self, E_props: FVMEdgeInfo, stress_calc: PhysicalSetup, flux_mat, device="cpu"):
        self.device = device
        self.E_props = E_props
        self.stress_calc = stress_calc

    def edge_fluxes(self, fluxes=None):
        E_props = self.E_props

        tau = self.stress_calc.tau
        F = (tau * E_props.normals.unsqueeze(-1)).sum(dim=-2)  # shape = [n_edges, 2]

        if fluxes is None:
            fluxes = torch.zeros(self.E_props.n_edges, self.E_props.n_comp, device=self.device)
            fluxes[:, :2] = F
            return fluxes
        else:
            fluxes[:, :2] += F


class Heating(FVMEdgeFunc):
    """ Viscous heating term: div(tau V) = sum_f tau_f * V_f * n_f
        Thermal conductivity term:  div(grad(T)) = sum(grad(T) * n_f)
    """
    E_props: FVMEdgeInfo
    def __init__(self, E_props: FVMEdgeInfo, stress_calc:PhysicalSetup, cfg: ConfigFVM, device="cpu"):
        self.E_props = E_props
        self.stress_calc = stress_calc

        self.kappa = cfg.thermal_cond
        self.device = device

    def edge_fluxes(self, fluxes=None):
        E_props = self.E_props
        normals = E_props.normals       # shape = [n_edges, 2]
        V_face = E_props.Vs_faces       # shape = [n_edges, edges=2, n_comp=2]

        tau = self.stress_calc.tau

        V_face = V_face.mean(dim=1)     # shape = [n_edges, 2]
        heating = (tau * V_face.unsqueeze(1) * normals.unsqueeze(-1)).sum(dim=(-1, -2))

        """ Thermal conductivity:
                div(grad(T)) = sum(grad(T) * n_f)
        """
        grad_T_n = E_props.grad_T_n     # shape = [n_edges]
        heating -= self.kappa * grad_T_n * E_props.edge_len.squeeze()

        if fluxes is None:
            fluxes = torch.zeros(self.E_props.n_edges, self.E_props.n_comp, device=self.device)
            fluxes[:, 3] = heating
            return fluxes
        else:
            fluxes[:, 3] += heating


class PressureForce(FVMEdgeFunc):
    """ Special case. N
        grad(rho) = div(rho I) """
    E_props: FVMEdgeInfo

    def __init__(self, E_props: FVMEdgeInfo, phy_setup: PhysicalSetup, device="cpu"):
        self.device = device
        self.E_props = E_props
        self.phy_setup = phy_setup

    def edge_fluxes(self, fluxes=None):
        normals = self.E_props.normals             # shape = [n_edges, 2]
        P_face = self.phy_setup.P_face      # shape = [n_edges, edges=2, n_comp=1]

        P_face = P_face.mean(dim=1)  # shape = [n_edges, 1]
        P_n = P_face * normals                 # shape = [n_edges, 2]

        if fluxes is None:
            fluxes = torch.zeros(self.E_props.n_edges, self.E_props.n_comp, device=self.device)
            fluxes[:, :2] = P_n
            return fluxes # _flat
        else:
            fluxes[:, :2] += P_n


class KTDiffusion(FVMEdgeFunc):
    """ Diffusion term from K-T solver """
    E_props: FVMEdgeInfo

    def __init__(self, v_factor, phy_setup: PhysicalSetup, E_props: FVMEdgeInfo, device="cpu"):
        self.device = device
        self.v_factor = v_factor
        self.E_props = E_props
        self.phy_setup = phy_setup
        self.a_clip = 1

    def edge_fluxes(self, dt):
        E_props = self.E_props
        rho_face = E_props.rho_faces
        Vs_face = E_props.Vs_faces      # shape = [n_edges, edges=2, n_comp=2]
        Q_face = E_props.Q_faces   # shape = [n_edges, edges=2, n_comp=1]
        mom_face = E_props.mom_faces

        Us = torch.cat([mom_face, rho_face, Q_face], dim=2)  # shape = [n_edges, 2, n_comp]

        # Wavespeed is c + v_max. Clip velocity wavespeed to k*c + v_max
        Vs = Vs_face.norm(dim=-1)            # shape = [n_edges, edges=2]
        Vs_max = Vs.max(dim=1, keepdim=True).values   # shape = [n_edges, 1]
        c = self.phy_setup.c.max(dim=1).values  # shape = [n_edges, 1]
        # a = Vs_max + c

        # Reduce diffusion for velocity for low Mach number flows
        M = (Vs_max / (c + 1e-8)).abs()
        v_factor = torch.clamp(M, min=self.v_factor, max=1)
        a = torch.cat([v_factor * c, v_factor * c, c, c], dim=1)  # shape = [n_edges, n_comp]
        a = a + Vs_max  # shape = [n_edges, n_comp]

        # Flux = a/2 * (U_L - U_R) * edge_len
        edge_len = E_props.edge_len
        kt_fluxes = (a/2) * (Us[:, 0] - Us[:, 1]) * edge_len  # shape = [n_edges, n_comp]
        return kt_fluxes


class FVMEquation:
    mesh: FVMMesh
    E_props: FVMEdgeInfo
    edges: FVMEdgeFunc
    cells: FVMCells
    n_comp: int

    def __init__(self, cfg: ConfigFVM, phy_setup: PhysicalSetup, mesh: FVMMesh, n_comp, bc_tag, us_init=None):
        self.cfg = cfg
        self.phy_setup = phy_setup
        self.device = cfg.device
        self.mesh = mesh
        self.n_comp = n_comp

        device = self.device

        # Physical parameters

        E_props = FVMEdgeInfo(self.phy_setup, cfg, mesh, n_comp, bc_tag, device=device)
        self.cells = FVMCells(mesh.n_cells, n_comp, init_val=us_init, phys_setup=self.phy_setup, device=device)
        self.E_props = E_props

        # Matrix for converting edge fluxes to cell divergence
        self.flux_mat = self.build_flux_mat(mesh.tri_to_edge, -mesh.tri_edge_signs, mesh.n_edges, mesh.areas)  # shape = [n_cells, n_edge]

        self.P_force = PressureForce(E_props, self.phy_setup, device=device)
        self.U_advect = Adevction(E_props, self.phy_setup, cfg=cfg, device=device)
        self.U_visc = Viscosity(E_props, self.phy_setup, flux_mat=self.flux_mat, device=device)
        self.Heat = Heating(E_props, self.phy_setup, cfg=cfg, device=device)
        self.KT_diff = KTDiffusion(cfg.v_factor, self.phy_setup, E_props, device=device)

        # self.t_solver = Adams4PC(self.cells, self, cfg=cfg)
        self.t_solver = get_solver(self.cells, self, cfg) #ButcherAdapt(self.cells, self, name="RK3_SSP4", cfg=cfg)

        E_props.clear_temp()
        c_print("Done FVMEquation", color="bright_magenta")


    def solve(self):
        self.t_solver.solve()


    def forward(self, primatives, dt, t):
        """ primatives.shape = (n_cells, n_component) """
        E_props = self.E_props
        E_props.precompute_shared(primatives, dt)

        self.phy_setup.update(E_props)

        # Advection term
        fluxes = self.U_advect.edge_fluxes()
        # Pressure term
        self.P_force.edge_fluxes(fluxes)
        # Viscosity term
        self.U_visc.edge_fluxes(fluxes)
        # Heating term
        self.Heat.edge_fluxes(fluxes)
        # MUSCL term
        fluxes += self.KT_diff.edge_fluxes(dt)
        # Compute divergence
        divergence = self._flux_to_div(fluxes)

        return divergence

    def build_flux_mat(self, tri_to_edge, tri_edge_sign, n_edges, areas, dtype=torch.float32):
        """
        Build the incidence matrix T of shape (n_tri, n_edges).
        For each triangle i and local edge j, we set:
            T[i, tri_to_edge[i, j]] = tri_edge_sign[i, j].
        """
        c_print("Constricting flux matrix", color="bright_magenta")
        # and that self.areas is a tensor of length n_tri.
        n_tri, n_local = tri_to_edge.shape  # typically, n_local == 3

        # Create row indices: each triangle i contributes n_local entries.
        row_indices = torch.arange(n_tri).unsqueeze(1).expand(n_tri, n_local).reshape(-1)

        # Flatten the edge indices from tri_to_edge for column indices.
        col_indices = tri_to_edge.reshape(-1)
        # Flatten the sign values from tri_edge_sign.
        values = tri_edge_sign.reshape(-1).to(dtype)
        # Compute the inverse areas (A_inv is diagonal) and scale the nonzero values.
        areas_inv = (1.0 / areas.cpu()).to(dtype)
        D_values = values * areas_inv[row_indices]
        # Stack row and column indices for the sparse tensor.
        D_indices = torch.stack([row_indices, col_indices])

        D_shape = [n_tri, n_edges]
        flux_mat = torch.sparse_coo_tensor(D_indices, D_values, size=D_shape, device="cpu", dtype=dtype).coalesce()
        flux_mat = to_csr(flux_mat, self.device)
        return flux_mat

    def _flux_to_div(self, fluxes):
        """ Compute cell divergence using fluxes.
            fluxes.shape = (n_edges * N_component)

            du/dt = -div(flux) = -sum_i (sign_i * flux_i)
        """
        # Matrix version
        divergence = torch.mm(self.flux_mat, fluxes)  # shape: (n_cells * n_component,)
        return divergence

    def plot_flux(self, fluxes, title="Fluxes", show_index=False, lims=None, Xlims=None):
        plot_edges(self.mesh.vertices.cpu(), self.mesh.edges.cpu(), title=title, colors=fluxes, show_index=show_index, lims=lims, Xlims=Xlims)

    def plot_cells(self, values, title="Cell Values", show_index=False, lims=None, Xlims=None):
        plot_points(self.mesh.centroids.cpu(), values.T, show_index=show_index, title=title, lims=lims, Xlims=Xlims)

    def plot_interp(self, values, title="Cell Values", Xlims=None, resolution=2000):
        plot_interp_cell(self.mesh.vertices, values.T, self.mesh.triangles, title=title, Xlims=Xlims)

    def pretty_plot(self, primatives, Xlims=None, title=None):
        Vx, Vy, rho, T = primatives[:, 0], primatives[:, 1], primatives[:, 2], primatives[:, 3]

        P = self.phy_setup.R * rho * T
        c = torch.sqrt(P / rho)

        Mx, My = Vx / c, Vy / c
        M_num = torch.sqrt(Mx ** 2 + My ** 2)

        plot_vals = torch.stack([P, M_num, self.divergence[:, 3] ], dim=0)

        title = [f"Pressure: {title}", f"Mach number: {title}", f'Heating: {title}']
        plot_interp_cell(self.mesh.vertices, plot_vals, self.mesh.triangles, title=title, Xlims=Xlims)
