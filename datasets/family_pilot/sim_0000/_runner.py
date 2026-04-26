
import os, sys, json, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **kw: None
sys.path.insert(0, 'C:\\Users\\Lenovo\\Desktop\\MPhil Dissertation\\fvm_solver-main\\fvm_solver-main')
sys.path.insert(0, os.path.join('C:\\Users\\Lenovo\\Desktop\\MPhil Dissertation\\fvm_solver-main\\fvm_solver-main', 'time_fvm'))


def _run():
    with open('C:\\Users\\Lenovo\\Desktop\\MPhil Dissertation\\fvm_solver-main\\fvm_solver-main\\datasets\\family_pilot\\sim_0000\\config.json') as f:
        params = json.load(f)

    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    from time_fvm.config_fvm import ConfigEllipse, ConfigNozzle
    from time_fvm.fvm_equation import FVMEquation, PhysicalSetup
    from time_fvm.run_fvm import generate_mesh, init_conds_ellipses, init_conds_nozzle
    from time_fvm.fvm_mesh import FVMMesh

    cfg_cls = ConfigEllipse if params['problem'] == 'ellipse' else ConfigNozzle
    cfg = cfg_cls()
    cfg.device = params['device']
    cfg.compile = params['compile']
    cfg.dt = params['dt']
    cfg.n_iter = params['n_iter']
    cfg.end_t = params['end_t']
    cfg.print_i = params['print_i']
    cfg.plot_t = 1e9                           # disable interactive plotting
    cfg.save_t = params['save_t']
    cfg.lnscale = params['lnscale']
    cfg.min_A = params['min_A']
    cfg.max_A = params['max_A']
    # PDE physics
    cfg.gamma = params['gamma']
    cfg.viscosity = params['viscosity']
    cfg.visc_bulk = params['visc_bulk']
    cfg.thermal_cond = params['thermal_cond']
    cfg.C_v = params['C_v']
    cfg.T_0 = params['T_0']
    cfg.inlet_cfg.rho_inf = params['rho_inf']
    cfg.inlet_cfg.T_inf = params['T_inf']
    cfg.inlet_cfg.v_n_inf = params['v_n_inf']
    cfg.exit_cfg.rho_inf = params['rho_inf']
    cfg.exit_cfg.T_inf = params['T_inf']
    cfg.exit_cfg.v_n_inf = -params['v_n_inf']

    phy = PhysicalSetup(cfg)
    prob = generate_mesh(cfg)
    Xs, tri_idx, all_edgs, bc_edge_mask, edge_tag, bound_edgs = prob
    mesh = FVMMesh(Xs, tri_idx, all_edgs, bc_edge_mask, device=cfg.device)
    init_fn = init_conds_ellipses if params['problem'] == 'ellipse' else init_conds_nozzle
    bc_tags, us_init = init_fn(mesh, edge_tag, bound_edgs, phy, cfg)

    # Re-route the saver's output into our chosen sim_NNNN/ directory
    import time_fvm.ds_generation.saving as saving
    _orig_init = saving.Saver.__init__
    def _patched_init(self, E_props, save_dir=None):
        return _orig_init(self, E_props, save_dir=params['out_dir'])
    saving.Saver.__init__ = _patched_init

    solver = FVMEquation(cfg, phy, mesh, cfg.N_comp, bc_tags, us_init=us_init)
    solver.solve()
    print('SIM_DONE', params['out_dir'])


if __name__ == '__main__':
    # Required: mesh_gen/create_mesh.py uses multiprocessing 'spawn'
    _run()
