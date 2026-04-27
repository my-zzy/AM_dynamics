"""Single-step MPC diagnostic.

Builds the MPC, runs exactly ONE solver call from the nominal hover state,
then plots:
  - Predicted EE trajectory (from solver x[0..N]) vs reference
  - Per-stage cost breakdown (EE error, velocity, effort)
  - Predicted joint angles and input forces

Run:
    python demo/mpc_single_step_debug.py
    python demo/mpc_single_step_debug.py --rebuild
"""

import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'basic'))

from ams.model import AerialManipulatorModel
from ams.kinematics import forward_kinematics
from ams.casadi_dynamics import ca_forward_kinematics
from demo.mpc_trajectory import EETrajectory
from demo.mpc_controller import MPCController, _W_STAGE_DIAG, _W_TERMINAL_DIAG

import casadi as ca

# ── Config (match mpc_reach_test.py) ────────────────────────────────────────
HOVER_START = np.array([0.00, 0.0, 1.000])
EE_TARGET   = np.array([0.40, 0.0, 1.125])
N           = 20
DT          = 0.05
TRAJ_DUR    = 4.0

# ── Build nominal hover state ────────────────────────────────────────────────
def nominal_state():
    """x at hover: pos=HOVER_START, vel=0, quat=[0,0,0,1], omega=0, theta=0, tdot=0"""
    x = np.zeros(17)
    x[0:3]  = HOVER_START
    x[9]    = 1.0          # quaternion w=1  (index 9 = [x,y,z,w][3])
    return x

def ee_from_state(model, x):
    p_A   = x[0:3]
    q_A   = x[6:10]
    theta = x[13:15]
    _, p, _ = forward_kinematics(model, q_A, p_A, theta)
    return p[3]  # (3,)

def main(rebuild=False, enable_tc=False):
    model = AerialManipulatorModel()

    # Compute EE start from nominal hover
    x0 = nominal_state()
    p_ee_start = ee_from_state(model, x0)
    print(f'EE start   : {p_ee_start}')
    print(f'EE target  : {EE_TARGET}')
    print(f'EE distance: {np.linalg.norm(p_ee_start - EE_TARGET)*1000:.1f} mm')

    traj = EETrajectory(p_start=p_ee_start, p_end=EE_TARGET,
                        T_f=TRAJ_DUR, dt=DT)

    print(f'\nBuilding MPCController...')
    mpc = MPCController(traj=traj, model=model, N=N, dt=DT,
                        rebuild=rebuild,
                        enable_terminal_constraint=enable_tc)

    # ── Single solver call ────────────────────────────────────────────────────
    print('Running ONE solver step from nominal hover state...')
    t_query = 0.0   # start of trajectory
    u0, info = mpc.solve(x0, t_query)

    print(f'\nSolver status : {info["status"]}  '
          f'(0=OK, 2=maxiter, 3=minstep, 4=NaN)')
    print(f'Total cost    : {info["cost"]:.4f}')
    print(f'Solve time    : {info["solve_time"]*1000:.2f} ms')
    print(f'u0 = F=[{u0[0]:.3f},{u0[1]:.3f},{u0[2]:.3f}]  '
          f'tau=[{u0[3]:.3f},{u0[4]:.3f},{u0[5]:.3f}]  '
          f'jt=[{u0[6]:.3f},{u0[7]:.3f}]')

    # ── Extract predicted trajectory ─────────────────────────────────────────
    xs = np.array([mpc._solver.get(k, 'x') for k in range(N + 1)])  # (N+1, 17)
    us = np.array([mpc._solver.get(k, 'u') for k in range(N)])      # (N, 8)

    # EE positions along predicted horizon
    pred_ee = np.array([ee_from_state(model, xs[k]) for k in range(N + 1)])  # (N+1, 3)

    # Reference EE positions along same window
    t_horizon = np.arange(N + 1) * DT
    ref_ee = np.array([traj._p(t) for t in t_horizon])   # (N+1, 3)

    # ── Per-stage cost ────────────────────────────────────────────────────────
    W = _W_STAGE_DIAG
    stage_ee_cost   = np.zeros(N)
    stage_vel_cost  = np.zeros(N)
    stage_omg_cost  = np.zeros(N)
    stage_u_cost    = np.zeros(N)

    for k in range(N):
        x_k  = xs[k]
        u_k  = us[k]
        p_ee_k = pred_ee[k]
        p_ref_k = ref_ee[k]
        hover_f = model.total_mass * 9.81

        # residuals
        r_ee  = p_ee_k  - p_ref_k
        r_vel = x_k[3:6]               # v_A ref = 0
        r_omg = x_k[10:13]             # omega_A ref = 0
        r_u   = u_k.copy()
        r_u[2] -= hover_f              # effort relative to hover

        stage_ee_cost[k]  = 0.5 * r_ee  @ (W[0:3]  * r_ee)
        stage_vel_cost[k] = 0.5 * r_vel @ (W[3:6]  * r_vel)
        stage_omg_cost[k] = 0.5 * r_omg @ (W[6:9]  * r_omg)
        stage_u_cost[k]   = 0.5 * (r_u[0:3] @ (W[9:12]  * r_u[0:3])
                                   + r_u[3:6] @ (W[12:15] * r_u[3:6])
                                   + r_u[6:8] @ (W[15:17] * r_u[6:8]))

    total_stage = stage_ee_cost + stage_vel_cost + stage_omg_cost + stage_u_cost
    print(f'\nPer-stage cost breakdown (sum over horizon):')
    print(f'  EE position  : {stage_ee_cost.sum():.3f}  ({stage_ee_cost.sum()/total_stage.sum()*100:.1f}%)')
    print(f'  Linear vel   : {stage_vel_cost.sum():.3f}  ({stage_vel_cost.sum()/total_stage.sum()*100:.1f}%)')
    print(f'  Angular vel  : {stage_omg_cost.sum():.3f}  ({stage_omg_cost.sum()/total_stage.sum()*100:.1f}%)')
    print(f'  Input effort : {stage_u_cost.sum():.3f}  ({stage_u_cost.sum()/total_stage.sum()*100:.1f}%)')

    # ── Quaternion health ─────────────────────────────────────────────────────
    qnorms = np.array([np.linalg.norm(xs[k, 6:10]) for k in range(N + 1)])
    print(f'\nQuaternion norm along horizon: min={qnorms.min():.6f}  max={qnorms.max():.6f}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'Single-step MPC diagnostic  |  status={info["status"]}  '
                 f'cost={info["cost"]:.2f}  solve={info["solve_time"]*1000:.1f}ms\n'
                 f'N={N}  dt={DT*1000:.0f}ms  enable_tc={enable_tc}', fontsize=11)
    gs = gridspec.GridSpec(3, 3, hspace=0.55, wspace=0.40)

    t_ax = t_horizon

    # 1. EE x vs time
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t_ax, pred_ee[:, 0], 'b-o', ms=3, label='predicted')
    ax.plot(t_ax, ref_ee[:, 0],  'r--',       label='reference')
    ax.axhline(EE_TARGET[0], color='r', lw=0.7, ls=':')
    ax.set_title('EE x'); ax.set_xlabel('t [s]'); ax.set_ylabel('m')
    ax.legend(fontsize=7); ax.grid(True, lw=0.3)

    # 2. EE z vs time
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t_ax, pred_ee[:, 2], 'b-o', ms=3, label='predicted')
    ax.plot(t_ax, ref_ee[:, 2],  'r--',       label='reference')
    ax.axhline(EE_TARGET[2], color='r', lw=0.7, ls=':')
    ax.set_title('EE z'); ax.set_xlabel('t [s]'); ax.set_ylabel('m')
    ax.legend(fontsize=7); ax.grid(True, lw=0.3)

    # 3. xz spatial
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(pred_ee[:, 0], pred_ee[:, 2], 'b-o', ms=3, label='predicted')
    ax.plot(ref_ee[:, 0],  ref_ee[:, 2],  'r--',       label='reference')
    ax.scatter(*p_ee_start[[0, 2]], c='gray', s=80, zorder=5, label='start')
    ax.scatter(EE_TARGET[0], EE_TARGET[2], c='red', s=160, marker='*', zorder=6, label='target')
    ax.set_title('xz EE'); ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]')
    ax.set_aspect('equal'); ax.legend(fontsize=7); ax.grid(True, lw=0.3)

    # 4. Per-stage cost stacked
    ax = fig.add_subplot(gs[1, 0:2])
    kk = np.arange(N)
    ax.bar(kk, stage_ee_cost,  label='EE pos',      color='tab:blue')
    ax.bar(kk, stage_vel_cost, bottom=stage_ee_cost, label='vel', color='tab:orange')
    bot2 = stage_ee_cost + stage_vel_cost
    ax.bar(kk, stage_omg_cost, bottom=bot2,           label='omega', color='tab:green')
    bot3 = bot2 + stage_omg_cost
    ax.bar(kk, stage_u_cost,   bottom=bot3,           label='effort', color='tab:red')
    ax.set_title('Per-stage cost breakdown')
    ax.set_xlabel('stage k'); ax.set_ylabel('cost')
    ax.legend(fontsize=7); ax.grid(True, lw=0.3, axis='y')

    # 5. Input F_ext_z and joint torques
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(np.arange(N)*DT, us[:, 2], 'b-o', ms=3, label='F_z')
    ax.axhline(model.total_mass*9.81, color='b', lw=0.7, ls=':', label=f'mg={model.total_mass*9.81:.1f}N')
    ax2 = ax.twinx()
    ax2.plot(np.arange(N)*DT, us[:, 6], 'g-o', ms=3, label='τ_j1')
    ax2.plot(np.arange(N)*DT, us[:, 7], 'm-o', ms=3, label='τ_j2')
    ax2.set_ylabel('joint torque [Nm]', color='g')
    ax.set_title('Inputs'); ax.set_xlabel('t [s]'); ax.set_ylabel('F_z [N]')
    ax.legend(fontsize=7, loc='upper left'); ax2.legend(fontsize=7, loc='upper right')
    ax.grid(True, lw=0.3)

    # 6. Joint angles along predicted horizon
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t_ax, np.rad2deg(xs[:, 13]), 'b-o', ms=3, label='θ₁')
    ax.plot(t_ax, np.rad2deg(xs[:, 14]), 'r-o', ms=3, label='θ₂')
    ax.axhline( 90, color='k', lw=0.7, ls='--'); ax.axhline(-90, color='k', lw=0.7, ls='--')
    ax.set_title('Predicted joint angles'); ax.set_xlabel('t [s]'); ax.set_ylabel('deg')
    ax.legend(fontsize=7); ax.grid(True, lw=0.3)

    # 7. Platform position along horizon
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t_ax, xs[:, 0], label='x'); ax.plot(t_ax, xs[:, 1], label='y'); ax.plot(t_ax, xs[:, 2], label='z')
    ax.set_title('Predicted drone pos'); ax.set_xlabel('t [s]'); ax.set_ylabel('m')
    ax.legend(fontsize=7); ax.grid(True, lw=0.3)

    # 8. Quaternion norm
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(t_ax, qnorms, 'k-o', ms=3)
    ax.axhline(1.0, color='r', lw=0.7, ls='--', label='ideal=1')
    ax.set_title('Quat norm along horizon'); ax.set_xlabel('t [s]')
    ax.legend(fontsize=7); ax.grid(True, lw=0.3)

    out = os.path.join(os.path.dirname(__file__), 'mpc_single_step_debug.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved → {out}')
    plt.show()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rebuild', action='store_true')
    p.add_argument('--tc',      action='store_true', help='Enable terminal constraint')
    args = p.parse_args()
    main(rebuild=args.rebuild, enable_tc=args.tc)
