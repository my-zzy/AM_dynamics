"""Interactive step-by-step MPC debugger.

At each MPC step the script:
  1. Solves the OCP from the current state and plots the predicted trajectory.
  2. Pauses — inspect the plot, then **close the window** to continue.
  3. Steps the MuJoCo simulation forward by dt_mpc seconds using the computed u0.
  4. Reads back the real state from MuJoCo and repeats.

This lets you walk through the trajectory one MPC decision at a time,
comparing predicted vs. actual motion and inspecting any undesired behaviour.

Run from workspace root:
    conda activate gz
    python demo/mpc_step_debug.py

Flags:
    --rebuild            Force recompile of acados C code
    --no-tc              Disable hard terminal constraint
    --dt-mpc=0.05        MPC shooting step (s)
    --horizon=20         MPC prediction horizon steps
    --traj-dur=4.0       Trajectory duration (s)
    --max-steps=999      Maximum MPC steps before stopping
    --no-viewer          Skip the passive MuJoCo viewer window
"""

import sys
import os
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'basic'))

from test_model import load_model
from ams.model import AerialManipulatorModel
from ams.kinematics import forward_kinematics
from demo.mpc_trajectory import EETrajectory
from demo.mpc_controller import MPCController, _W_STAGE_DIAG, _W_RATE_DIAG, _U_MIN, _U_MAX

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOVER_START = np.array([0.0, 0.0, 1.0])
EE_TARGET   = np.array([0.40, 0.0, 1.125])

CTRL_J1, CTRL_J2 = 0, 1

# ---------------------------------------------------------------------------
# Helpers (shared with mpc_reach_test.py)
# ---------------------------------------------------------------------------

def get_am_state(mj_model, mj_data):
    p  = mj_data.qpos[0:3].copy()
    v  = mj_data.qvel[0:3].copy()
    q_wxyz = mj_data.qpos[3:7]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    omega  = mj_data.qvel[3:6].copy()
    theta  = mj_data.qpos[7:9].copy()
    theta_dot = mj_data.qvel[6:8].copy()
    return {'pos': p, 'vel': v, 'quat': q_xyzw, 'omega': omega,
            'theta': theta, 'theta_dot': theta_dot}


def pack_state(st):
    q = st['quat'].copy()
    q_norm = np.linalg.norm(q)
    if q_norm > 1e-8:
        q /= q_norm
    return np.concatenate([st['pos'], st['vel'], q, st['omega'],
                           st['theta'], st['theta_dot']])


def ee_world_pos(am_model, st):
    _, p, _ = forward_kinematics(am_model, st['quat'], st['pos'], st['theta'])
    return p[3]


def apply_mpc_u(mj_model, mj_data, u0):
    base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'base')
    mj_data.xfrc_applied[base_id, :] = np.concatenate([u0[0:3], u0[3:6]])
    mj_data.ctrl[CTRL_J1] = float(np.clip(u0[6], -0.5, 0.5))
    mj_data.ctrl[CTRL_J2] = float(np.clip(u0[7], -0.5, 0.5))


def clear_mpc_u(mj_model, mj_data):
    base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'base')
    mj_data.xfrc_applied[base_id, :] = 0.0


# ---------------------------------------------------------------------------
# Step plot
# ---------------------------------------------------------------------------

def plot_step(step_idx, t_mpc, mpc, traj, am_model,
              x_current, u0, p_ee_current,
              history_t, history_ee, history_ref,
              info):
    """Full diagnostic plot matching mpc_single_step_debug.py, plus history overlays.
    Blocks until the user closes the window."""

    N  = mpc.N
    dt = mpc.dt

    # ── Extract predicted states and inputs from solver ──────────────────────
    xs = np.array([mpc._solver.get(k, 'x') for k in range(N + 1)])  # (N+1, 17)
    us = np.array([mpc._solver.get(k, 'u') for k in range(N)])      # (N,   8)

    pred_ee = []
    for k in range(N + 1):
        p_A = xs[k, 0:3]; q_A = xs[k, 6:10]; th = xs[k, 13:15]
        try:
            _, p_fk, _ = forward_kinematics(am_model, q_A, p_A, th)
            pred_ee.append(p_fk[3].copy())
        except Exception:
            pred_ee.append(p_A.copy())
    pred_ee = np.array(pred_ee)           # (N+1, 3)

    t_ax  = t_mpc + np.arange(N + 1) * dt
    t_u   = t_mpc + np.arange(N) * dt
    ref_ee = np.array([traj._p(t) for t in t_ax])   # (N+1, 3)

    # ── Per-stage cost breakdown ─────────────────────────────────────────────
    W  = _W_STAGE_DIAG   # (17,)  [p_EE(3), v_A(3), omega(3), u(8)]
    Wr = _W_RATE_DIAG    # (8,)   [delta_u]
    hover_f = am_model.total_mass * 9.81
    stage_ee_cost   = np.zeros(N)
    stage_vel_cost  = np.zeros(N)
    stage_omg_cost  = np.zeros(N)
    stage_u_cost    = np.zeros(N)
    stage_rate_cost = np.zeros(N)
    for k in range(N):
        r_ee  = pred_ee[k] - ref_ee[k]
        r_vel = xs[k, 3:6]
        r_omg = xs[k, 10:13]
        r_u   = us[k].copy(); r_u[2] -= hover_f
        u_prev = us[k - 1] if k > 0 else us[0]   # Δu relative to previous
        r_du  = us[k] - u_prev
        stage_ee_cost[k]   = 0.5 * r_ee  @ (W[0:3]  * r_ee)
        stage_vel_cost[k]  = 0.5 * r_vel @ (W[3:6]  * r_vel)
        stage_omg_cost[k]  = 0.5 * r_omg @ (W[6:9]  * r_omg)
        stage_u_cost[k]    = 0.5 * (r_u[0:3] @ (W[9:12]  * r_u[0:3])
                                     + r_u[3:6] @ (W[12:15] * r_u[3:6])
                                     + r_u[6:8] @ (W[15:17] * r_u[6:8]))
        stage_rate_cost[k] = 0.5 * r_du @ (Wr * r_du)

    # ── Quaternion norms ─────────────────────────────────────────────────────
    qnorms = np.array([np.linalg.norm(xs[k, 6:10]) for k in range(N + 1)])

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f'MPC Step {step_idx}   t_traj={t_mpc:.3f} s   '
        f'status={info["status"]}   cost={info["cost"]:.2f}   '
        f'solve={info["solve_time"]*1000:.1f} ms\n'
        f'EE error to target = {np.linalg.norm(p_ee_current - EE_TARGET)*1000:.1f} mm   '
        f'(close window to advance)',
        fontsize=11)
    gs = gridspec.GridSpec(3, 3, hspace=0.55, wspace=0.40)

    hee = np.array(history_ee) if len(history_ee) > 1 else None

    # 1. EE x vs time
    ax = fig.add_subplot(gs[0, 0])
    if hee is not None:
        ax.plot(history_t, hee[:, 0], 'tab:gray', lw=1.3, label='actual')
    ax.plot(t_ax, pred_ee[:, 0], 'b-o', ms=3, label='predicted')
    ax.plot(t_ax, ref_ee[:, 0],  'r--',        label='reference')
    ax.axhline(EE_TARGET[0], color='r', lw=0.7, ls=':')
    ax.set_title('EE x'); ax.set_xlabel('t [s]'); ax.set_ylabel('m')
    ax.legend(fontsize=7); ax.grid(True, lw=0.3)

    # 2. EE z vs time
    ax = fig.add_subplot(gs[0, 1])
    if hee is not None:
        ax.plot(history_t, hee[:, 2], 'tab:gray', lw=1.3, label='actual')
    ax.plot(t_ax, pred_ee[:, 2], 'b-o', ms=3, label='predicted')
    ax.plot(t_ax, ref_ee[:, 2],  'r--',        label='reference')
    ax.axhline(EE_TARGET[2], color='r', lw=0.7, ls=':')
    ax.set_title('EE z'); ax.set_xlabel('t [s]'); ax.set_ylabel('m')
    ax.legend(fontsize=7); ax.grid(True, lw=0.3)

    # 3. xz spatial
    ax = fig.add_subplot(gs[0, 2])
    if hee is not None:
        ax.plot(hee[:, 0], hee[:, 2], 'tab:gray', lw=1.3, label='actual')
    ax.plot(pred_ee[:, 0], pred_ee[:, 2], 'b-o', ms=3, label='predicted')
    ax.plot(ref_ee[:, 0],  ref_ee[:, 2],  'r--',        label='reference')
    ax.scatter(p_ee_current[0], p_ee_current[2], c='tab:blue', s=80, zorder=5, label='current')
    ax.scatter(EE_TARGET[0], EE_TARGET[2], c='red', s=160, marker='*', zorder=6, label='target')
    ax.set_title('xz EE'); ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]')
    ax.set_aspect('equal'); ax.legend(fontsize=7); ax.grid(True, lw=0.3)

    # 4. Per-stage cost stacked bar (spans 2 columns)
    ax = fig.add_subplot(gs[1, 0:2])
    kk = np.arange(N)
    ax.bar(kk, stage_ee_cost,   label='EE pos',   color='tab:blue')
    ax.bar(kk, stage_vel_cost,  bottom=stage_ee_cost, label='vel', color='tab:orange')
    bot2 = stage_ee_cost + stage_vel_cost
    ax.bar(kk, stage_omg_cost,  bottom=bot2,       label='omega',  color='tab:green')
    bot3 = bot2 + stage_omg_cost
    ax.bar(kk, stage_u_cost,    bottom=bot3,       label='effort', color='tab:red')
    bot4 = bot3 + stage_u_cost
    ax.bar(kk, stage_rate_cost, bottom=bot4,       label='rate',   color='tab:purple')
    ax.set_title('Per-stage cost breakdown')
    ax.set_xlabel('stage k'); ax.set_ylabel('cost')
    ax.legend(fontsize=7); ax.grid(True, lw=0.3, axis='y')

    # 5. Inputs: F_z and joint torques
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t_u, us[:, 2], 'b-o', ms=3, label='F_z')
    ax.axhline(hover_f, color='b', lw=0.7, ls=':', label=f'mg={hover_f:.1f}N')
    ax2 = ax.twinx()
    ax2.plot(t_u, us[:, 6], 'g-o', ms=3, label='τ_j1')
    ax2.plot(t_u, us[:, 7], 'm-o', ms=3, label='τ_j2')
    ax2.set_ylabel('joint torque [Nm]', color='g')
    ax.set_title('Inputs'); ax.set_xlabel('t [s]'); ax.set_ylabel('F_z [N]')
    ax.legend(fontsize=7, loc='upper left')
    ax2.legend(fontsize=7, loc='upper right')
    ax.grid(True, lw=0.3)

    # 6. Joint angles along predicted horizon
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t_ax, np.rad2deg(xs[:, 13]), 'b-o', ms=3, label='θ₁')
    ax.plot(t_ax, np.rad2deg(xs[:, 14]), 'r-o', ms=3, label='θ₂')
    ax.axhline( 20, color='k', lw=0.7, ls='--')
    ax.axhline(-20, color='k', lw=0.7, ls='--')
    ax.set_title('Predicted joint angles'); ax.set_xlabel('t [s]'); ax.set_ylabel('deg')
    ax.legend(fontsize=7); ax.grid(True, lw=0.3)

    # 7. Platform position along horizon
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t_ax, xs[:, 0], label='x')
    ax.plot(t_ax, xs[:, 1], label='y')
    ax.plot(t_ax, xs[:, 2], label='z')
    ax.set_title('Predicted drone pos'); ax.set_xlabel('t [s]'); ax.set_ylabel('m')
    ax.legend(fontsize=7); ax.grid(True, lw=0.3)

    # 8. Quaternion norm along horizon
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(t_ax, qnorms, 'k-o', ms=3)
    ax.axhline(1.0, color='r', lw=0.7, ls='--', label='ideal=1')
    ax.set_title('Quat norm along horizon'); ax.set_xlabel('t [s]')
    ax.legend(fontsize=7); ax.grid(True, lw=0.3)

    plt.show()   # blocks until user closes window


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dt_mpc=0.05, N=20, traj_dur=4.0, max_steps=999,
        rebuild=False, enable_tc=True, use_viewer=True):

    mj_model, mj_data = load_model()
    am_model = AerialManipulatorModel()

    # Robot mass
    _excl = {'world'}
    robot_ids = [i for i in range(mj_model.nbody)
                 if mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) not in _excl]
    total_mass = sum(mj_model.body_mass[i] for i in robot_ids)
    print(f'Robot mass : {total_mass:.3f} kg  (mg = {total_mass*9.81:.2f} N)')

    sim_dt = mj_model.opt.timestep
    steps_per_mpc = max(1, int(round(dt_mpc / sim_dt)))

    # ── Teleport drone to hover state (zero velocity) ─────────────────────
    mujoco.mj_resetData(mj_model, mj_data)
    base_jnt_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'free_joint')
    qp_start = mj_model.jnt_qposadr[base_jnt_id]
    mj_data.qpos[qp_start:qp_start+3] = HOVER_START
    mj_data.qpos[qp_start+3] = 1.0   # quaternion w=1 (upright)
    # all velocities already zero from resetData
    mujoco.mj_forward(mj_model, mj_data)

    # EE start position (FK from nominal)
    st0 = get_am_state(mj_model, mj_data)
    p_ee_start = ee_world_pos(am_model, st0)
    print(f'EE start (FK) : {p_ee_start}')
    print(f'EE target     : {EE_TARGET}')

    # Build trajectory and controller
    traj = EETrajectory(p_start=p_ee_start, p_end=EE_TARGET,
                        T_f=traj_dur, dt=dt_mpc)
    print('\nBuilding MPCController (may compile acados C code)...')
    mpc = MPCController(traj=traj, model=am_model, N=N, dt=dt_mpc,
                        rebuild=rebuild, enable_terminal_constraint=enable_tc)
    print('MPCController ready.\n')

    # Launch viewer (shows drone already at hover position)
    if use_viewer:
        viewer_ctx = mujoco.viewer.launch_passive(mj_model, mj_data)
        viewer_ctx.__enter__()
        viewer_ctx.sync()
    else:
        viewer_ctx = None

    print('Starting MPC step-by-step loop.\n'
          'Close each plot to advance to the next step.\n'
          'Press Ctrl-C to abort.\n')

    # ----------------------------------------------------------------
    # MPC step-by-step loop
    history_t   = []   # trajectory time at each MPC step
    history_ee  = []   # actual EE position at each MPC step

    mpc_step = 0
    t_mpc    = 0.0     # elapsed time within the MPC trajectory

    try:
        while mpc_step < max_steps:
            # ── Read current state ────────────────────────────────
            st      = get_am_state(mj_model, mj_data)
            x_now   = pack_state(st)
            p_ee    = ee_world_pos(am_model, st)
            ee_err  = np.linalg.norm(p_ee - EE_TARGET)

            t_query = float(np.clip(t_mpc, 0.0, traj.T_f))

            print(f'Step {mpc_step:4d}  t_traj={t_mpc:.3f}s  '
                  f'EE={np.round(p_ee,4)}  err={ee_err*1000:.1f} mm', end='  ')

            # ── Solve OCP ─────────────────────────────────────────
            u0, info = mpc.solve(x_now, t_query)
            print(f'solve={info["solve_time"]*1000:.1f} ms  status={info["status"]}')

            # ── Check if any input hit its bound ──────────────────
            _labels = ['Fx', 'Fy', 'Fz', 'τx', 'τy', 'τz', 'τj1', 'τj2']
            _tol = 1e-3
            for _i, (_lb, _ub, _v, _lbl) in enumerate(
                    zip(_U_MIN, _U_MAX, u0, _labels)):
                if _i in (0, 1):   # Fx, Fy are fixed to zero — skip
                    continue
                if _v <= _lb + _tol:
                    print(f'  [BOUND] {_lbl} hit LOWER bound ({_v:.4f} ≈ {_lb})')
                elif _v >= _ub - _tol:
                    print(f'  [BOUND] {_lbl} hit UPPER bound ({_v:.4f} ≈ {_ub})')

            # ── Record history ────────────────────────────────────
            history_t.append(t_mpc)
            history_ee.append(p_ee.copy())

            # ── Plot and block until user closes window ───────────
            plot_step(
                step_idx=mpc_step,
                t_mpc=t_mpc,
                mpc=mpc,
                traj=traj,
                am_model=am_model,
                x_current=x_now,
                u0=u0,
                p_ee_current=p_ee,
                history_t=list(history_t),
                history_ee=list(history_ee),
                history_ref=[traj._p(t) for t in history_t],
                info=info,
            )

            # ── Simulate dt_mpc seconds forward in MuJoCo ────────
            for _ in range(steps_per_mpc):
                apply_mpc_u(mj_model, mj_data, u0)
                mujoco.mj_step(mj_model, mj_data)
                if viewer_ctx is not None:
                    viewer_ctx.sync()

            t_mpc    += dt_mpc
            mpc_step += 1

            # ── Check termination ──────────────────────────────────
            if ee_err < 0.015 and np.linalg.norm(st['vel']) < 0.05:
                print('\n=== EE reached target. Stopping. ===')
                break
            if t_mpc >= traj.T_f + 2.0:
                print('\n=== Trajectory time exceeded. Stopping. ===')
                break

    except KeyboardInterrupt:
        print('\nAborted by user.')

    finally:
        if viewer_ctx is not None:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception:
                pass

    # Final state
    st_f   = get_am_state(mj_model, mj_data)
    p_ee_f = ee_world_pos(am_model, st_f)
    print(f'\n=== Done after {mpc_step} MPC steps ===')
    print(f'Final EE position : {np.round(p_ee_f, 4)}')
    print(f'Final EE error    : {np.linalg.norm(p_ee_f - EE_TARGET)*1000:.1f} mm')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description='MPC Step-by-Step Debugger')
    p.add_argument('--rebuild',     action='store_true', help='Force acados recompile')
    p.add_argument('--no-tc',       action='store_true', help='Disable terminal constraint')
    p.add_argument('--no-viewer',   action='store_true', help='No MuJoCo viewer window')
    p.add_argument('--dt-mpc',      type=float, default=0.05,  metavar='S')
    p.add_argument('--horizon',     type=int,   default=20,    metavar='N')
    p.add_argument('--traj-dur',    type=float, default=4.0,   metavar='S')
    p.add_argument('--max-steps',   type=int,   default=999,   metavar='K')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse()
    print('=== MPC Step-by-Step Debugger ===')
    run(
        dt_mpc=args.dt_mpc,
        N=args.horizon,
        traj_dur=args.traj_dur,
        max_steps=args.max_steps,
        rebuild=args.rebuild,
        enable_tc=not args.no_tc,
        use_viewer=not args.no_viewer,
    )
