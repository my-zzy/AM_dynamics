"""MPC reach test: hover → follow trajectory → hold at target.

A minimal, self-contained test for the MPC controller that skips takeoff,
gripping, and transport phases. The drone starts stationary at HOVER_START,
the MPC drives the EE along a minimum-jerk trajectory to EE_TARGET, then
holds that position.

Run from workspace root:
    conda activate mjc
    python demo/mpc_reach_test.py

Flags:
    --rebuild           Force recompile of acados C code
    --no-viewer         Run headless
    --no-tc             Disable hard terminal constraint
    --dt-mpc=0.05       MPC shooting step (s)
    --horizon=20        MPC prediction horizon steps
    --traj-dur=4.0      Minimum-jerk trajectory duration (s)
    --hold-dur=5.0      How long to hold at target after arrival (s)
    --timeout=10.0      Max time allowed for MPC reach phase (s)
"""

import sys
import os
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'basic'))

from test_model import load_model, apply_platform_control
from pid_controller import DroneController
from ams.model import AerialManipulatorModel
from ams.kinematics import forward_kinematics
from demo.mpc_trajectory import EETrajectory
from demo.mpc_controller import MPCController

# Actuator indices in am_robot.xml
CTRL_J1 = 0
CTRL_J2 = 1


def get_am_state(mj_model, mj_data):
    """Return state dict from am_robot.xml (no box, no gripper joints)."""
    p = mj_data.qpos[0:3].copy()
    v = mj_data.qvel[0:3].copy()
    q_wxyz = mj_data.qpos[3:7]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    omega = mj_data.qvel[3:6].copy()
    theta = mj_data.qpos[7:9].copy()
    theta_dot = mj_data.qvel[6:8].copy()
    return {
        'pos': p, 'vel': v, 'quat': q_xyzw, 'omega': omega,
        'theta': theta, 'theta_dot': theta_dot,
    }

# ---------------------------------------------------------------------------
# Config — edit these to match your scene
# ---------------------------------------------------------------------------
HOVER_START  = np.array([0.25, 0.0, 1.45])   # drone base hover setpoint
EE_TARGET    = np.array([0.40, 0.0, 1.125])   # box / grasp target (world frame)

EE_POS_TOL   = 0.015    # 15 mm — "arrived" threshold
EE_VEL_TOL   = 0.05     # 5 cm/s

PID_GAINS = dict(
    kp_z=10.0, ki_z=2.0, kd_z=6.0,
    kp_xy=5.5, kd_xy=1.0,
    kp_rp=10.0, ki_rp=4.5, kd_rp=2.5,
    kp_yaw=5.0, ki_yaw=0.2, kd_yaw=1.5,
)
KP_ARM, KD_ARM = 8.0, 0.8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pack_state(st):
    return np.concatenate([
        st['pos'], st['vel'], st['quat'], st['omega'],
        st['theta'], st['theta_dot'],
    ])


def ee_world_pos(am_model, st):
    _, p, _ = forward_kinematics(am_model, st['quat'], st['pos'], st['theta'])
    return p[3]


def apply_mpc_u(mj_model, mj_data, u0):
    base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'base')
    mj_data.xfrc_applied[base_id, :] = np.concatenate([u0[0:3], u0[3:6]])
    mj_data.ctrl[CTRL_J1] = float(np.clip(u0[6], -5.0, 5.0))
    mj_data.ctrl[CTRL_J2] = float(np.clip(u0[7], -5.0, 5.0))


def clear_mpc_u(mj_model, mj_data):
    base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'base')
    mj_data.xfrc_applied[base_id, :] = 0.0


def apply_arm_pd(mj_data, theta_des, theta_cur, theta_dot, bias=None):
    tau = KP_ARM * (theta_des - theta_cur) - KD_ARM * theta_dot
    if bias is not None:
        tau += bias
    mj_data.ctrl[CTRL_J1] = float(np.clip(tau[0], -5.0, 5.0))
    mj_data.ctrl[CTRL_J2] = float(np.clip(tau[1], -5.0, 5.0))

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run(dt_mpc=0.05, N=20, traj_dur=4.0, hold_dur=5.0, timeout=10.0,
        rebuild=False, enable_tc=True, use_viewer=True):

    mj_model, mj_data = load_model()
    am_model = AerialManipulatorModel()

    # Robot mass (exclude world body only)
    _excl = {'world'}
    robot_ids = [i for i in range(mj_model.nbody)
                 if mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) not in _excl]
    total_mass = sum(mj_model.body_mass[i] for i in robot_ids)
    print(f'Robot mass : {total_mass:.3f} kg  (mg = {total_mass*9.81:.2f} N)')

    pid = DroneController(mass=total_mass, **PID_GAINS)

    sim_dt   = mj_model.opt.timestep
    steps_per_solve = max(1, int(round(dt_mpc / sim_dt)))

    # Precompute joint DOF indices for gravity FF
    _j1_dof = mj_model.jnt_dofadr[
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'joint1')]
    _j2_dof = mj_model.jnt_dofadr[
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'joint2')]

    # Place drone at hover height
    mujoco.mj_resetData(mj_model, mj_data)
    base_jnt_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'free_joint')
    qp_start = mj_model.jnt_qposadr[base_jnt_id]
    mj_data.qpos[qp_start:qp_start+3] = HOVER_START
    mj_data.qpos[qp_start+3] = 1.0   # quaternion w = 1
    mujoco.mj_forward(mj_model, mj_data)
    pid.reset()

    # ----------------------------------------------------------------
    # Phase 0: PID stabilise at hover (3 s)
    # Phase 1: MPC reach
    # Phase 2: MPC hold at target
    # ----------------------------------------------------------------
    STAB_DUR = 3.0

    # Determine EE start after stabilisation (use FK from nominal state)
    st_nominal = get_am_state(mj_model, mj_data)
    p_ee_start = ee_world_pos(am_model, st_nominal)
    print(f'EE start (FK) : {p_ee_start}')
    print(f'EE target     : {EE_TARGET}')

    # Build MPC objects
    traj = EETrajectory(p_start=p_ee_start, p_end=EE_TARGET,
                        T_f=traj_dur, dt=dt_mpc)

    print(f'\nBuilding MPCController (may compile acados C code)...')
    t_b = time.perf_counter()
    mpc = MPCController(traj=traj, model=am_model, N=N, dt=dt_mpc,
                        rebuild=rebuild, enable_terminal_constraint=enable_tc)
    print(f'MPCController ready in {time.perf_counter()-t_b:.1f} s\n')

    # ----------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------
    log = {k: [] for k in ('t', 'phase',
                            'ee_x', 'ee_z',
                            'ref_x', 'ref_z',
                            'drone_x', 'drone_z',
                            'ee_err', 'solve_ms',
                            'th1', 'th2', 'th1d', 'th2d')}

    # ----------------------------------------------------------------
    # Sim loop
    # ----------------------------------------------------------------
    phase       = 0          # 0=stabilise, 1=MPC reach, 2=MPC hold
    step_idx    = 0
    mpc_step    = 0          # counts sim steps inside MPC phases
    mpc_t0_sim  = 0.0        # sim time when MPC reach started
    last_u0     = None
    arrived     = False
    hold_t0     = 0.0
    theta_des   = np.zeros(2)

    total_time = STAB_DUR + timeout + hold_dur + 1.0

    def loop_body(viewer):
        nonlocal phase, step_idx, mpc_step, mpc_t0_sim
        nonlocal last_u0, arrived, hold_t0, theta_des

        t0_wall = time.perf_counter()
        t0_sim  = mj_data.time

        while mj_data.time - t0_sim < total_time:
            if viewer is not None and not viewer.is_running():
                break

            t   = mj_data.time - t0_sim
            st  = get_am_state(mj_model, mj_data)
            p_ee = ee_world_pos(am_model, st)
            p_ref, _ = traj.get_ref(max(0.0, t - STAB_DUR))
            bias = np.array([mj_data.qfrc_bias[_j1_dof],
                             mj_data.qfrc_bias[_j2_dof]])

            # ── Phase transitions ──────────────────────────────────
            if phase == 0 and t >= STAB_DUR:
                phase = 1
                mpc_t0_sim = t
                mpc.reset()
                print(f'[t={t:.2f}s] Phase 1: MPC REACH  '
                      f'(traj_dur={traj_dur:.1f}s, timeout={timeout:.1f}s)')

            if phase == 1:
                t_mpc  = t - mpc_t0_sim
                ee_err = np.linalg.norm(p_ee - EE_TARGET)
                done   = (ee_err < EE_POS_TOL
                          and np.linalg.norm(st['vel']) < EE_VEL_TOL)
                timed  = t_mpc >= timeout
                if done or timed:
                    phase    = 2
                    arrived  = done
                    hold_t0  = t
                    reason   = 'ARRIVED' if done else 'TIMEOUT'
                    print(f'[t={t:.2f}s] Phase 2: MPC HOLD  '
                          f'({reason}  ee_err={ee_err*1000:.1f} mm)')

            if phase == 2 and (t - hold_t0) >= hold_dur:
                print(f'[t={t:.2f}s] Hold complete. Stopping.')
                break

            # ── Control ───────────────────────────────────────────
            if phase == 0:
                # PID stabilise — keep arm folded
                clear_mpc_u(mj_model, mj_data)
                T, tau_b = pid.compute(st['pos'], st['vel'], st['quat'],
                                       st['omega'], HOVER_START, 0.0, sim_dt)
                apply_platform_control(mj_data, T, tau_b)
                apply_arm_pd(mj_data, theta_des, st['theta'],
                             st['theta_dot'], bias=bias)

            elif phase in (1, 2):
                # MPC — solve at dt_mpc rate
                t_mpc = t - mpc_t0_sim
                t_query = min(t_mpc, traj.T_f)   # clamp in hold phase

                if mpc_step % steps_per_solve == 0:
                    x_now = pack_state(st)
                    try:
                        u0_new, info = mpc.solve(x_now, t_query)
                        last_u0 = u0_new
                        # Log solve time every 20 solves
                        if mpc_step % (steps_per_solve * 20) == 0:
                            ee_err = np.linalg.norm(p_ee - EE_TARGET)
                            print(f'  t={t:.2f}s  ee_err={ee_err*1000:.1f} mm  '
                                  f'solve={info["solve_time"]*1000:.2f} ms  '
                                  f'status={info["status"]}')
                        log['solve_ms'].append(info['solve_time'] * 1000)
                    except Exception as exc:
                        print(f'  MPC solve error at t={t:.2f}s: {exc}')
                mpc_step += 1

                if last_u0 is not None:
                    apply_mpc_u(mj_model, mj_data, last_u0)
                else:
                    # Gravity fallback on very first step
                    u_fb = np.zeros(8)
                    u_fb[2] = am_model.total_mass * 9.81
                    apply_mpc_u(mj_model, mj_data, u_fb)

            # ── Log ───────────────────────────────────────────────
            log['t'].append(t)
            log['phase'].append(phase)
            log['ee_x'].append(p_ee[0])
            log['ee_z'].append(p_ee[2])
            log['ref_x'].append(p_ref[0])
            log['ref_z'].append(p_ref[2])
            log['drone_x'].append(st['pos'][0])
            log['drone_z'].append(st['pos'][2])
            log['ee_err'].append(np.linalg.norm(p_ee - EE_TARGET) * 1000)
            log['th1'].append(np.rad2deg(st['theta'][0]))
            log['th2'].append(np.rad2deg(st['theta'][1]))
            log['th1d'].append(np.rad2deg(theta_des[0]))
            log['th2d'].append(np.rad2deg(theta_des[1]))

            # ── Step ──────────────────────────────────────────────
            mujoco.mj_step(mj_model, mj_data)
            step_idx += 1

            # Real-time sync
            if viewer is not None:
                sim_elapsed  = mj_data.time - t0_sim
                wall_elapsed = time.perf_counter() - t0_wall
                if sim_elapsed > wall_elapsed:
                    time.sleep(sim_elapsed - wall_elapsed)
                viewer.sync()

        # Final stats
        st_f   = get_am_state(mj_model, mj_data)
        p_ee_f = ee_world_pos(am_model, st_f)
        print(f'\n=== Done ===')
        print(f'Final EE position : {p_ee_f}')
        print(f'Final EE error    : {np.linalg.norm(p_ee_f - EE_TARGET)*1000:.1f} mm')
        if log['solve_ms']:
            sms = np.array(log['solve_ms'])
            print(f'MPC solve time    : mean={sms.mean():.2f} ms  max={sms.max():.2f} ms')
        if viewer is not None:
            print('Close viewer to exit.')
            while viewer.is_running():
                time.sleep(0.05)

    if use_viewer:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            viewer.sync()
            loop_body(viewer)
    else:
        loop_body(None)

    plot_results(log, EE_TARGET, traj)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(log, ee_target, traj):
    if not log['t']:
        return

    t   = np.array(log['t'])
    ph  = np.array(log['phase'])

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f'MPC Reach Test   target={np.round(ee_target,3)}\n'
        f'traj={traj.T_f:.1f} s   N={traj.N}   dt={traj.dt*1000:.0f} ms',
        fontsize=11)
    gs = gridspec.GridSpec(3, 2, hspace=0.50, wspace=0.35)

    # ── xz spatial trajectory ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[:, 0])
    for ph_id, col, lbl in [(0, 'gray', 'stabilise'), (1, 'tab:blue', 'MPC reach'),
                             (2, 'tab:green', 'MPC hold')]:
        m = ph == ph_id
        if not np.any(m):
            continue
        ax.plot(np.array(log['ee_x'])[m], np.array(log['ee_z'])[m],
                color=col, lw=1.8, label=f'EE ({lbl})')
    ax.plot(np.array(log['ref_x']), np.array(log['ref_z']),
            'k--', lw=1.0, label='Traj reference')
    ax.scatter(np.array(log['ee_x'])[0],  np.array(log['ee_z'])[0],
               color='gray', s=80, marker='o', zorder=6, label='start')
    ax.scatter(ee_target[0], ee_target[2], color='tab:red', s=160,
               marker='*', zorder=7, label='target')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title('xz EE trajectory')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)

    # ── EE tracking error ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, log['ee_err'], color='tab:blue', lw=1.3)
    ax.axhline(15.0, color='tab:red', lw=0.8, ls='--', label='15 mm threshold')
    ax.set_ylabel('EE error [mm]')
    ax.set_title('EE position error to target')
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)
    # Shade phases
    _shade_phases(ax, t, ph)

    # ── EE x and z vs time ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, log['ee_x'],  color='tab:blue',   lw=1.3, label='EE x')
    ax.plot(t, log['ref_x'], color='tab:blue',   lw=0.8, ls='--', label='ref x')
    ax.plot(t, log['ee_z'],  color='tab:orange', lw=1.3, label='EE z')
    ax.plot(t, log['ref_z'], color='tab:orange', lw=0.8, ls='--', label='ref z')
    ax.axhline(ee_target[0], color='tab:blue',   lw=0.6, ls=':')
    ax.axhline(ee_target[2], color='tab:orange', lw=0.6, ls=':')
    ax.set_ylabel('position [m]')
    ax.set_title('EE x/z vs time  (dashed=reference  dotted=target)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, lw=0.4)
    _shade_phases(ax, t, ph)

    # ── Joint angles ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t, log['th1'], color='tab:blue',   lw=1.3, label='j1 actual')
    ax.plot(t, log['th2'], color='tab:orange', lw=1.3, label='j2 actual')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('deg')
    ax.set_title('Joint angles')
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)
    _shade_phases(ax, t, ph)

    out = os.path.join(os.path.dirname(__file__), 'mpc_reach_test.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Plot saved → {out}')
    try:
        plt.show()
    except Exception:
        pass


def _shade_phases(ax, t, ph):
    """Shade time axis by phase (light background colours)."""
    colours = {0: '#e0e0e0', 1: '#cce5ff', 2: '#ccffcc'}
    for ph_id, col in colours.items():
        m = ph == ph_id
        if not np.any(m):
            continue
        ax.axvspan(t[m][0], t[m][-1], alpha=0.25, color=col, zorder=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description='MPC Reach Test')
    p.add_argument('--rebuild',    action='store_true', help='Force acados recompile')
    p.add_argument('--no-viewer',  action='store_true', help='Run headless')
    p.add_argument('--no-tc',      action='store_true', help='Disable terminal constraint')
    p.add_argument('--dt-mpc',     type=float, default=0.05,  metavar='S')
    p.add_argument('--horizon',    type=int,   default=20,    metavar='N')
    p.add_argument('--traj-dur',   type=float, default=4.0,   metavar='S')
    p.add_argument('--hold-dur',   type=float, default=5.0,   metavar='S')
    p.add_argument('--timeout',    type=float, default=10.0,  metavar='S')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse()
    print('=== MPC Reach Test ===')
    run(
        dt_mpc=args.dt_mpc,
        N=args.horizon,
        traj_dur=args.traj_dur,
        hold_dur=args.hold_dur,
        timeout=args.timeout,
        rebuild=args.rebuild,
        enable_tc=not args.no_tc,
        use_viewer=not args.no_viewer,
    )
