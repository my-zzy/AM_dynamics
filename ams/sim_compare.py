"""Simulator validation: ams (RK4) vs MuJoCo (RK4).

Mirrors the five experiments in simulator.py __main__:
  1. Hover hold          – default arm config, gravity-compensating wrench, 1 s
  2. Free fall           – zero input, 0.5 s
  3. Hover arm at 90°    – theta1=pi/2, gravity-compensating wrench, 1 s
  4. Quaternion norm     – checks ams quat norm stays at 1.0 (from exp 1)
  5. Coast (no gravity)  – constant velocity, zero input, 1 s

Differences neutralised before comparing
-----------------------------------------
  • Joint damping  : XML default damping="0.02" is zeroed for joint1/joint2.
  • Integrator     : Both are set to RK4.
  • Gripper joints : Locked with high damping; ams treats EE as fixed body.

Run from workspace root (conda activate main):
    python ams/sim_compare.py
"""
import sys, os
import time
import numpy as np
import mujoco
import mujoco.viewer

_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'basic'))

from ams.model import AerialManipulatorModel
from ams.simulator import rk4_step
from ams.state import SystemState, N_JOINTS
from ams.dynamics import inverse_dynamics
from basic.test_model import load_model, get_state, set_state

# ===========================================================================
#  EXPERIMENT PARAMETERS
# ===========================================================================
DT = 0.01   # timestep [s] shared by all experiments (matches simulator.py)


# ===========================================================================
#  HELPERS
# ===========================================================================

def build_initial_state_vector(p, q_xyzw, theta):
    """Build ams state vector from components (all velocities zero)."""
    s = SystemState()
    s.platform.position         = p.copy()
    s.platform.quaternion       = q_xyzw.copy()
    s.manipulator.joint_angles  = theta.copy()
    return s.to_vector()


def quat_to_euler_zyx(q_xyzw):
    """Roll-pitch-yaw from quaternion [x,y,z,w]."""
    x, y, z, w = q_xyzw
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp  = np.clip(2*(w*y - z*x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([roll, pitch, yaw])


def state_to_channels(x):
    """Split ams state vector into named channels dict."""
    return {
        'pos'   : x[0:3].copy(),
        'vel'   : x[3:6].copy(),
        'quat'  : x[6:10].copy(),
        'omega' : x[10:13].copy(),
        'theta' : x[13:15].copy(),
        'thdot' : x[15:17].copy(),
        'rpy'   : quat_to_euler_zyx(x[6:10]),
    }


# ===========================================================================
#  SETUP
# ===========================================================================

def setup_mujoco():
    """Load MuJoCo model and configure for clean comparison."""
    mjm, mjd = load_model()

    # Switch to RK4 to match ams integrator
    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4

    # Zero damping ONLY for joint1 and joint2 (the joints ams models).
    # Do NOT touch gripper joint damping: ams lumps the EE + fingers as a single
    # rigid body (mass=0.22, fixed inertia).  If gripper damping is zeroed the
    # fingers slide freely under inertial/gravity loads, changing the effective
    # inertia seen by joint1/joint2 and causing the joint-angle mismatch.
    j1_dof = mjm.jnt_dofadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, 'joint1')]
    j2_dof = mjm.jnt_dofadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, 'joint2')]
    mjm.dof_damping[j1_dof] = 0.0
    mjm.dof_damping[j2_dof] = 0.0

    # Gripper fingers are locked by zeroing qpos/qvel after every step in
    # run_mujoco (see below).  High damping (e.g. 1e6) is NOT used here:
    # with explicit RK4 and dt=0.01 s the damping time constant m/c ≈ 2.5e-8 s
    # is ~400 000× smaller than dt, making the ODE catastrophically stiff.

    # Timestep
    mjm.opt.timestep = DT

    return mjm, mjd


def setup_ams():
    return AerialManipulatorModel()


# ===========================================================================
#  SIMULATION LOOPS
# ===========================================================================

def run_mujoco(mjm, mjd, u_const, t_end, dt=DT):
    """Simulate in MuJoCo; return (t_hist, x_hist) using ams state convention."""
    F_ext   = u_const[:3]
    tau_ext = u_const[3:6]
    jt      = u_const[6:]

    base_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, 'base')
    mjm.opt.timestep = dt

    # Gripper joint addresses for hard-locking after each step
    gl_id   = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, 'gripper_left_joint')
    gr_id   = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, 'gripper_right_joint')
    gl_qadr = mjm.jnt_qposadr[gl_id]
    gr_qadr = mjm.jnt_qposadr[gr_id]
    gl_vadr = mjm.jnt_dofadr[gl_id]
    gr_vadr = mjm.jnt_dofadr[gr_id]

    steps   = int(round(t_end / dt))
    t_hist  = np.zeros(steps + 1)
    x_hist  = np.zeros((steps + 1, 17))
    x_hist[0] = get_state(mjm, mjd)

    with mujoco.viewer.launch_passive(mjm, mjd) as viewer:
        viewer.sync()
        t0_wall = time.perf_counter()   # start clock AFTER viewer is ready
        for i in range(steps):
            if not viewer.is_running():
                # User closed the window early – fill remainder with last state
                x_hist[i + 1:] = x_hist[i]
                t_hist[i + 1:] = np.arange(i + 1, steps + 1) * dt
                break

            # Apply wrench (world frame, at base CoM)
            mjd.xfrc_applied[base_id, :3] = F_ext
            mjd.xfrc_applied[base_id, 3:] = tau_ext
            # Joint torques via actuators (ctrl[2/3] = gripper, stays 0)
            mjd.ctrl[0] = jt[0]
            mjd.ctrl[1] = jt[1]

            mujoco.mj_step(mjm, mjd)

            # Hard-lock gripper joints: zero position and velocity so their
            # stiffness/damping adds no implicit forces to the next step.
            mjd.qpos[gl_qadr] = 0.0
            mjd.qpos[gr_qadr] = 0.0
            mjd.qvel[gl_vadr] = 0.0
            mjd.qvel[gr_vadr] = 0.0

            t_hist[i + 1] = mjd.time
            x_hist[i + 1] = get_state(mjm, mjd)

            # Real-time pacing (both measured from loop start)
            sim_elapsed  = (i + 1) * dt
            wall_elapsed = time.perf_counter() - t0_wall
            if sim_elapsed > wall_elapsed:
                time.sleep(sim_elapsed - wall_elapsed)

            viewer.sync()

    return t_hist, x_hist


def run_ams(ams_model, x0, u_const, t_end, dt=DT):
    """Simulate in ams RK4; return (t_hist, x_hist)."""
    steps  = int(round(t_end / dt))
    t_hist = np.zeros(steps + 1)
    x_hist = np.zeros((steps + 1, 17))
    x_hist[0] = x0

    x = x0.copy()
    for i in range(steps):
        x = rk4_step(ams_model, x, u_const, dt)
        t_hist[i + 1]  = (i + 1) * dt
        x_hist[i + 1]  = x

    return t_hist, x_hist


# ===========================================================================
#  EXPERIMENT RUNNER
# ===========================================================================

def run_and_compare(name, ams_model, mjm, mjd, x0, u_const, t_end):
    """Run both simulators for one experiment, print error table, return results."""
    print(f'\n{"=" * 60}')
    print(f'  EXPERIMENT: {name}')
    print(f'{"=" * 60}')
    print(f'  Running ams RK4 ...')
    t_ams, x_ams = run_ams(ams_model, x0, u_const, t_end)

    mujoco.mj_resetData(mjm, mjd)
    set_state(mjm, mjd, x0)
    mujoco.mj_forward(mjm, mjd)
    print(f'  Running MuJoCo RK4 ...')
    t_mjc, x_mjc = run_mujoco(mjm, mjd, u_const, t_end)

    print_error_table(x_ams, x_mjc)
    return t_ams, x_ams, x_mjc


# ===========================================================================
#  ERROR ANALYSIS
# ===========================================================================

CHANNEL_SLICES = [
    ('pos_x  [m]  ', 0),
    ('pos_y  [m]  ', 1),
    ('pos_z  [m]  ', 2),
    ('vel_x  [m/s]', 3),
    ('vel_y  [m/s]', 4),
    ('vel_z  [m/s]', 5),
    ('qx           ', 6),
    ('qy           ', 7),
    ('qz           ', 8),
    ('qw           ', 9),
    ('omega_x[r/s]', 10),
    ('omega_y[r/s]', 11),
    ('omega_z[r/s]', 12),
    ('theta1 [rad] ', 13),
    ('theta2 [rad] ', 14),
    ('thdot1[r/s] ', 15),
    ('thdot2[r/s] ', 16),
]


def print_error_table(x_ams, x_mjc):
    diff = x_ams - x_mjc
    w = 52
    print('\n' + '=' * w)
    print('  TRAJECTORY ERROR  (ams RK4  vs  MuJoCo RK4)')
    print('=' * w)
    print(f'  {"Channel":<16}  {"RMSE":>12}  {"Max |err|":>12}')
    print('-' * w)
    for label, idx in CHANNEL_SLICES:
        e    = diff[:, idx]
        rmse = float(np.sqrt(np.mean(e ** 2)))
        maxe = float(np.max(np.abs(e)))
        print(f'  {label:<16}  {rmse:>12.6f}  {maxe:>12.6f}')
    # Overall 3-D position RMSE
    pos_err = np.linalg.norm(x_ams[:, :3] - x_mjc[:, :3], axis=1)
    print('-' * w)
    print(f'  {"3D pos [m]":<16}  {float(np.sqrt(np.mean(pos_err**2))):>12.6f}'
          f'  {float(np.max(pos_err)):>12.6f}')
    print('=' * w)


# ===========================================================================
#  PLOT
# ===========================================================================

def plot_comparison(t, x_ams, x_mjc, title=''):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Convert quaternion → RPY for plotting
    rpy_ams = np.degrees(np.array([quat_to_euler_zyx(x_ams[i, 6:10])
                                    for i in range(len(t))]))
    rpy_mjc = np.degrees(np.array([quat_to_euler_zyx(x_mjc[i, 6:10])
                                    for i in range(len(t))]))

    fig = plt.figure(figsize=(14, 16))
    sup = f'ams vs MuJoCo – {title}' if title else 'Simulator Comparison: ams (RK4) vs MuJoCo (RK4)'
    fig.suptitle(sup, fontsize=13)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    # -- Position --
    ax_pos = fig.add_subplot(gs[0, :])
    labels_p = ['x [m]', 'y [m]', 'z [m]']
    colors   = ['tab:blue', 'tab:orange', 'tab:green']
    for i, (lbl, col) in enumerate(zip(labels_p, colors)):
        ax_pos.plot(t, x_ams[:, i], color=col, lw=1.5,
                    label=f'ams {lbl}')
        ax_pos.plot(t, x_mjc[:, i], color=col, lw=1.0,
                    linestyle='--', label=f'mjc {lbl}')
    ax_pos.set_ylabel('Position [m]')
    ax_pos.set_title('Position')
    ax_pos.legend(ncol=3, fontsize=7)
    ax_pos.grid(True, linewidth=0.4)

    # -- Attitude (RPY) --
    ax_att = fig.add_subplot(gs[1, :])
    labels_a = ['roll [°]', 'pitch [°]', 'yaw [°]']
    for i, (lbl, col) in enumerate(zip(labels_a, colors)):
        ax_att.plot(t, rpy_ams[:, i], color=col, lw=1.5, label=f'ams {lbl}')
        ax_att.plot(t, rpy_mjc[:, i], color=col, lw=1.0,
                    linestyle='--', label=f'mjc {lbl}')
    ax_att.set_ylabel('Angle [°]')
    ax_att.set_title('Attitude (roll / pitch / yaw)')
    ax_att.legend(ncol=3, fontsize=7)
    ax_att.grid(True, linewidth=0.4)

    # -- Joint angles --
    ax_jt = fig.add_subplot(gs[2, 0])
    ax_jt.plot(t, np.degrees(x_ams[:, 13]), 'tab:blue',  lw=1.5, label='ams θ₁')
    ax_jt.plot(t, np.degrees(x_mjc[:, 13]), 'tab:blue',  lw=1.0, ls='--', label='mjc θ₁')
    ax_jt.plot(t, np.degrees(x_ams[:, 14]), 'tab:orange', lw=1.5, label='ams θ₂')
    ax_jt.plot(t, np.degrees(x_mjc[:, 14]), 'tab:orange', lw=1.0, ls='--', label='mjc θ₂')
    ax_jt.set_xlabel('time [s]')
    ax_jt.set_ylabel('Angle [°]')
    ax_jt.set_title('Joint angles')
    ax_jt.legend(fontsize=7)
    ax_jt.grid(True, linewidth=0.4)

    # -- Position error --
    ax_perr = fig.add_subplot(gs[2, 1])
    pos_err = np.linalg.norm(x_ams[:, :3] - x_mjc[:, :3], axis=1)
    ax_perr.semilogy(t, np.maximum(pos_err, 1e-15), color='tab:red', lw=1.2)
    ax_perr.set_xlabel('time [s]')
    ax_perr.set_ylabel('||Δpos|| [m]  (log)')
    ax_perr.set_title('3-D position error')
    ax_perr.grid(True, linewidth=0.4, which='both')

    # -- Per-channel error (velocity + omega) --
    ax_verr = fig.add_subplot(gs[3, 0])
    vel_err = np.linalg.norm(x_ams[:, 3:6] - x_mjc[:, 3:6], axis=1)
    omg_err = np.linalg.norm(x_ams[:, 10:13] - x_mjc[:, 10:13], axis=1)
    ax_verr.semilogy(t, np.maximum(vel_err, 1e-15), color='tab:blue', lw=1.2, label='Δvel')
    ax_verr.semilogy(t, np.maximum(omg_err, 1e-15), color='tab:orange', lw=1.2, label='Δω')
    ax_verr.set_xlabel('time [s]')
    ax_verr.set_ylabel('||Δ·|| (log)')
    ax_verr.set_title('Velocity / ang-vel error')
    ax_verr.legend(fontsize=7)
    ax_verr.grid(True, linewidth=0.4, which='both')

    # -- Joint error --
    ax_jerr = fig.add_subplot(gs[3, 1])
    jt_err = np.abs(x_ams[:, 13:15] - x_mjc[:, 13:15])
    ax_jerr.semilogy(t, np.maximum(jt_err[:, 0], 1e-15), 'tab:blue',   lw=1.2, label='|Δθ₁|')
    ax_jerr.semilogy(t, np.maximum(jt_err[:, 1], 1e-15), 'tab:orange', lw=1.2, label='|Δθ₂|')
    ax_jerr.set_xlabel('time [s]')
    ax_jerr.set_ylabel('|Δθ| [rad]  (log)')
    ax_jerr.set_title('Joint angle error')
    ax_jerr.legend(fontsize=7)
    ax_jerr.grid(True, linewidth=0.4, which='both')

    safe = (title.replace(' ', '_').replace('/', '').replace('°', 'deg')
             .replace('(', '').replace(')', '').replace(',', '').replace('=', ''))
    fname = f'sim_compare_{safe}.png' if safe else 'sim_compare.png'
    out_path = os.path.join(os.path.dirname(__file__), fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved → {out_path}')
    try:
        plt.show()
    except Exception:
        pass


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    ams_model = setup_ams()
    mjm, mjd  = setup_mujoco()

    # ---- Build initial state (hover at z=1 m, arm at zero config) ----
    s0 = SystemState()
    s0.platform.position[:] = [0, 0, 1.0]
    x0 = s0.to_vector()

    p0     = x0[0:3]
    v0     = x0[3:6]
    q0     = x0[6:10]
    omega0 = x0[10:13]
    theta0 = x0[13:15]
    thdot0 = x0[15:17]
    z3     = np.zeros(3)
    zn     = np.zeros(N_JOINTS)

    # ---- Gravity-compensating control via inverse dynamics ----
    # Ask: "what wrench + joint torques are needed to hold x0 stationary?"
    # Answer: inverse_dynamics with all desired accelerations = 0.
    # ams has no hard-lock; both simulators use the same torque vector.
    F_ext, tau_ext, jt = inverse_dynamics(
        ams_model, q0, p0, omega0, v0, theta0, thdot0, z3, z3, zn)
    u_const = np.concatenate([F_ext+[0,0,1], tau_ext, jt])

    print('=' * 60)
    print('  SIMULATOR COMPARISON: ams RK4  vs  MuJoCo RK4')
    print('  Gravity-compensating hover (static equilibrium at x0)')
    print('=' * 60)
    print(f'  dt      = {DT*1e3:.1f} ms')
    print(f'  F_ext   = {F_ext}  N')
    print(f'  tau_ext = {tau_ext}  Nm')
    print(f'  jt      = {jt}  Nm')

    t, x_ams, x_mjc = run_and_compare(
        'Gravity-comp hover (1 s)', ams_model, mjm, mjd, x0, u_const, 15.0)

    plot_comparison(t, x_ams, x_mjc, 'Gravity-comp hover')


if __name__ == '__main__':
    main()
