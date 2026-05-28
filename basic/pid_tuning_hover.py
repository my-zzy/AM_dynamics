"""PID hover experiment for the aerial manipulator drone base.

Three phases:
    1. START_HOVER  – stabilise at HOVER_START for HOVER_START_DUR seconds
    2. RAMP         – cosine ramp to TARGET over P2P_RAMP seconds
    3. HOLD         – hover at TARGET for P2P_DURATION seconds

Performance indices printed at the end
───────────────────────────────────────
  Integral (ramp + hold, t=0 at ramp start):
      IAE   [m·s]       – integral absolute error
      ISE   [m²·s]      – integral squared error
      ITAE  [m·s²]      – time-weighted IAE (penalises slow convergence)

  Step-response (measured w.r.t. TARGET):
      Rise time    [s]  – first reach within 10 % of step distance
      Settling time[s]  – first entry into 5 % band, held for SETTLE_HOLD s
      Overshoot    [%]  – max dist to target during hold / step dist × 100
      SS error     [m]  – mean dist to target in final SS_WINDOW seconds

  Hold-phase tracking:
      RMSE  [m]  –  root-mean-square error
      MAE   [m]  –  mean absolute error
      Max   [m]  –  worst-case peak error

  Attitude RMSE (hold phase):
      Roll / Pitch / Yaw  [deg]

  Control effort (ramp + hold):
      Mean thrust  [N]   – average vertical force command
      Thrust var.  [N]   – total variation Σ|ΔT|  (smoothness proxy)

Usage (from workspace root, conda activate main):
    python basic/pid_tuning_hover.py
"""
import sys
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, _ROOT)
from test_model import load_model, get_state, apply_platform_control
from pid_controller import DroneController, quat_to_euler_zyx
from ams.model import AerialManipulatorModel
from ams.kinematics import forward_kinematics

# ===========================================================================
#  PID GAINS  – edit to tune
# ===========================================================================
GAINS = dict(
    kp_z   = 10.0,
    ki_z   =  2.0,
    kd_z   =  6.0,
    kp_xy  =  5.5,
    kd_xy  =  1.0,
    kp_rp  = 10.0,
    ki_rp  =  4.5,
    kd_rp  =  2.5,
    kp_yaw =  5.0,
    ki_yaw =  0.2,
    kd_yaw =  1.5,
)

# ===========================================================================
#  EXPERIMENT CONFIG
# ===========================================================================
HOVER_START     = np.array([0.0, 0.0, 1.0])   # start hover position [m]
TARGET          = np.array([1.0, 0.0, 2.0])   # EE target position [m]  (drone base target derived at runtime)
YAW_TARGET      = np.deg2rad(0.0)             # desired yaw at target  [rad]

HOVER_START_DUR = 8.0   # hover at start before moving    [s]
P2P_RAMP        = 5.0   # cosine ramp duration             [s]
P2P_DURATION    = 5.0   # hover at target after arrival    [s]

# Settling-time criterion
SETTLE_BAND = 0.05   # fraction of step distance defining the settling band
SETTLE_HOLD = 0.5    # must stay in band continuously for this long [s]

# Steady-state window
SS_WINDOW = 2.0      # last N seconds of hold phase used for SS error [s]

# ===========================================================================
#  ARM JOINT SETPOINTS
# ===========================================================================
ARM_JOINTS = dict(joint1=0.0, joint2=0.0)
_KP_ARM    = 8.0
_KD_ARM    = 0.8
_ARM_MAP   = [('joint1', 0), ('joint2', 1)]

LOG_INTERVAL = 0.25   # console print interval [s]


# ---------------------------------------------------------------------------
#  Arm PD
# ---------------------------------------------------------------------------
def set_arm(data, model, arm_j_dofs=None):
    """PD control for arm joints with optional gravity feed-forward.

    arm_j_dofs: [j1_dof_idx, j2_dof_idx] from mj_model.jnt_dofadr, or None.
    """
    for i, (jname, ctrl_idx) in enumerate(_ARM_MAP):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        q   = data.qpos[model.jnt_qposadr[jid]]
        qd  = data.qvel[model.jnt_dofadr[jid]]
        tau = _KP_ARM * (ARM_JOINTS[jname] - q) - _KD_ARM * qd
        if arm_j_dofs is not None:
            tau += data.qfrc_bias[arm_j_dofs[i]]
        data.ctrl[ctrl_idx] = np.clip(tau, -5.0, 5.0)


# ---------------------------------------------------------------------------
#  Smooth cosine step
# ---------------------------------------------------------------------------
def smooth_step(t, t0, duration):
    """Returns 0→1 over [t0, t0+duration] using a cosine ramp."""
    s = np.clip((t - t0) / duration, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * s))


# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------
def make_log():
    return {'t': [], 'pos': [], 'pos_des': [], 'ee_pos': [],
            'rpy': [], 'rpy_des': [], 'thrust': [], 'phase': []}


def log_step(log, t, x, pos_des, ctrl, yaw_des, T, phase, ee_pos):
    roll, pitch, yaw = quat_to_euler_zyx(x[6:10])
    log['t'].append(t)
    log['pos'].append(x[:3].copy())
    log['pos_des'].append(pos_des.copy())
    log['ee_pos'].append(ee_pos.copy())
    log['rpy'].append(np.array([roll, pitch, yaw]))
    log['rpy_des'].append(np.array([ctrl.roll_des, ctrl.pitch_des, yaw_des]))
    log['thrust'].append(float(T))
    log['phase'].append(phase)


# ---------------------------------------------------------------------------
#  Performance indices
# ---------------------------------------------------------------------------
def compute_indices(log, step_dist, dt_sim):
    """Compute all performance indices from the recorded log.

    Parameters
    ----------
    log       : dict returned by make_log()
    step_dist : scalar distance from HOVER_START to TARGET [m]
    dt_sim    : simulation timestep [s]

    Returns
    -------
    dict of scalar index values
    """
    t          = np.array(log['t'])
    ee_pos_arr = np.array(log['ee_pos'])     # (N, 3)  EE world position
    rpy        = np.array(log['rpy'])        # (N, 3) [rad]
    rpy_d      = np.array(log['rpy_des'])    # (N, 3) [rad]
    thrust     = np.array(log['thrust'])     # (N,)
    phases     = np.array(log['phase'])      # (N,) str

    # EE distance to final target — used for all position indices
    # (directly comparable to mpc_reach_test compute_reach_indices)
    err3d   = np.linalg.norm(ee_pos_arr - TARGET, axis=1)
    err_tgt = err3d
    # Per-axis attitude error [rad]
    att_err = np.abs(rpy - rpy_d)

    # Phase masks
    mask_ramp   = phases == 'RAMP'
    mask_hold   = phases == 'HOLD'
    mask_motion = mask_ramp | mask_hold   # ramp + hold (excludes start hover)

    # Time relative to ramp start (t=0 when the drone begins moving)
    t_ramp_start = t[mask_motion][0] if mask_motion.any() else 0.0
    t_rel        = np.clip(t - t_ramp_start, 0.0, None)

    # ── Integral indices (ramp + hold) ──────────────────────────────────────
    t_m  = t[mask_motion]
    e_m  = err3d[mask_motion]
    tr_m = t_rel[mask_motion]

    IAE  = float(np.trapz(e_m,        t_m))
    ISE  = float(np.trapz(e_m ** 2,   t_m))
    ITAE = float(np.trapz(tr_m * e_m, t_m))

    # ── Step-response indices ────────────────────────────────────────────────
    band = SETTLE_BAND * step_dist

    # Rise time: first step where distance to TARGET < 10 % of step_dist
    rise_thresh = 0.10 * step_dist
    rise_mask   = err_tgt < rise_thresh
    rise_time   = float(t_rel[np.argmax(rise_mask)]) if rise_mask.any() else float('nan')

    # Settling time: first entry into SETTLE_BAND held for SETTLE_HOLD seconds
    in_band             = err_tgt < band
    settle_hold_steps   = max(1, int(SETTLE_HOLD / dt_sim))
    settle_time         = float('nan')
    for i in range(len(in_band) - settle_hold_steps + 1):
        if np.all(in_band[i: i + settle_hold_steps]):
            settle_time = float(t_rel[i])
            break

    # Overshoot: max distance to TARGET during hold / step_dist × 100 %
    if mask_hold.any():
        overshoot_pct = float(np.max(err_tgt[mask_hold]) / step_dist * 100.0)
    else:
        overshoot_pct = float('nan')

    # Steady-state error: mean distance to TARGET in the last SS_WINDOW seconds
    if mask_hold.any():
        t_hold    = t[mask_hold]
        mask_ss   = mask_hold & (t >= t_hold[-1] - SS_WINDOW)
        ss_err    = float(np.mean(err_tgt[mask_ss])) if mask_ss.any() else float('nan')
    else:
        ss_err = float('nan')

    # ── Hold-phase tracking ──────────────────────────────────────────────────
    if mask_hold.any():
        e_hold  = err3d[mask_hold]
        rmse    = float(np.sqrt(np.mean(e_hold ** 2)))
        mae     = float(np.mean(e_hold))
        max_err = float(np.max(e_hold))
    else:
        rmse = mae = max_err = float('nan')

    # ── Attitude RMSE (hold phase) ───────────────────────────────────────────
    if mask_hold.any():
        att_rmse = [float(np.sqrt(np.mean(att_err[mask_hold, i] ** 2)))
                    for i in range(3)]
    else:
        att_rmse = [float('nan')] * 3

    # ── Control effort (ramp + hold) ─────────────────────────────────────────
    T_m = thrust[mask_motion]
    mean_thrust = float(np.mean(T_m))        if T_m.size > 0 else float('nan')
    thrust_var  = float(np.sum(np.abs(np.diff(T_m)))) if T_m.size > 1 else float('nan')

    return dict(
        IAE=IAE, ISE=ISE, ITAE=ITAE,
        rise_time=rise_time, settle_time=settle_time,
        overshoot_pct=overshoot_pct, ss_err=ss_err,
        rmse=rmse, mae=mae, max_err=max_err,
        att_rmse_roll=att_rmse[0], att_rmse_pitch=att_rmse[1], att_rmse_yaw=att_rmse[2],
        mean_thrust=mean_thrust, thrust_var=thrust_var,
    )


def _fmt(v, fmt='.4f'):
    """Format a value, showing 'n/a' for NaN."""
    return f'{v:{fmt}}' if not (isinstance(v, float) and np.isnan(v)) else 'n/a'


def print_indices(idx, step_dist):
    w = 50
    print('\n' + '=' * w)
    print('  PERFORMANCE INDICES')
    print('=' * w)
    print(f'  Step distance              : {step_dist:.4f} m')
    print(f'  Settling band ({SETTLE_BAND*100:.0f}%)         : {SETTLE_BAND*step_dist:.4f} m')
    print('-' * w)
    print('  Integral indices  (ramp + hold, t=0 at ramp start)')
    print(f'    IAE   [m·s]              : {_fmt(idx["IAE"])}')
    print(f'    ISE   [m²·s]             : {_fmt(idx["ISE"])}')
    print(f'    ITAE  [m·s²]             : {_fmt(idx["ITAE"])}')
    print('-' * w)
    print('  Step-response indices  (w.r.t. TARGET)')
    print(f'    Rise time    (90%)   [s] : {_fmt(idx["rise_time"], ".3f")}')
    print(f'    Settling time (5%)   [s] : {_fmt(idx["settle_time"], ".3f")}')
    print(f'    Overshoot            [%] : {_fmt(idx["overshoot_pct"], ".2f")}')
    print(f'    Steady-state error   [m] : {_fmt(idx["ss_err"])}')
    print('-' * w)
    print('  Hold-phase tracking')
    print(f'    RMSE             [m]     : {_fmt(idx["rmse"])}')
    print(f'    MAE              [m]     : {_fmt(idx["mae"])}')
    print(f'    Max error        [m]     : {_fmt(idx["max_err"])}')
    print('-' * w)
    print('  Attitude RMSE  (hold phase)')
    roll_deg  = np.degrees(idx['att_rmse_roll'])  if not np.isnan(idx['att_rmse_roll'])  else float('nan')
    pitch_deg = np.degrees(idx['att_rmse_pitch']) if not np.isnan(idx['att_rmse_pitch']) else float('nan')
    yaw_deg   = np.degrees(idx['att_rmse_yaw'])   if not np.isnan(idx['att_rmse_yaw'])   else float('nan')
    print(f'    Roll             [deg]   : {_fmt(roll_deg,  ".3f")}')
    print(f'    Pitch            [deg]   : {_fmt(pitch_deg, ".3f")}')
    print(f'    Yaw              [deg]   : {_fmt(yaw_deg,   ".3f")}')
    print('-' * w)
    print('  Control effort  (ramp + hold)')
    print(f'    Mean thrust      [N]     : {_fmt(idx["mean_thrust"], ".3f")}')
    print(f'    Thrust variation [N]     : {_fmt(idx["thrust_var"],  ".3f")}')
    print('=' * w)


# ---------------------------------------------------------------------------
#  Plot
# ---------------------------------------------------------------------------
def plot_results(log, title='Hover P2P'):
    import matplotlib.pyplot as plt

    t     = np.array(log['t'])
    pos   = np.array(log['pos'])
    pos_d = np.array(log['pos_des'])
    rpy   = np.degrees(np.array(log['rpy']))
    rpy_d = np.degrees(np.array(log['rpy_des']))
    phases = np.array(log['phase'])

    labels_pos = ['x [m]', 'y [m]', 'z [m]']
    labels_att = ['roll [deg]', 'pitch [deg]', 'yaw [deg]']

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(title, fontsize=13)

    # Shade phases
    phase_colors = {'START_HOVER': '#e8f4e8', 'RAMP': '#fff3cd', 'HOLD': '#e8f0fe'}
    phase_labels = {'START_HOVER': 'Start hover', 'RAMP': 'Ramp', 'HOLD': 'Hold'}
    shaded = set()
    for ax in axes:
        prev_phase = phases[0]
        seg_start  = t[0]
        for k in range(1, len(t)):
            if phases[k] != prev_phase or k == len(t) - 1:
                color = phase_colors.get(prev_phase, 'white')
                label = phase_labels.get(prev_phase, prev_phase) if prev_phase not in shaded else None
                ax.axvspan(seg_start, t[k], color=color, alpha=0.4, label=label)
                shaded.add(prev_phase)
                seg_start  = t[k]
                prev_phase = phases[k]

    for i in range(3):
        ax = axes[i]
        ax.plot(t, pos[:, i],   color='tab:blue',   label='actual')
        ax.plot(t, pos_d[:, i], color='tab:orange', linestyle='--', label='desired')
        ax.set_ylabel(labels_pos[i])
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, linewidth=0.4)

    for i in range(3):
        ax = axes[3 + i]
        ax.plot(t, rpy[:, i],   color='tab:green',  label='actual')
        ax.plot(t, rpy_d[:, i], color='tab:red',    linestyle='--', label='desired')
        ax.set_ylabel(labels_att[i])
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, linewidth=0.4)

    axes[-1].set_xlabel('time [s]')
    plt.tight_layout()

    fname    = title.lower().replace(' ', '_')
    out_path = os.path.join(os.path.dirname(__file__), f'pid_tuning_{fname}.png')
    plt.savefig(out_path, dpi=150)
    print(f'Plot saved → {out_path}')
    try:
        plt.show()
    except Exception:
        pass


# ---------------------------------------------------------------------------
#  Main simulation loop
# ---------------------------------------------------------------------------
def run_hover_p2p(model, data, ctrl, dt):
    # ── EE inverse kinematics (arm fixed → constant body-frame offset) ────
    am_model = AerialManipulatorModel()
    _q_level  = np.array([0., 0., 0., 1.])            # level drone, xyzw
    _theta_nom = np.array([ARM_JOINTS['joint1'],
                           ARM_JOINTS['joint2']])
    _, p_nom, _ = forward_kinematics(am_model, _q_level,
                                     np.zeros(3), _theta_nom)
    ee_offset    = p_nom[3]                            # EE pos when drone at origin
    DRONE_TARGET = TARGET - ee_offset                  # drone base → EE lands at TARGET
    EE_START     = HOVER_START + ee_offset             # EE pos at start hover

    step_dist = float(np.linalg.norm(TARGET - EE_START))  # EE step distance

    # ── Precompute joint DOF indices for gravity feed-forward ─────────────
    arm_j_dofs = [
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint1')],
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint2')],
    ]

    total_time = HOVER_START_DUR + P2P_RAMP + P2P_DURATION
    t0_wall  = time.perf_counter()
    t0_sim   = data.time
    last_log = -LOG_INTERVAL
    log      = make_log()

    print(f'\n=== Hover P2P  (EE → TARGET) ===')
    print(f'  EE start              : {np.round(EE_START, 4)}')
    print(f'  EE target  (TARGET)   : {TARGET}')
    print(f'  Drone base target     : {np.round(DRONE_TARGET, 4)}')
    print(f'  EE step distance      : {step_dist:.4f} m')
    print(f'  EE offset             : {np.round(ee_offset, 4)}')
    print(f'  Arm joints            : joint1={ARM_JOINTS["joint1"]:.3f}  '
          f'joint2={ARM_JOINTS["joint2"]:.3f} rad')
    print(f'  Phase 1 – START HOVER : {HOVER_START_DUR:.1f} s')
    print(f'  Phase 2 – RAMP        : {P2P_RAMP:.1f} s')
    print(f'  Phase 3 – HOLD        : {P2P_DURATION:.1f} s')

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while data.time - t0_sim < total_time:
            if not viewer.is_running():
                break

            t = data.time - t0_sim
            x = get_state(model, data)

            # ── EE position via forward kinematics ────────────────────────
            _, p_ee_frames, _ = forward_kinematics(am_model, x[6:10],
                                                   x[:3], x[13:15])
            ee_pos = p_ee_frames[3]

            # ── Determine phase and drone-base setpoint ───────────────────
            if t < HOVER_START_DUR:
                pos_des = HOVER_START.copy()
                yaw_des = 0.0
                phase   = 'START_HOVER'

            elif t < HOVER_START_DUR + P2P_RAMP:
                s       = smooth_step(t, HOVER_START_DUR, P2P_RAMP)
                pos_des = HOVER_START + s * (DRONE_TARGET - HOVER_START)
                yaw_des = s * YAW_TARGET
                phase   = 'RAMP'

            else:
                pos_des = DRONE_TARGET.copy()
                yaw_des = YAW_TARGET
                phase   = 'HOLD'

            # ── Control ───────────────────────────────────────────────────
            T, tau = ctrl.compute(x[:3], x[3:6], x[6:10], x[10:13],
                                  pos_des, yaw_des, dt)
            apply_platform_control(data, T, tau)
            set_arm(data, model, arm_j_dofs)
            mujoco.mj_step(model, data)
            log_step(log, t, x, pos_des, ctrl, yaw_des, T, phase, ee_pos)

            # ── Console output ────────────────────────────────────────────
            if t - last_log >= LOG_INTERVAL:
                ee_err = np.linalg.norm(ee_pos - TARGET)
                print(f'  t={t:6.2f}s  [{phase:12s}]  '
                      f'EE=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}]  '
                      f'EE_err={ee_err*1000:.1f} mm')
                last_log = t

            # ── Real-time pacing ──────────────────────────────────────────
            sim_elapsed  = data.time - t0_sim
            wall_elapsed = time.perf_counter() - t0_wall
            if sim_elapsed > wall_elapsed:
                time.sleep(sim_elapsed - wall_elapsed)
            viewer.sync()

        print('\nSimulation complete.  Close viewer to compute indices and plot.')
        while viewer.is_running():
            time.sleep(0.05)

    # ── Final EE stats (mirrors mpc_reach_test output) ────────────────────
    ee_pos_arr = np.array(log['ee_pos'])
    t_arr      = np.array(log['t'])
    final_ee_err = float(np.linalg.norm(ee_pos_arr[-1] - TARGET))
    if len(t_arr) >= 2:
        dt_last  = t_arr[-1] - t_arr[-2]
        ee_vel_f = (ee_pos_arr[-1] - ee_pos_arr[-2]) / dt_last if dt_last > 0 else np.zeros(3)
        ee_spd_f = float(np.linalg.norm(ee_vel_f))
    else:
        ee_vel_f = np.zeros(3)
        ee_spd_f = 0.0
    print(f'\n=== Done ===')
    print(f'Final EE position : {np.round(ee_pos_arr[-1], 4)}')
    print(f'Final EE error    : {final_ee_err*1000:.1f} mm')
    print(f'Final EE speed    : {ee_spd_f*1000:.2f} mm/s'
          f'  (vx={ee_vel_f[0]*1000:.2f}  vy={ee_vel_f[1]*1000:.2f}'
          f'  vz={ee_vel_f[2]*1000:.2f} mm/s)')

    # ── Post-run analysis ─────────────────────────────────────────────────
    idx = compute_indices(log, step_dist, dt)
    print_indices(idx, step_dist)
    plot_results(log, title='Hover P2P')
    return log, idx


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------
def main():
    model, data = load_model()
    total_mass  = np.sum(model.body_mass)
    print(f'Model mass : {total_mass:.3f} kg')
    print(f'Gains      : {GAINS}')

    ctrl = DroneController(mass=total_mass, **GAINS)

    mujoco.mj_resetData(model, data)
    data.qpos[0] = HOVER_START[0]
    data.qpos[1] = HOVER_START[1]
    data.qpos[2] = HOVER_START[2]
    mujoco.mj_forward(model, data)
    ctrl.reset()

    dt = model.opt.timestep
    run_hover_p2p(model, data, ctrl, dt)


if __name__ == '__main__':
    main()
