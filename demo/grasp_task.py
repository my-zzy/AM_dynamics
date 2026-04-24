"""Grasp demo: pick-and-place with the aerial manipulator.

Run from workspace root:
    conda activate main
    python demo/grasp_task.py

Phases (see demo/grasp.md for full design):
    0  TAKEOFF     ramp from ground to hover setpoint
    1  INIT        stabilise hover
    2  ARM READY   hold arm at initial angle, open gripper
    3  APPROACH    translate drone closer, IK to box centre
    4  GRASP       close gripper
    5  LIFT        raise hover setpoint
    6  TRANSPORT   fly to drop-off
    7  PLACE       descend, open gripper
    8  RETRACT     fold arm, return home
"""

## TODO:
# edit hover setpoint beyond reach
# arm joint tracking

import sys
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# Allow imports from basic/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'basic'))
from test_model import get_state, apply_platform_control
from pid_controller import DroneController

# ---------------------------------------------------------------------------
# Arm constants (from grasp_scene.xml)
# ---------------------------------------------------------------------------
L1 = 0.12        # link1 length
L2 = 0.238       # link2 + palm + EE site
MOUNT_Z = 0.05   # arm mount offset below base origin

# Actuator ctrl indices
CTRL_J1 = 0
CTRL_J2 = 1
CTRL_GL = 2      # gripper left
CTRL_GR = 3      # gripper right

# Joint PD gains for arm position control (torque actuators)
KP_ARM = 8.0
KD_ARM = 0.8

# ---------------------------------------------------------------------------
# IK  (updated sign convention — see grasp.md)
# ---------------------------------------------------------------------------

def arm_ik(base_pos, target_pos):
    """2-D inverse kinematics for the 2-link arm.

    FK:
        ex = bx - L1*sin(t1) + L2*cos(t1+t2)
        ez = bz - 0.05 - L1*cos(t1) - L2*sin(t1+t2)

    Returns (theta1, theta2) or None if unreachable.
    """
    dx = target_pos[0] - base_pos[0]
    dz = target_pos[2] - base_pos[2] + MOUNT_Z

    r2 = dx**2 + dz**2
    cos_reach = (r2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(cos_reach) > 1.0:
        return None

    # sin(θ2) formulation from L-config zero pose
    sin_t2 = cos_reach  # same expression, different trig identity
    t2 = np.arcsin(np.clip(sin_t2, -1.0, 1.0))  # positive = elbow-down

    t1 = np.arctan2(-dz, dx) - np.arctan2(L1 + L2 * np.sin(t2), L2 * np.cos(t2))

    return t1, t2


def arm_fk(base_pos, t1, t2):
    """Forward kinematics — returns EE world position (3,)."""
    ex = base_pos[0] - L1 * np.sin(t1) + L2 * np.cos(t1 + t2)
    ey = base_pos[1]
    ez = base_pos[2] - MOUNT_Z - L1 * np.cos(t1) - L2 * np.sin(t1 + t2)
    return np.array([ex, ey, ez])


def drone_pos_from_joints(box_target, t1, t2):
    """Back-compute drone base position so that FK(base, t1, t2) == box_target.

    Inverts arm_fk analytically for x and z; y is taken from box_target.
    """
    base_x = box_target[0] + L1 * np.sin(t1) - L2 * np.cos(t1 + t2)
    base_z = box_target[2] + MOUNT_Z + L1 * np.cos(t1) + L2 * np.sin(t1 + t2)
    return np.array([base_x, box_target[1], base_z])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smooth_ramp(t, t0, duration, start, end):
    """Cosine smooth interpolation from start to end over [t0, t0+duration]."""
    s = np.clip((t - t0) / duration, 0.0, 1.0)
    s = 0.5 * (1.0 - np.cos(np.pi * s))  # smooth step
    return start + s * (end - start)


def load_grasp_scene():
    xml_path = os.path.join(os.path.dirname(__file__), 'grasp_scene.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data


def get_grasp_state(model, data):
    """Extended state: platform + arm + gripper joints."""
    # Platform: pos(3), vel(3), quat_xyzw(4), omega(3)
    # free_joint qpos indices: 0-2 pos, 3-6 quat(wxyz)
    # But we need to skip box_free joint first.
    # Joint order in qpos: box_free(7), free_joint(7), joint1(1), joint2(1),
    #                       gripper_left(1), gripper_right(1)
    # Let's use named access for robustness.
    base_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'free_joint')
    qa = data.qpos[model.jnt_qposadr[base_jnt_id]:]
    va_start = model.jnt_dofadr[base_jnt_id]

    p = qa[0:3].copy()
    q_wxyz = qa[3:7]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    v = data.qvel[va_start:va_start+3].copy()
    omega = data.qvel[va_start+3:va_start+6].copy()

    # Arm joints
    j1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint1')
    j2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint2')
    gl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'gripper_left_joint')
    gr_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'gripper_right_joint')

    theta1 = data.qpos[model.jnt_qposadr[j1_id]]
    theta2 = data.qpos[model.jnt_qposadr[j2_id]]
    g_left = data.qpos[model.jnt_qposadr[gl_id]]
    g_right = data.qpos[model.jnt_qposadr[gr_id]]

    theta1_dot = data.qvel[model.jnt_dofadr[j1_id]]
    theta2_dot = data.qvel[model.jnt_dofadr[j2_id]]

    return {
        'pos': p, 'vel': v, 'quat': q_xyzw, 'omega': omega,
        'theta': np.array([theta1, theta2]),
        'theta_dot': np.array([theta1_dot, theta2_dot]),
        'gripper': np.array([g_left, g_right]),
    }


def get_box_pos(model, data):
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'box')
    return data.xpos[box_id].copy()


def apply_arm_pd(data, theta_des, theta_cur, theta_dot, bias=None):
    """PD torque control for arm joints with optional gravity feedforward.

    bias: array([tau_grav_j1, tau_grav_j2]) from data.qfrc_bias at joint DOFs.
          When provided, cancels gravity/Coriolis so steady-state error → 0.
    """
    tau = KP_ARM * (theta_des - theta_cur) - KD_ARM * theta_dot
    if bias is not None:
        tau = tau + bias          # gravity feedforward
    tau = np.clip(tau, -5.0, 5.0)
    data.ctrl[CTRL_J1] = tau[0]
    data.ctrl[CTRL_J2] = tau[1]


def apply_gripper(data, close=False):
    """Set gripper actuator forces."""
    if close:
        data.ctrl[CTRL_GL] = -20.0
        data.ctrl[CTRL_GR] = 20.0
    else:
        data.ctrl[CTRL_GL] = 0.0
        data.ctrl[CTRL_GR] = 0.0


def plot_approach_phase(log):
    """Dedicated 2-D xz plot for the approach phase (phase 3)."""
    import matplotlib.pyplot as plt

    t        = np.array(log['t'])
    phases   = np.array(log['phase'])
    drone    = np.array(log['drone_pos'])
    ee       = np.array(log['ee_pos'])
    hover_d  = np.array(log['hover_des'])

    mask = phases == 3
    if not np.any(mask):
        print('No approach-phase data to plot.')
        return

    t_ap     = t[mask]
    drone_ap = drone[mask]
    ee_ap    = ee[mask]
    hdes_ap  = hover_d[mask]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f'Approach Phase  (t = {t_ap[0]:.1f} – {t_ap[-1]:.1f} s)   '
                 f'method={GRASP_METHOD!r}', fontsize=12)

    # ── Left: xz spatial plot ──────────────────────────────────────────────
    ax = axes[0]
    # Drone actual
    ax.plot(drone_ap[:, 0], drone_ap[:, 2],
            color='tab:blue', lw=1.8, label='Drone base (actual)')
    ax.scatter(drone_ap[0, 0],  drone_ap[0, 2],  color='tab:blue', s=70,
               marker='o', zorder=6)
    ax.scatter(drone_ap[-1, 0], drone_ap[-1, 2], color='tab:blue', s=70,
               marker='x', zorder=6)
    # Drone desired
    ax.plot(hdes_ap[:, 0], hdes_ap[:, 2],
            color='tab:blue', lw=1.2, ls='--', label='Drone desired')
    # EE actual
    ax.plot(ee_ap[:, 0], ee_ap[:, 2],
            color='tab:orange', lw=1.8, label='Gripper EE (actual)')
    ax.scatter(ee_ap[0, 0],  ee_ap[0, 2],  color='tab:orange', s=70,
               marker='o', zorder=6)
    ax.scatter(ee_ap[-1, 0], ee_ap[-1, 2], color='tab:orange', s=70,
               marker='x', zorder=6)
    # Target
    ax.scatter(BOX_TARGET[0], BOX_TARGET[2], color='tab:green', s=180,
               marker='*', zorder=7, label='Box target')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title('xz Trajectory  (o = start,  x = end,  -- = desired)')
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.4)

    # ── Right: position vs time ────────────────────────────────────────────
    ax = axes[1]
    # x components
    ax.plot(t_ap, drone_ap[:, 0], color='tab:blue',   lw=1.5, label='Drone x (actual)')
    ax.plot(t_ap, hdes_ap[:, 0],  color='tab:blue',   lw=1.0, ls='--', label='Drone x (desired)')
    ax.plot(t_ap, ee_ap[:, 0],    color='tab:orange', lw=1.5, label='EE x')
    ax.axhline(BOX_TARGET[0], color='tab:green', lw=1.0, ls=':', label='Target x')
    # z components
    ax.plot(t_ap, drone_ap[:, 2], color='royalblue',  lw=1.5, ls='-',  label='Drone z (actual)')
    ax.plot(t_ap, hdes_ap[:, 2],  color='royalblue',  lw=1.0, ls='--', label='Drone z (desired)')
    ax.plot(t_ap, ee_ap[:, 2],    color='darkorange',  lw=1.5, ls='-',  label='EE z')
    ax.axhline(BOX_TARGET[2], color='darkgreen', lw=1.0, ls=':', label='Target z')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('position [m]')
    ax.set_title('x and z vs time  (solid=actual  dashed=desired  dotted=target)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, lw=0.4)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'approach_phase.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Approach plot saved → {out_path}')
    try:
        plt.show()
    except Exception:
        pass


def plot_trajectories(log):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    t      = np.array(log['t'])
    drone  = np.array(log['drone_pos'])
    ee     = np.array(log['ee_pos'])
    box    = np.array(log['box_pos'])
    theta  = np.degrees(np.array(log['theta']))
    tdes   = np.degrees(np.array(log['theta_des']))

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Grasp Task Trajectories', fontsize=13)
    gs = gridspec.GridSpec(5, 2, figure=fig)

    # 2-D trajectory in xz-plane (left column, spans all rows)
    ax2d = fig.add_subplot(gs[:, 0])
    if PLOT_DRONE:
        ax2d.plot(drone[:, 0], drone[:, 2],
                  color='tab:blue', linewidth=1.5, label='Drone base')
        ax2d.scatter(drone[0, 0],  drone[0, 2],  color='tab:blue', s=60, marker='o', zorder=5)
        ax2d.scatter(drone[-1, 0], drone[-1, 2], color='tab:blue', s=60, marker='x', zorder=5)
    ax2d.plot(ee[:, 0], ee[:, 2],
              color='tab:orange', linewidth=1.5, label='Gripper EE')
    ax2d.scatter(ee[0, 0],  ee[0, 2],  color='tab:orange', s=60, marker='o', zorder=5)
    ax2d.scatter(ee[-1, 0], ee[-1, 2], color='tab:orange', s=60, marker='x', zorder=5)
    ax2d.scatter(box[0, 0], box[0, 2], color='tab:green', s=120, marker='*', zorder=6, label='Box target')
    ax2d.set_xlabel('x [m]')
    ax2d.set_ylabel('z [m]')
    ax2d.legend(fontsize=8)
    ax2d.set_title('xz Trajectory  (o=start  x=end)')
    ax2d.grid(True, linewidth=0.4)
    ax2d.set_aspect('equal')

    # Position time series (right column, rows 0-2)
    pos_labels = ['x [m]', 'y [m]', 'z [m]']
    for i in range(3):
        ax = fig.add_subplot(gs[i, 1])
        if PLOT_DRONE:
            ax.plot(t, drone[:, i], color='tab:blue',   linewidth=1.2, label='Drone base')
        ax.plot(t, ee[:, i],    color='tab:orange', linewidth=1.2, label='Gripper EE')
        ax.plot(t, box[:, i],   color='tab:green',  linewidth=1.2, linestyle='--', label='Box')
        ax.set_ylabel(pos_labels[i])
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, linewidth=0.4)

    # Joint angle time series (right column, rows 3-4)
    joint_labels = ['joint1 [deg]', 'joint2 [deg]']
    for i in range(2):
        ax = fig.add_subplot(gs[3 + i, 1])
        ax.plot(t, theta[:, i], color='tab:purple', linewidth=1.2, label='actual')
        ax.plot(t, tdes[:, i],  color='tab:red',    linewidth=1.2, linestyle='--', label='desired')
        ax.set_ylabel(joint_labels[i])
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, linewidth=0.4)
        if i == 1:
            ax.set_xlabel('time [s]')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'grasp_trajectory.png')
    plt.savefig(out_path, dpi=150)
    print(f'Trajectory plot saved → {out_path}')
    try:
        plt.show()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Phase timing
# ---------------------------------------------------------------------------
PHASE_TIMES = [
    (0,  5.0,  'TAKEOFF'),
    (1,  10.0,  'INIT'),
    (2,  20.0, 'ARM READY'),
    (3,  30.0, 'APPROACH'),
    (4,  35.0, 'GRASP'),
    (5,  40.0, 'LIFT'),
    (6,  45.0, 'TRANSPORT'),
    (7,  50.0, 'PLACE'),
    (8,  55.0, 'RETRACT'),
]

# z-height of base when resting on the ground (feet at z=-0.212 from base)
GROUND_Z = 0.22

# Hover setpoints per phase
HOVER = {
    0: np.array([0.0,  0.0, 1.5]),   # takeoff target
    1: np.array([0.0,  0.0, 1.5]),
    2: np.array([0.0,  0.0, 1.5]),
    3: np.array([0.25, 0.0, 1.45]),
    4: np.array([0.25, 0.0, 1.45]),
    5: np.array([0.25, 0.0, 1.65]),
    6: np.array([-0.5, 0.0, 1.7]),
    7: np.array([-0.5, 0.0, 1.3]),
    8: np.array([0.0,  0.0, 1.5]),
}

BOX_TARGET = np.array([0.40, 0.0, 1.125])

# ---------------------------------------------------------------------------
# Approach method
# ---------------------------------------------------------------------------
# 'arm_only'   – drone hovers fixed (pre-approached in phase 2), arm ramps to IK
# 'drone_only' – arm locked at S2_JOINTS, drone flies to back-computed position
# 'hybrid'     – drone moves to HOVER[3] while arm tracks live IK  (original)
GRASP_METHOD = 'drone_only'

# Fixed joint angles used by drone_only (arm pre-aimed at box)
S2_JOINTS = np.array([-0.2, 0.2])   # [theta1, theta2] rad

# ---------------------------------------------------------------------------
# Plot flags
# ---------------------------------------------------------------------------
PLOT_DRONE = False    # set False to hide drone base trajectory from plots

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GAINS = dict(
    kp_z    = 10.0,
    ki_z    =  2.0,
    kd_z    =  6.0,
    kp_xy   =  5.5,
    kd_xy   =  1.0,
    kp_rp   =  10.0,
    ki_rp   =  4.5,
    kd_rp   =  2.5,
    kp_yaw  =  5.0,
    ki_yaw  =  0.2,
    kd_yaw  =  1.5,
)


def run_grasp():
    model, data = load_grasp_scene()
    # Only sum robot bodies; exclude world, pedestal, and free box
    _exclude = {'world', 'pedestal', 'box'}
    robot_body_ids = [
        i for i in range(model.nbody)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) not in _exclude
    ]
    total_mass = sum(model.body_mass[i] for i in robot_body_ids)
    print(f'Robot mass: {total_mass:.3f} kg  (mg = {total_mass*9.81:.2f} N)')

    ctrl = DroneController(mass=total_mass, **GAINS)
    dt = model.opt.timestep

    # Precompute joint DOF addresses for gravity feedforward
    _j1_dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint1')]
    _j2_dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'joint2')]

    # Reset — place drone on the ground
    mujoco.mj_resetData(model, data)
    base_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'free_joint')
    data.qpos[model.jnt_qposadr[base_jnt_id] + 2] = GROUND_Z
    mujoco.mj_forward(model, data)
    ctrl.reset()

    # Phase tracking
    current_phase = -1
    hover_des = HOVER[0].copy()
    hover_prev = HOVER[0].copy()
    theta_des = np.array([0.0, 0.0])
    phase_t0 = 0.0

    # Pre-compute IK targets
    # Phase 3 (APPROACH): verify box is reachable from APPROACH setpoint
    ik_phase3 = arm_ik(HOVER[3], BOX_TARGET)
    if ik_phase3 is None:
        print('ERROR: Phase 3 IK unreachable — check setpoints!')
        return
    ee3 = arm_fk(HOVER[3], *ik_phase3)
    print(f'Phase 3 IK: theta1={np.rad2deg(ik_phase3[0]):.1f}°  theta2={np.rad2deg(ik_phase3[1]):.1f}°')
    print(f'Phase 3 FK check: EE = {ee3}  (target: {BOX_TARGET})')

    # Method-specific pre-computation
    print(f'\nGRASP_METHOD = {GRASP_METHOD!r}')
    if GRASP_METHOD == 'drone_only':
        _s2_hover = drone_pos_from_joints(BOX_TARGET, S2_JOINTS[0], S2_JOINTS[1])
        ee_s2 = arm_fk(_s2_hover, S2_JOINTS[0], S2_JOINTS[1])
        print(f'  S2_JOINTS = [{np.rad2deg(S2_JOINTS[0]):.1f}°, {np.rad2deg(S2_JOINTS[1]):.1f}°]')
        print(f'  computed hover = {_s2_hover}')
        print(f'  FK check EE = {ee_s2}  (target: {BOX_TARGET})')
    else:
        _s2_hover = HOVER[3].copy()   # unused placeholder

    # arm_only runtime state (filled at phase 3 transition)
    _arm_only_theta_start = np.zeros(2)
    _arm_only_theta_end   = np.zeros(2)
    # drone_only runtime state (arm start captured at phase 2 transition)
    _drone_only_arm_start = np.zeros(2)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        t0_wall = time.perf_counter()
        t0_sim = data.time
        sim_duration = 50.0
        _retract_start = np.array([0.0, 0.0])

        # Trajectory logging
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
        log = {'t': [], 'drone_pos': [], 'ee_pos': [], 'box_pos': [],
               'theta': [], 'theta_des': [], 'hover_des': [], 'phase': []}

        while data.time - t0_sim < sim_duration:
            if not viewer.is_running():
                return

            t = data.time - t0_sim

            # --- Determine phase ---
            new_phase = 7
            for ph, t_end, _ in PHASE_TIMES:
                if t < t_end:
                    new_phase = ph
                    break

            if new_phase != current_phase:
                current_phase = new_phase
                phase_t0 = t
                label = PHASE_TIMES[current_phase][2] if current_phase < len(PHASE_TIMES) else '?'
                print(f'\n[t={t:.1f}s] === Phase {current_phase}: {label} ===')

                # Store previous hover for smooth ramp
                st = get_grasp_state(model, data)
                hover_prev = st['pos'].copy()
                # Override y to 0 (we only move in xz)
                hover_prev[1] = 0.0

                # arm_only: lock in IK target once at phase 3 start
                if current_phase == 3 and GRASP_METHOD == 'arm_only':
                    ik_ao = arm_ik(hover_prev, BOX_TARGET)
                    if ik_ao is None:
                        print('  WARNING: arm_only IK unreachable — drone too far from box')
                    else:
                        _arm_only_theta_end = np.array(ik_ao)
                    _arm_only_theta_start = st['theta'].copy()
                    print(f'  hover point: {hover_prev} box target: {BOX_TARGET}')
                    print(f'  arm_only joints: {np.rad2deg(_arm_only_theta_start)} '
                          f'→ {np.rad2deg(_arm_only_theta_end)} deg')

                # drone_only: capture arm start position at phase 2 transition
                if current_phase == 2 and GRASP_METHOD == 'drone_only':
                    _drone_only_arm_start = st['theta'].copy()
                    print(f'  drone_only arm ramp: {np.rad2deg(_drone_only_arm_start)} '
                          f'→ {np.rad2deg(S2_JOINTS)} deg  (over phase 2)')

            st = get_grasp_state(model, data)
            phase_dt = t - phase_t0

            # Record trajectory
            log['t'].append(t)
            log['drone_pos'].append(st['pos'].copy())
            log['ee_pos'].append(data.site_xpos[ee_site_id].copy())
            log['box_pos'].append(get_box_pos(model, data))
            log['theta'].append(st['theta'].copy())
            log['theta_des'].append(theta_des.copy())
            log['hover_des'].append(hover_des.copy())
            log['phase'].append(current_phase)

            # --- Phase logic ---
            gripper_close = False

            if current_phase == 0:
                # TAKEOFF: ramp from ground to hover setpoint
                ramp_dur = 4.0
                hover_des = smooth_ramp(phase_dt, 0, ramp_dur,
                                        np.array([0.0, 0.0, GROUND_Z]), HOVER[0])
                theta_des = np.array([0.0, 0.0])

            elif current_phase == 1:
                # INIT: stabilise at hover, arm folded
                hover_des = HOVER[1].copy()
                theta_des = np.array([0.0, 0.0])

            elif current_phase == 2:
                # ARM READY:
                #   arm_only  – drone pre-approaches to HOVER[3], arm stays folded
                #   drone_only – drone holds, arm ramps to S2_JOINTS (pre-position)
                #   hybrid    – drone holds, arm stays folded
                if GRASP_METHOD == 'arm_only':
                    ramp_dur = 8.0
                    hover_des = smooth_ramp(phase_dt, 0, ramp_dur, hover_prev, HOVER[3])
                    theta_des = np.array([0.0, 0.0])
                elif GRASP_METHOD == 'drone_only':
                    hover_des = HOVER[2].copy()
                    ramp_dur = 8.0
                    theta_des = smooth_ramp(phase_dt, 0, ramp_dur,
                                            _drone_only_arm_start, S2_JOINTS)
                else:
                    hover_des = HOVER[2].copy()
                    theta_des = np.array([0.0, 0.0])

            elif current_phase == 3:
                ramp_dur = 8.0
                if GRASP_METHOD == 'hybrid':
                    # Drone moves to HOVER[3], arm tracks live IK simultaneously
                    hover_des = smooth_ramp(phase_dt, 0, ramp_dur, hover_prev, HOVER[3])
                    ik_live = arm_ik(st['pos'], BOX_TARGET)
                    if ik_live is not None:
                        theta_des = np.array(ik_live)

                elif GRASP_METHOD == 'arm_only':
                    # Drone holds position, arm ramps from folded to IK solution
                    hover_des = hover_prev.copy()
                    theta_des = smooth_ramp(phase_dt, 0, ramp_dur,
                                            _arm_only_theta_start, _arm_only_theta_end)

                elif GRASP_METHOD == 'drone_only':
                    # Arm already at S2_JOINTS from phase 2; only drone moves
                    hover_des = smooth_ramp(phase_dt, 0, ramp_dur, hover_prev, _s2_hover)
                    theta_des = S2_JOINTS.copy()

            elif current_phase == 4:
                # GRASP: hold end-of-approach position, close gripper
                hover_des = hover_prev.copy()
                gripper_close = True
                # Hold arm angles from end of phase 3

            elif current_phase == 5:
                # LIFT: ramp hover z up
                ramp_dur = 3.0
                hover_des = smooth_ramp(phase_dt, 0, ramp_dur, hover_prev, HOVER[5])
                gripper_close = True
                box_pos = get_box_pos(model, data)
                # if phase_dt > 2.0 and phase_dt < 2.1:
                #     print(f'    Box z = {box_pos[2]:.3f} (started at 1.125)')

            elif current_phase == 6:
                # TRANSPORT: smooth ramp to drop-off
                ramp_dur = 4.0
                hover_des = smooth_ramp(phase_dt, 0, ramp_dur, hover_prev, HOVER[6])
                gripper_close = True

            elif current_phase == 7:
                # PLACE: descend, open gripper
                ramp_dur = 1.5
                hover_des = smooth_ramp(phase_dt, 0, ramp_dur, hover_prev, HOVER[7])
                gripper_close = False

            elif current_phase == 8:
                # RETRACT: fold arm, return home
                ramp_dur = 1.5
                hover_des = smooth_ramp(phase_dt, 0, ramp_dur, hover_prev, HOVER[8])
                if phase_dt < dt * 1.5:
                    _retract_start = st['theta'].copy()
                theta_des[0] = smooth_ramp(phase_dt, 0, ramp_dur, _retract_start[0], 0.0)
                theta_des[1] = smooth_ramp(phase_dt, 0, ramp_dur, _retract_start[1], 0.0)

            # --- Apply controls ---
            # Drone PID
            T, tau = ctrl.compute(
                st['pos'], st['vel'], st['quat'], st['omega'],
                hover_des, 0.0, dt)
            apply_platform_control(data, T, tau)

            # Arm PD + gravity feedforward
            arm_bias = np.array([data.qfrc_bias[_j1_dof], data.qfrc_bias[_j2_dof]])
            apply_arm_pd(data, theta_des, st['theta'], st['theta_dot'], bias=arm_bias)

            # Gripper
            apply_gripper(data, close=gripper_close)

            # Step
            mujoco.mj_step(model, data)

            # Real-time sync
            sim_elapsed = data.time - t0_sim
            wall_elapsed = time.perf_counter() - t0_wall
            if sim_elapsed > wall_elapsed:
                time.sleep(sim_elapsed - wall_elapsed)

            viewer.sync()

        # Final report
        box_pos = get_box_pos(model, data)
        print(f'\nDone. Final box position: {box_pos}')
        print('Close viewer to exit.')
        while viewer.is_running():
            time.sleep(0.05)

    plot_trajectories(log)
    plot_approach_phase(log)


if __name__ == '__main__':
    print('=== Grasp Task Demo ===')
    run_grasp()
