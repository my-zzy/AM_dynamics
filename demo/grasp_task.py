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
KP_ARM = 1.0
KD_ARM = 0.3

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


def apply_arm_pd(data, theta_des, theta_cur, theta_dot):
    """PD torque control for arm joints."""
    tau = KP_ARM * (theta_des - theta_cur) - KD_ARM * theta_dot
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


def plot_trajectories(log):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    t     = np.array(log['t'])
    drone = np.array(log['drone_pos'])
    ee    = np.array(log['ee_pos'])
    box   = np.array(log['box_pos'])

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Grasp Task Trajectories', fontsize=13)
    gs = gridspec.GridSpec(3, 2, figure=fig)

    # 3-D trajectory (left column)
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    if PLOT_DRONE:
        ax3d.plot(drone[:, 0], drone[:, 1], drone[:, 2],
                  color='tab:blue', linewidth=1.5, label='Drone base')
        ax3d.scatter(*drone[0],  color='tab:blue', s=60, marker='o', zorder=5)
        ax3d.scatter(*drone[-1], color='tab:blue', s=60, marker='x', zorder=5)
    ax3d.plot(ee[:, 0], ee[:, 1], ee[:, 2],
              color='tab:orange', linewidth=1.5, label='Gripper EE')
    ax3d.scatter(*ee[0],  color='tab:orange', s=60, marker='o', zorder=5)
    ax3d.scatter(*ee[-1], color='tab:orange', s=60, marker='x', zorder=5)
    ax3d.plot(box[:, 0], box[:, 1], box[:, 2],
              color='tab:green', linewidth=1.2, linestyle='--', label='Box')
    ax3d.set_xlabel('x [m]')
    ax3d.set_ylabel('y [m]')
    ax3d.set_zlabel('z [m]')
    ax3d.legend(fontsize=8)
    ax3d.set_title('3D Trajectory  (o=start  x=end)')

    # Time series (right column)
    labels = ['x [m]', 'y [m]', 'z [m]']
    for i in range(3):
        ax = fig.add_subplot(gs[i, 1])
        if PLOT_DRONE:
            ax.plot(t, drone[:, i], color='tab:blue',   linewidth=1.2, label='Drone base')
        ax.plot(t, ee[:, i],    color='tab:orange', linewidth=1.2, label='Gripper EE')
        ax.plot(t, box[:, i],   color='tab:green',  linewidth=1.2, linestyle='--', label='Box')
        ax.set_ylabel(labels[i])
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, linewidth=0.4)
        if i == 2:
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
    (1,  9.0,  'INIT'),
    (2,  13.0, 'ARM READY'),
    (3,  17.0, 'APPROACH'),
    (4,  19.0, 'GRASP'),
    (5,  23.0, 'LIFT'),
    (6,  28.0, 'TRANSPORT'),
    (7,  30.0, 'PLACE'),
    (8,  32.0, 'RETRACT'),
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

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        t0_wall = time.perf_counter()
        t0_sim = data.time
        sim_duration = 17.0 # 34.0
        _retract_start = np.array([0.0, 0.0])

        # Trajectory logging
        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
        log = {'t': [], 'drone_pos': [], 'ee_pos': [], 'box_pos': []}

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

            st = get_grasp_state(model, data)
            phase_dt = t - phase_t0

            # Record trajectory
            log['t'].append(t)
            log['drone_pos'].append(st['pos'].copy())
            log['ee_pos'].append(data.site_xpos[ee_site_id].copy())
            log['box_pos'].append(get_box_pos(model, data))

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
                # ARM READY: hold arm at initial (L-config) angles, open gripper
                hover_des = HOVER[2].copy()
                theta_des = np.array([0.0, 0.0])

            elif current_phase == 3:
                # APPROACH: translate drone, ramp arm to box-level IK
                ramp_dur = 3.0
                hover_des = smooth_ramp(phase_dt, 0, ramp_dur, hover_prev, HOVER[3])
                # Recompute IK dynamically based on current base position
                ik_live = arm_ik(st['pos'], BOX_TARGET)
                if ik_live is not None:
                    theta_des = np.array(ik_live)
                # else hold previous

            elif current_phase == 4:
                # GRASP: hold position, close gripper
                hover_des = HOVER[4].copy()
                gripper_close = True
                # Hold arm angles from end of phase 3

            elif current_phase == 5:
                # LIFT: ramp hover z up
                ramp_dur = 3.0
                hover_des = smooth_ramp(phase_dt, 0, ramp_dur, hover_prev, HOVER[5])
                gripper_close = True
                box_pos = get_box_pos(model, data)
                if phase_dt > 2.0 and phase_dt < 2.1:
                    print(f'    Box z = {box_pos[2]:.3f} (started at 1.125)')

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

            # Arm PD
            apply_arm_pd(data, theta_des, st['theta'], st['theta_dot'])

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


if __name__ == '__main__':
    print('=== Grasp Task Demo ===')
    run_grasp()
