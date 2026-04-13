"""arm_effect.py – Arm-swing influence on aerial manipulator base.

Hover strategy
--------------
A constant upward force  F = total_mass * g  is applied as xfrc_applied on
the base body (world frame, +z direction).  Zero torque is applied to the
base.  No position/attitude feedback – the platform drifts freely under the
reaction forces caused by the moving arm.

Arm motion
----------
Joint-1 (shoulder, y-axis hinge) is driven to track a sinusoidal reference
angle via a simple PD law on the joint motor.  Joint-2 (elbow) is kept at 0°.

Outputs
-------
  • Real-time MuJoCo viewer
  • After the viewer closes: three-panel matplotlib figure + PNG saved to
    mpc_mjc/arm_effect.png

Run from workspace root:
    conda activate mjc
    python mpc_mjc/arm_effect.py
"""

import sys
import os
import time

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from test_model import load_model, get_state, apply_platform_control
from pid_controller import DroneController

# ─────────────────────────────────────────────────────────────────────────────
# Simulation parameters
# ─────────────────────────────────────────────────────────────────────────────
SIM_DURATION = 20.0   # s  – total simulation time

# ── Arm joint waypoints (degrees) ────────────────────────────────────────────
# Set START to the resting pose and END to the desired final pose.
# Both joints are held at START until MOTION_START_S, then smoothly
# driven to END over MOTION_DURATION_S seconds and held there.
JOINT1_START_DEG  =   0.0   # joint-1 start angle  [°]
JOINT1_END_DEG    =   -30.0   # joint-1 end   angle  [°]
JOINT2_START_DEG  =   30.0   # joint-2 start angle  [°]
JOINT2_END_DEG    =  30.0   # joint-2 end   angle  [°]

MOTION_START_S    =  5.0    # time before arm starts moving  [s]  (let drone stabilise)
MOTION_DURATION_S = 10.0   # duration of the start→end sweep  [s]

# PD gains for arm joint tracking
ARM_KP = 4.0
ARM_KD = 0.

# Fixed hover position for the drone base (PID setpoint)
HOVER_POS = np.array([0.0, 0.0, 1.])   # x, y, z  [m]
HOVER_YAW = 0.0                          # [rad]

LOG_EVERY_N_STEPS = 10   # downsample logs (every 10 × 2 ms = 20 ms)


# ─────────────────────────────────────────────────────────────────────────────
def run():
    model, data = load_model()

    # ── System info ───────────────────────────────────────────────────
    total_mass = float(np.sum(model.body_mass))
    g          = float(abs(model.opt.gravity[2]))   # 9.81 m s⁻²

    print(f'Body masses (kg):')
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f'  [{i}] {name:12s}  {model.body_mass[i]:.4f} kg')
    print(f'Total mass : {total_mass:.4f} kg')
    print(f'Gravity    : {g:.4f} m s⁻²')
    print(f'Hover setpoint: {HOVER_POS}  yaw={HOVER_YAW} rad')

    # ── IDs ──────────────────────────────────────────────────────────────
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')

    # ── PID controller for the drone base ──────────────────────────────
    ctrl = DroneController(mass=total_mass)
    ctrl.reset()

    # ── Reset to hover height ────────────────────────────────────────────
    mujoco.mj_resetData(model, data)
    data.qpos[2] = HOVER_POS[2]  # start at hover height
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep   # 0.002 s

    # Pre-convert joint waypoints to radians
    j1_start = np.deg2rad(JOINT1_START_DEG)
    j1_end   = np.deg2rad(JOINT1_END_DEG)
    j2_start = np.deg2rad(JOINT2_START_DEG)
    j2_end   = np.deg2rad(JOINT2_END_DEG)

    # ── Log buffers ───────────────────────────────────────────────────────
    log_t      = []   # simulation time
    log_pos    = []   # drone base xyz
    log_ee_pos = []   # end-effector world xyz
    log_theta  = []   # [theta1, theta2] in rad
    step_idx   = 0

    # ── Run ───────────────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        t_wall0 = time.perf_counter()

        while data.time < SIM_DURATION:
            if not viewer.is_running():
                break

            t = data.time

            # ── Platform: PID hover at fixed position ────────────────
            x     = get_state(model, data)
            pos   = x[0:3]
            vel   = x[3:6]
            q     = x[6:10]
            omega_b = x[10:13]
            T, tau = ctrl.compute(pos, vel, q, omega_b, HOVER_POS, HOVER_YAW, dt)
            apply_platform_control(data, T, tau)

            # ── Arm: smooth ramp from start to end angles ─────────────
            # Normalised motion time; clamp to [0, 1]
            t_motion = t - MOTION_START_S
            if t_motion <= 0.0:
                alpha, alpha_dot = 0.0, 0.0
            elif t_motion >= MOTION_DURATION_S:
                alpha, alpha_dot = 1.0, 0.0
            else:
                tn        = t_motion / MOTION_DURATION_S          # [0, 1]
                alpha     = 3*tn**2 - 2*tn**3                     # smooth-step
                alpha_dot = (6*tn - 6*tn**2) / MOTION_DURATION_S  # d(alpha)/dt

            theta1_des     = j1_start + alpha     * (j1_end - j1_start)
            theta1_dot_des =            alpha_dot * (j1_end - j1_start)
            theta2_des     = j2_start + alpha     * (j2_end - j2_start)
            theta2_dot_des =            alpha_dot * (j2_end - j2_start)

            theta1     = data.qpos[7]
            theta1_dot = data.qvel[6]
            theta2     = data.qpos[8]
            theta2_dot = data.qvel[7]

            data.ctrl[0] = (ARM_KP * (theta1_des - theta1)
                            - ARM_KD * (theta1_dot - theta1_dot_des))
            data.ctrl[1] = ARM_KP * (theta2_des - theta2) - ARM_KD * (theta2_dot - theta2_dot_des)

            mujoco.mj_step(model, data)
            step_idx += 1

            # ── Log ────────────────────────────────────────────────────
            if step_idx % LOG_EVERY_N_STEPS == 0:
                log_t.append(data.time)
                log_pos.append(data.qpos[:3].copy())
                log_ee_pos.append(data.site_xpos[ee_id].copy())
                log_theta.append([data.qpos[7], data.qpos[8]])

            # ── Real-time sync ─────────────────────────────────────────
            wall_elapsed = time.perf_counter() - t_wall0
            if data.time > wall_elapsed:
                time.sleep(data.time - wall_elapsed)

            viewer.sync()

        print('\nSimulation complete.  Close the viewer window to generate plots.')
        while viewer.is_running():
            time.sleep(0.05)

    # ─────────────────────────────────────────────────────────────────────
    # Post-process & plot
    # ─────────────────────────────────────────────────────────────────────
    t_arr   = np.array(log_t)
    pos_arr = np.array(log_pos)
    ee_arr  = np.array(log_ee_pos)
    th_arr  = np.array(log_theta)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(
        'Effect of Arm Motion on Aerial Manipulator Base  (PID hover)\n'
        f'J1: {JOINT1_START_DEG:.0f}°→{JOINT1_END_DEG:.0f}°   '
        f'J2: {JOINT2_START_DEG:.0f}°→{JOINT2_END_DEG:.0f}°   '
        f'start={MOTION_START_S:.1f}s  dur={MOTION_DURATION_S:.1f}s',
        fontsize=12,
    )

    # ── Panel 1: drone base position ─────────────────────────────────────
    ax = axes[0]
    ax.plot(t_arr, pos_arr[:, 0], label='x')
    ax.plot(t_arr, pos_arr[:, 1], label='y')
    ax.plot(t_arr, pos_arr[:, 2]-np.ones_like(pos_arr[:, 2]), label='z')
    ax.set_ylabel('Position  (m)')
    ax.set_title('Drone Base Position')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.4)

    # ── Panel 2: end-effector world position ─────────────────────────────
    ax = axes[1]
    ax.plot(t_arr, ee_arr[:, 0], label='x')
    ax.plot(t_arr, ee_arr[:, 1], label='y')
    ax.plot(t_arr, ee_arr[:, 2]-np.ones_like(pos_arr[:, 2]), label='z')
    ax.set_ylabel('Position  (m)')
    ax.set_title('End-Effector World Position')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.4)

    # ── Panel 3: joint angles ─────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t_arr, np.rad2deg(th_arr[:, 0]), label='θ₁ actual', linewidth=1.8)
    # ax.plot(t_arr, th1_ref,                  label='θ₁ desired', linestyle='--', alpha=0.6)
    ax.plot(t_arr, np.rad2deg(th_arr[:, 1]), label='θ₂ actual')
    ax.set_ylabel('Angle  (°)')
    ax.set_title('Arm Joint Angles')
    ax.set_xlabel('Time  (s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'arm_effect.png')
    plt.savefig(out_path, dpi=150)
    print(f'Plot saved → {out_path}')
    # plt.show()


if __name__ == '__main__':
    print('=== Arm-Sweep Effect Demo ===')
    run()
