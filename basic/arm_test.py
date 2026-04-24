"""Arm joint performance test — drone base is fixed in space.

Run from workspace root:
    conda activate main
    python basic/arm_test.py

TEST_MODE controls the reference trajectory:
    'step'   - instantaneous step changes at specified times
    'sine'   - sinusoidal tracking for both joints
    'ramp'   - linearly interpolated waypoints
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(__file__))
from test_model import load_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEST_MODE      = 'ramp'          # 'step' | 'sine' | 'ramp'
SIM_DURATION   = 20.0            # seconds
DRONE_POS      = np.array([0.0, 0.0, 1.5])  # fixed base position [m]

# PD gains for joint torque control
KP_ARM         = 8.0             # N·m / rad
KD_ARM         = 0.8             # N·m·s / rad
USE_GRAVITY_FF = True            # add gravity compensation via qfrc_bias

# ---------------------------------------------------------------------------
# Reference profiles
# ---------------------------------------------------------------------------
# Step: list of (time_s, theta1_des_rad, theta2_des_rad)
STEP_WAYPOINTS = [
    ( 0.0,  0.0,  0.0),
    ( 3.0,  0.5,  0.0),
    ( 7.0,  0.5, -0.5),
    (12.0,  0.0, -0.5),
    (16.0,  0.0,  0.0),
]

# Sine: amplitude [rad] and frequency [Hz] per joint
SINE_AMP1  = 0.5
SINE_FREQ1 = 0.2
SINE_AMP2  = 0.4
SINE_FREQ2 = 0.3

# Ramp: same format as STEP — values are linearly interpolated
RAMP_WAYPOINTS = [
    ( 0.0,  0.0,  0.0),
    ( 5.0,  0.6,  0.0),
    (10.0,  0.6, -0.6),
    (18.0,  0.0,  0.0),
]

# ---------------------------------------------------------------------------
# Desired angle generators
# ---------------------------------------------------------------------------

def _step_desired(t):
    t1d, t2d = STEP_WAYPOINTS[0][1], STEP_WAYPOINTS[0][2]
    for tw, t1, t2 in STEP_WAYPOINTS:
        if t >= tw:
            t1d, t2d = t1, t2
    return t1d, t2d


def _sine_desired(t):
    t1d = SINE_AMP1 * np.sin(2.0 * np.pi * SINE_FREQ1 * t)
    t2d = SINE_AMP2 * np.sin(2.0 * np.pi * SINE_FREQ2 * t)
    return t1d, t2d


def _ramp_desired(t):
    wpts = RAMP_WAYPOINTS
    if t <= wpts[0][0]:
        return wpts[0][1], wpts[0][2]
    if t >= wpts[-1][0]:
        return wpts[-1][1], wpts[-1][2]
    for i in range(len(wpts) - 1):
        ta, t1a, t2a = wpts[i]
        tb, t1b, t2b = wpts[i + 1]
        if ta <= t < tb:
            s = (t - ta) / (tb - ta)
            return t1a + s * (t1b - t1a), t2a + s * (t2b - t2a)
    return wpts[-1][1], wpts[-1][2]


def desired(t):
    if TEST_MODE == 'step':
        return _step_desired(t)
    if TEST_MODE == 'sine':
        return _sine_desired(t)
    if TEST_MODE == 'ramp':
        return _ramp_desired(t)
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Drone locking
# ---------------------------------------------------------------------------

def fix_drone(data):
    """Override free-joint state so the drone is a rigid stand."""
    data.qpos[:3]  = DRONE_POS   # position
    data.qpos[3]   = 1.0         # quaternion w  (no rotation)
    data.qpos[4:7] = 0.0         # quaternion xyz
    data.qvel[:6]  = 0.0         # linear + angular velocity


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def main():
    model, data = load_model()

    # Joint DOF indices in qvel / qfrc_bias
    # free_joint contributes 6 DOFs → joint1 is index 6, joint2 is index 7
    J1_V = 6
    J2_V = 7

    # qpos indices
    J1_Q = 7
    J2_Q = 8

    fix_drone(data)
    mujoco.mj_forward(model, data)

    n_steps = int(SIM_DURATION / model.opt.timestep)

    log = {k: [] for k in ('t', 't1', 't2', 't1d', 't2d',
                            'tau1', 'tau2', 'e1', 'e2')}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Point camera at the arm
        viewer.cam.lookat[:] = DRONE_POS + np.array([0.1, 0.0, -0.2])
        viewer.cam.distance  = 1.2
        viewer.cam.elevation = -15

        for _ in range(n_steps):
            if not viewer.is_running():
                break

            # Fix drone before every physics step
            fix_drone(data)
            mujoco.mj_forward(model, data)   # recompute kinematics + bias forces

            t      = data.time
            theta1 = data.qpos[J1_Q]
            theta2 = data.qpos[J2_Q]
            dth1   = data.qvel[J1_V]
            dth2   = data.qvel[J2_V]

            t1_des, t2_des = desired(t)

            # PD torques
            tau1 = KP_ARM * (t1_des - theta1) - KD_ARM * dth1
            tau2 = KP_ARM * (t2_des - theta2) - KD_ARM * dth2

            # Gravity feedforward: qfrc_bias already accounts for full arm weight
            if USE_GRAVITY_FF:
                tau1 += data.qfrc_bias[J1_V]
                tau2 += data.qfrc_bias[J2_V]

            data.ctrl[0] = np.clip(tau1, -5.0,  5.0)
            data.ctrl[1] = np.clip(tau2, -5.0,  5.0)

            mujoco.mj_step(model, data)

            # Log at pre-step state
            log['t'].append(t)
            log['t1'].append(np.rad2deg(theta1))
            log['t2'].append(np.rad2deg(theta2))
            log['t1d'].append(np.rad2deg(t1_des))
            log['t2d'].append(np.rad2deg(t2_des))
            log['tau1'].append(float(data.ctrl[0]))
            log['tau2'].append(float(data.ctrl[1]))
            log['e1'].append(np.rad2deg(t1_des - theta1))
            log['e2'].append(np.rad2deg(t2_des - theta2))

            viewer.sync()

    plot_results(log)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(log):
    if not log['t']:
        print('No data logged.')
        return

    t = np.array(log['t'])

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(
        f'Arm Joint Test  |  mode={TEST_MODE}   KP={KP_ARM}   KD={KD_ARM}   '
        f'gravity_ff={USE_GRAVITY_FF}',
        fontsize=11
    )
    gs = gridspec.GridSpec(3, 2, hspace=0.50, wspace=0.35)

    # ── Joint 1 angle tracking ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, log['t1d'], 'r--', lw=1.5, label='desired')
    ax.plot(t, log['t1'],  color='tab:blue', lw=1.2, label='actual')
    ax.set_ylabel('deg')
    ax.set_title('Joint 1 angle')
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)

    # ── Joint 2 angle tracking ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, log['t2d'], 'r--', lw=1.5, label='desired')
    ax.plot(t, log['t2'],  color='tab:purple', lw=1.2, label='actual')
    ax.set_ylabel('deg')
    ax.set_title('Joint 2 angle')
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)

    # ── Tracking error joint 1 ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, log['e1'], color='tab:blue', lw=1.2)
    ax.axhline(0, color='k', lw=0.7, ls='--')
    ax.set_ylabel('deg')
    ax.set_title('Joint 1 tracking error')
    ax.grid(True, lw=0.4)

    # ── Tracking error joint 2 ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, log['e2'], color='tab:purple', lw=1.2)
    ax.axhline(0, color='k', lw=0.7, ls='--')
    ax.set_ylabel('deg')
    ax.set_title('Joint 2 tracking error')
    ax.grid(True, lw=0.4)

    # ── Control torques ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, log['tau1'], color='tab:blue',   lw=1.2, label='joint 1')
    ax.plot(t, log['tau2'], color='tab:purple', lw=1.2, label='joint 2')
    ax.axhline( 5.0, color='k', lw=0.6, ls=':')
    ax.axhline(-5.0, color='k', lw=0.6, ls=':')
    ax.axhline( 0,   color='k', lw=0.6, ls='--')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('N·m')
    ax.set_title('Control torques  (± 5 N·m limit)')
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)

    # ── Error phase portrait ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(log['e1'], log['e2'], color='tab:orange', lw=1.0)
    ax.scatter(log['e1'][0],  log['e2'][0],  color='green', s=60,
               zorder=5, label='start')
    ax.scatter(log['e1'][-1], log['e2'][-1], color='red',   s=60,
               marker='x', zorder=5, label='end')
    ax.set_xlabel('e₁ [deg]')
    ax.set_ylabel('e₂ [deg]')
    ax.set_title('Error phase portrait')
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4)

    out_path = os.path.join(os.path.dirname(__file__), 'arm_test_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Plot saved → {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
