"""PID tuning script for the quadrotor body only (no robot arm).

This is a stripped-down version of pid_tuning.py that loads quad_only.xml
so you can tune position and attitude gains in isolation before adding the arm.

Two trajectory modes:
    p2p      - smooth cosine point-to-point
    figure8  - Lissajous figure-8 in the xz-plane

Usage (from workspace root, conda activate main):
    python basic/pid_tuning_quad.py --mode p2p
    python basic/pid_tuning_quad.py --mode figure8
"""
import sys
import os
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(__file__))
from pid_controller import DroneController, quat_to_euler_zyx

# ===========================================================================
#  PID GAINS  – edit these to tune
# ===========================================================================
GAINS = dict(
    kp_z    = 10.0,
    ki_z    =  2.0,
    kd_z    =  6.0,
    kp_xy   = 15.5,
    kd_xy   =  3.0,
    kp_rp   =  9.0,
    ki_rp   =  0.5,
    kd_rp   =  2.5,
    kp_yaw  =  5.0,
    ki_yaw  =  0.2,
    kd_yaw  =  1.5,
)

# ===========================================================================
#  POINT-TO-POINT CONFIG
# ===========================================================================
P2P_START    = np.array([0.0, 0.0, 0.3])
P2P_TARGET   = np.array([0.4, 1.5, 1.5])
P2P_DURATION = 5.0   # seconds to hold target
P2P_RAMP     = 5.0    # cosine ramp duration [s]

# ===========================================================================
#  FIGURE-8 CONFIG
# ===========================================================================
FIG8_CENTRE   = np.array([0.0, 0.0, 1.5])
FIG8_AMP_X    = 1.2
FIG8_AMP_Y    = 0.6
FIG8_AMP_Z    = 0.0
FIG8_PERIOD   = 12.0
FIG8_DURATION = 24.0

# ===========================================================================
#  LOGGING
# ===========================================================================
LOG_INTERVAL = 0.25   # console print interval [s]


# ---------------------------------------------------------------------------
#  Model helpers
# ---------------------------------------------------------------------------

def load_model():
    xml_path = os.path.join(os.path.dirname(__file__), 'model', 'quad_only.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    return model, data


def get_state(model, data):
    """Return [p(3), v(3), q_xyzw(4), omega(3)] for the free-joint quadrotor."""
    p     = data.qpos[:3].copy()
    v     = data.qvel[:3].copy()
    q_wxyz = data.qpos[3:7]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    omega  = data.qvel[3:6].copy()
    return np.concatenate([p, v, q_xyzw, omega])   # length 13


def apply_platform_control(data, model, T, tau_body):
    """Apply thrust T (body-z) and body-frame torques to the base body."""
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')
    R       = data.xmat[base_id].reshape(3, 3)
    data.xfrc_applied[base_id, :3] = R @ np.array([0.0, 0.0, T])
    data.xfrc_applied[base_id, 3:] = R @ np.asarray(tau_body, dtype=float)


# ---------------------------------------------------------------------------
#  Logging helpers
# ---------------------------------------------------------------------------

def make_log():
    return {'t': [], 'pos': [], 'pos_des': [], 'rpy': [], 'rpy_des': []}


def log_step(log, t, x, pos_des, ctrl, yaw_des=0.0):
    roll, pitch, yaw = quat_to_euler_zyx(x[6:10])
    log['t'].append(t)
    log['pos'].append(x[:3].copy())
    log['pos_des'].append(pos_des.copy())
    log['rpy'].append(np.array([roll, pitch, yaw]))
    log['rpy_des'].append(np.array([ctrl.roll_des, ctrl.pitch_des, yaw_des]))


def plot_results(log, title='Quad PID'):
    import matplotlib.pyplot as plt

    t     = np.array(log['t'])
    pos   = np.array(log['pos'])
    pos_d = np.array(log['pos_des'])
    rpy   = np.degrees(np.array(log['rpy']))
    rpy_d = np.degrees(np.array(log['rpy_des']))

    labels_pos = ['x [m]', 'y [m]', 'z [m]']
    labels_att = ['roll [deg]', 'pitch [deg]', 'yaw [deg]']

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(title, fontsize=13)

    for i in range(3):
        ax = axes[i]
        ax.plot(t, pos[:, i],   color='tab:blue',   label='actual')
        ax.plot(t, pos_d[:, i], color='tab:orange', linestyle='--', label='desired')
        ax.set_ylabel(labels_pos[i])
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, linewidth=0.4)

    for i in range(3):
        ax = axes[3 + i]
        ax.plot(t, rpy[:, i],   color='tab:green', label='actual')
        ax.plot(t, rpy_d[:, i], color='tab:red',   linestyle='--', label='desired')
        ax.set_ylabel(labels_att[i])
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, linewidth=0.4)

    axes[-1].set_xlabel('time [s]')
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__),
                            f'pid_quad_{title.lower().replace(" ", "_")}.png')
    plt.savefig(out_path, dpi=150)
    print(f'Plot saved → {out_path}')
    try:
        plt.show()
    except Exception:
        pass


def smooth_step(t, t0, duration):
    s = np.clip((t - t0) / duration, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * s))


def figure8_setpoint(t):
    w = 2.0 * np.pi / FIG8_PERIOD
    x = FIG8_CENTRE[0] + FIG8_AMP_X * np.sin(2.0 * w * t)
    y = FIG8_CENTRE[1] + FIG8_AMP_Y * np.sin(w * t)
    z = FIG8_CENTRE[2] + FIG8_AMP_Z * np.sin(w * t)
    return np.array([x, y, z])


# ---------------------------------------------------------------------------
#  Trajectory runners
# ---------------------------------------------------------------------------

def run_p2p(model, data, ctrl, dt):
    total_time = P2P_RAMP + P2P_DURATION
    t0_wall = time.perf_counter()
    t0_sim  = data.time
    last_log = -LOG_INTERVAL
    log = make_log()

    print(f'\n=== P2P  {P2P_START} → {P2P_TARGET}  '
          f'(ramp {P2P_RAMP:.1f}s, hold {P2P_DURATION:.1f}s) ===')

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while data.time - t0_sim < total_time:
            if not viewer.is_running():
                break

            t       = data.time - t0_sim
            s       = smooth_step(t, 0.0, P2P_RAMP)
            pos_des = P2P_START + s * (P2P_TARGET - P2P_START)

            x = get_state(model, data)
            T, tau = ctrl.compute(x[:3], x[3:6], x[6:10], x[10:13], pos_des, 0.0, dt)
            apply_platform_control(data, model, T, tau)
            mujoco.mj_step(model, data)
            log_step(log, t, x, pos_des, ctrl)

            if t - last_log >= LOG_INTERVAL:
                err   = np.linalg.norm(x[:3] - pos_des)
                phase = 'RAMP' if t < P2P_RAMP else 'HOLD'
                print(f'  t={t:6.2f}s  [{phase}]  '
                      f'pos=[{x[0]:.3f},{x[1]:.3f},{x[2]:.3f}]  '
                      f'des=[{pos_des[0]:.3f},{pos_des[1]:.3f},{pos_des[2]:.3f}]  '
                      f'err={err:.4f}m')
                last_log = t

            sim_elapsed  = data.time - t0_sim
            wall_elapsed = time.perf_counter() - t0_wall
            if sim_elapsed > wall_elapsed:
                time.sleep(sim_elapsed - wall_elapsed)
            viewer.sync()

        x_final   = get_state(model, data)
        final_err = np.linalg.norm(x_final[:3] - P2P_TARGET)
        print(f'\nFinal pos=[{x_final[0]:.4f},{x_final[1]:.4f},{x_final[2]:.4f}]  '
              f'err={final_err:.4f}m')
        print('Close viewer to exit.')
        while viewer.is_running():
            time.sleep(0.05)

    plot_results(log, title='P2P')
    return log


def run_figure8(model, data, ctrl, dt):
    RAMP_DUR = 3.0
    t0_wall  = time.perf_counter()
    t0_sim   = data.time
    last_log = -LOG_INTERVAL
    log      = make_log()
    rmse_sum = 0.0
    rmse_n   = 0

    print(f'\n=== Figure-8 (horizontal)  centre={FIG8_CENTRE}  '
          f'Ax={FIG8_AMP_X} Ay={FIG8_AMP_Y}  T={FIG8_PERIOD:.1f}s ===')

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while data.time - t0_sim < RAMP_DUR + FIG8_DURATION:
            if not viewer.is_running():
                break

            t = data.time - t0_sim
            x = get_state(model, data)

            if t < RAMP_DUR:
                s       = smooth_step(t, 0.0, RAMP_DUR)
                pos_des = x[:3] + s * (FIG8_CENTRE - x[:3])
            else:
                pos_des = figure8_setpoint(t - RAMP_DUR)

            T, tau = ctrl.compute(x[:3], x[3:6], x[6:10], x[10:13], pos_des, 0.0, dt)
            apply_platform_control(data, model, T, tau)
            mujoco.mj_step(model, data)
            log_step(log, t, x, pos_des, ctrl)

            if t >= RAMP_DUR:
                rmse_sum += np.sum((x[:3] - pos_des) ** 2)
                rmse_n   += 1

            if t - last_log >= LOG_INTERVAL:
                err   = np.linalg.norm(x[:3] - pos_des)
                phase = 'RAMP' if t < RAMP_DUR else f'FIG-8 t={t-RAMP_DUR:.1f}s'
                print(f'  t={t:6.2f}s  [{phase}]  '
                      f'pos=[{x[0]:.3f},{x[1]:.3f},{x[2]:.3f}]  '
                      f'des=[{pos_des[0]:.3f},{pos_des[1]:.3f},{pos_des[2]:.3f}]  '
                      f'err={err:.4f}m')
                last_log = t

            sim_elapsed  = data.time - t0_sim
            wall_elapsed = time.perf_counter() - t0_wall
            if sim_elapsed > wall_elapsed:
                time.sleep(sim_elapsed - wall_elapsed)
            viewer.sync()

        rmse = np.sqrt(rmse_sum / rmse_n) if rmse_n > 0 else float('nan')
        print(f'\nTracking RMSE over figure-8: {rmse:.4f} m')
        print('Close viewer to exit.')
        while viewer.is_running():
            time.sleep(0.05)

    plot_results(log, title='Figure-8')
    return log


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Quadrotor-only PID tuning')
    parser.add_argument('--mode', choices=['p2p', 'figure8'], default='p2p')
    args = parser.parse_args()

    model, data = load_model()
    total_mass  = np.sum(model.body_mass)
    print(f'Model: quad_only.xml   mass: {total_mass:.3f} kg')
    print(f'Gains: {GAINS}')

    ctrl = DroneController(mass=total_mass, **GAINS)

    mujoco.mj_resetData(model, data)
    if args.mode == 'p2p':
        data.qpos[0] = P2P_START[0]
        data.qpos[1] = P2P_START[1]
        data.qpos[2] = P2P_START[2]
    else:
        data.qpos[0] = FIG8_CENTRE[0]
        data.qpos[1] = FIG8_CENTRE[1]
        data.qpos[2] = FIG8_CENTRE[2]
    mujoco.mj_forward(model, data)
    ctrl.reset()

    dt = model.opt.timestep

    if args.mode == 'p2p':
        run_p2p(model, data, ctrl, dt)
    else:
        run_figure8(model, data, ctrl, dt)


if __name__ == '__main__':
    main()
