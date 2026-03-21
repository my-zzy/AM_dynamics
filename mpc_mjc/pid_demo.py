"""Real-time visual demo: PID position controller on the aerial manipulator.

Run from workspace root:
    conda activate mjc
    python mpc_mjc/pid_demo.py

Waypoint sequence (each held until settled or timeout):
    1. Take off   → [0, 0, 2.0] m
    2. Forward    → [2, 0, 2.0] m
    3. Sideways   → [2, 2, 2.0] m
    4. Climb      → [2, 2, 4.0] m
    5. Return     → [0, 0, 2.0] m
    6. Land       → [0, 0, 0.3] m
"""
import sys
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# Allow imports from mpc_mjc/ when running from workspace root
sys.path.insert(0, os.path.dirname(__file__))
from test_model import load_model, get_state, apply_platform_control
from pid_controller import DroneController


# ---------------------------------------------------------------------------
# Waypoints: (pos_des, yaw_des_deg, duration_s, label)
# ---------------------------------------------------------------------------
WAYPOINTS = [
    (np.array([0.0, 0.0, 2.0]),  0,  6,  'Take off → 2 m'),
    (np.array([2.0, 0.0, 2.0]),  0,  7,  'Forward  → x=2 m'),
    (np.array([2.0, 2.0, 2.0]),  0,  7,  'Sideways → y=2 m'),
    (np.array([2.0, 2.0, 4.0]),  0,  6,  'Climb    → z=4 m'),
    (np.array([0.0, 0.0, 2.0]),  0,  8,  'Return   → home'),
    (np.array([0.0, 0.0, 0.25]), 0,  6,  'Land'),
]


def run_demo():
    model, data = load_model()
    total_mass = np.sum(model.body_mass)
    print(f'Total mass: {total_mass:.3f} kg  (mg = {total_mass*9.81:.2f} N)')

    ctrl = DroneController(mass=total_mass)

    # Start on the ground
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.25
    mujoco.mj_forward(model, data)
    ctrl.reset()

    dt = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        for pos_des, yaw_deg, duration, label in WAYPOINTS:
            yaw_des = np.deg2rad(yaw_deg)
            print(f'\n--- {label}  →  pos={pos_des}  yaw={yaw_deg}°  ({duration} s) ---')

            t0_wall = time.perf_counter()
            t0_sim  = data.time

            while data.time - t0_sim < duration:
                if not viewer.is_running():
                    return

                x     = get_state(model, data)
                pos   = x[0:3]
                vel   = x[3:6]
                q     = x[6:10]
                omega = x[10:13]

                T, tau = ctrl.compute(pos, vel, q, omega, pos_des, yaw_des, dt)
                apply_platform_control(data, T, tau)
                mujoco.mj_step(model, data)

                # Real-time sync
                sim_elapsed  = data.time - t0_sim
                wall_elapsed = time.perf_counter() - t0_wall
                if sim_elapsed > wall_elapsed:
                    time.sleep(sim_elapsed - wall_elapsed)

                viewer.sync()

            x_final = get_state(model, data)
            err = np.linalg.norm(x_final[:3] - pos_des)
            print(f'    pos = [{x_final[0]:.3f}, {x_final[1]:.3f}, {x_final[2]:.3f}]'
                  f'  err = {err:.3f} m')

        print('\nAll waypoints done. Close viewer to exit.')
        while viewer.is_running():
            time.sleep(0.05)


if __name__ == '__main__':
    print('=== PID Position Controller Demo ===')
    run_demo()
