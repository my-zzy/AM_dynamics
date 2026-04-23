"""MPC-controlled grasp task for the Aerial Manipulator.

Replaces the decoupled PID + arm-PD approach in grasp_task.py with a
unified NMPC controller (see demo/mpc_controller.py) that treats the
full coupled Newton-Euler system as a single plant.

Phases
------
  0  TAKEOFF   PID ascent to hover height (arm folded, NMPC not yet active)
  1  HOVER      Stabilise at hover; NMPC activated, arm-down hold trajectory
  2  MPC REACH  NMPC drives EE from current position to grasp point
  3  GRASP      Hold EE at grasp point; close gripper
  4  LIFT        PID raises platform; NMPC holds arm; gripper closed
  5  TRANSPORT  PID flies to drop-off; NMPC holds arm; gripper closed
  6  PLACE      Descend; open gripper
  7  RETRACT    Arm folds back to zero; return home

Run from workspace root:
    conda activate main
    python demo/mpc_grasp_task.py

Flags:
    --no-terminal-constraint   Disable hard terminal equality (softer problem)
    --rebuild                  Force recompile of acados C code
    --no-viewer                Run headless (no window)
    --dt-mpc=0.05              MPC shooting step size (seconds)
    --horizon=20               MPC prediction horizon steps
"""

import sys
import os
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer

# Allow imports from project root and basic/
_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'basic'))

from test_model import apply_platform_control
from pid_controller import DroneController
from ams.model import AerialManipulatorModel
from ams.kinematics import forward_kinematics

# grasp_task helpers (re-used)
from demo.grasp_task import (
    arm_ik, arm_fk, smooth_ramp,
    get_grasp_state, get_box_pos, apply_gripper,
    load_grasp_scene,
    GROUND_Z, BOX_TARGET, HOVER, CTRL_J1, CTRL_J2,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Platform hover position before MPC reach phase
HOVER_PRE_MPC = np.array([0.25, 0.0, 1.45])

# MPC trajectory parameters
MPC_TRAJ_TF   = 4.0    # trajectory duration (seconds)

# Terminal detection thresholds
EE_POS_TOL  = 0.015   # 15 mm  position error to declare "arrived"
EE_VEL_TOL  = 0.05    # 5 cm/s EE velocity to declare "stopped"

# After grasp, lift target
LIFT_Z = 1.65


# ---------------------------------------------------------------------------
# State packing / unpacking helpers
# ---------------------------------------------------------------------------

def pack_state(st):
    """Pack get_grasp_state dict -> (17,) state vector for MPC."""
    return np.concatenate([
        st['pos'],        # 0:3   p_A
        st['vel'],        # 3:6   v_A
        st['quat'],       # 6:10  q_A [x,y,z,w]
        st['omega'],      # 10:13 omega_A  (body frame in MuJoCo)
        st['theta'],      # 13:15 joint angles
        st['theta_dot'],  # 15:17 joint velocities
    ])


def ee_pos_from_state(am_model, st):
    """Compute world-frame EE position directly from AMS model."""
    R, p, _ = forward_kinematics(
        am_model, st['quat'], st['pos'], st['theta'])
    return p[3]   # frame {3} = end-effector


def apply_mpc_control(model, data, u0):
    """Inject MPC output u0 = [F_ext(3), tau_ext(3), tau_j(2)] into MuJoCo.

    Platform forces/torques are applied as body external wrench on the 'base'
    body in world frame. Joint torques are written to ctrl[0:2].
    """
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')

    F_ext   = u0[0:3]
    tau_ext = u0[3:6]
    tau_j   = u0[6:8]

    # xfrc_applied layout: [fx, fy, fz, tx, ty, tz] in world frame
    data.xfrc_applied[base_id, :] = np.concatenate([F_ext, tau_ext])

    # Joint torques
    data.ctrl[CTRL_J1] = float(np.clip(tau_j[0], -5.0, 5.0))
    data.ctrl[CTRL_J2] = float(np.clip(tau_j[1], -5.0, 5.0))


def clear_mpc_forces(model, data):
    """Zero out external body wrench applied by MPC."""
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')
    data.xfrc_applied[base_id, :] = 0.0


# ---------------------------------------------------------------------------
# PD arm hold (used outside MPC phases)
# ---------------------------------------------------------------------------

KP_ARM = 1.0
KD_ARM = 0.3


def apply_arm_pd(data, theta_des, theta_cur, theta_dot):
    tau = KP_ARM * (theta_des - theta_cur) - KD_ARM * theta_dot
    tau = np.clip(tau, -5.0, 5.0)
    data.ctrl[CTRL_J1] = tau[0]
    data.ctrl[CTRL_J2] = tau[1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_mpc_grasp(dt_mpc=0.05, N=20, rebuild=False,
                  enable_terminal_constraint=True,
                  use_viewer=True):

    mj_model, mj_data = load_grasp_scene()
    am_model = AerialManipulatorModel()

    # Robot mass (exclude world, pedestal, box)
    _exclude = {'world', 'pedestal', 'box'}
    robot_ids = [i for i in range(mj_model.nbody)
                 if mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
                 not in _exclude]
    total_mass = sum(mj_model.body_mass[i] for i in robot_ids)
    print(f'Robot mass: {total_mass:.3f} kg')

    # PID for platform (used during non-MPC phases)
    pid = DroneController(mass=total_mass)

    sim_dt = mj_model.opt.timestep   # 0.002 s
    mpc_step_counter = 0
    mpc_steps_per_solve = max(1, int(round(dt_mpc / sim_dt)))  # solve every N sim steps

    # Reset
    mujoco.mj_resetData(mj_model, mj_data)
    base_jnt_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'free_joint')
    mj_data.qpos[mj_model.jnt_qposadr[base_jnt_id] + 2] = GROUND_Z
    mujoco.mj_forward(mj_model, mj_data)
    pid.reset()

    # Phase state machine
    PHASES = {
        0: 'TAKEOFF',
        1: 'HOVER',
        2: 'MPC_REACH',
        3: 'GRASP',
        4: 'LIFT',
        5: 'TRANSPORT',
        6: 'PLACE',
        7: 'RETRACT',
    }
    # Phase end times (seconds from simulation start)
    PHASE_END = {
        0: 5.0,    # takeoff
        1: 9.0,    # hover stabilise
        2: None,   # MPC phase ends when EE arrives  (or max 14 s)
        3: None,   # grasp phase ends after 2 s hold
        4: None,   # lift ends after 4 s
        5: None,   # transport ends after 6 s
        6: None,   # place ends after 2 s
        7: None,   # retract ends after 3 s
    }
    MAX_PHASE_DUR = {2: 14.0, 3: 2.0, 4: 4.0, 5: 6.0, 6: 2.0, 7: 3.0}

    phase          = 0
    phase_t0       = 0.0
    hover_des      = np.array([0.0, 0.0, GROUND_Z])
    hover_prev     = hover_des.copy()
    theta_des      = np.zeros(2)
    gripper_close  = False
    mpc_ctrl       = None    # built on entry to phase 2
    mpc_traj       = None
    mpc_t0         = 0.0     # simulation time when MPC phase started
    last_u0        = None    # cached MPC output between solves
    _retract_start = np.zeros(2)

    # Logging for phase 2
    ee_errors   = []
    solve_times = []

    def enter_phase(new_phase, t, st):
        nonlocal phase, phase_t0, hover_prev, hover_des
        nonlocal theta_des, gripper_close
        nonlocal mpc_ctrl, mpc_traj, mpc_t0, last_u0
        nonlocal _retract_start

        phase    = new_phase
        phase_t0 = t
        hover_prev = st['pos'].copy()
        hover_prev[1] = 0.0
        print(f'\n[t={t:.2f}s] === Phase {new_phase}: {PHASES[new_phase]} ===')

        if new_phase == 2:
            # Build MPC trajectory: current EE → grasp point
            from demo.mpc_trajectory import EETrajectory
            from demo.mpc_controller import MPCController

            p_ee_now = ee_pos_from_state(am_model, st)
            print(f'  EE start: {p_ee_now}')
            print(f'  EE target: {BOX_TARGET}')

            mpc_traj = EETrajectory(
                p_start=p_ee_now,
                p_end=BOX_TARGET,
                T_f=MPC_TRAJ_TF,
                dt=dt_mpc,
            )
            print(f'  Trajectory: {MPC_TRAJ_TF:.1f} s, {mpc_traj.N} steps @ {dt_mpc*1000:.0f} ms')

            print('  Building MPCController (may compile acados C code) ...')
            t_build_start = time.perf_counter()
            mpc_ctrl = MPCController(
                traj=mpc_traj,
                model=am_model,
                N=N,
                dt=dt_mpc,
                rebuild=rebuild,
                enable_terminal_constraint=enable_terminal_constraint,
            )
            t_build = time.perf_counter() - t_build_start
            print(f'  MPCController ready in {t_build:.1f} s')
            mpc_t0 = t
            last_u0 = None

        elif new_phase == 3:
            gripper_close = False    # will be set True after 0.3 s

        elif new_phase == 7:
            _retract_start = st['theta'].copy()

    def advance_phase(t, st):
        """Check if current phase is done and transition."""
        elapsed = t - phase_t0
        if phase == 0 and t >= PHASE_END[0]:
            enter_phase(1, t, st)
        elif phase == 1 and t >= PHASE_END[1]:
            enter_phase(2, t, st)
        elif phase == 2:
            # Terminate on arrival or timeout
            p_ee = ee_pos_from_state(am_model, st)
            ee_err = np.linalg.norm(p_ee - BOX_TARGET)
            v_ee   = np.linalg.norm(st['vel'])
            arrived = (ee_err < EE_POS_TOL and v_ee < EE_VEL_TOL)
            timed_out = (elapsed >= MAX_PHASE_DUR[2])
            if arrived or timed_out:
                reason = 'arrived' if arrived else 'timeout'
                print(f'  Phase 2 ended ({reason}): '
                      f'ee_err={ee_err*1000:.1f} mm  v={v_ee*100:.1f} cm/s')
                enter_phase(3, t, st)
        elif phase == 3 and elapsed >= MAX_PHASE_DUR[3]:
            enter_phase(4, t, st)
        elif phase == 4 and elapsed >= MAX_PHASE_DUR[4]:
            enter_phase(5, t, st)
        elif phase == 5 and elapsed >= MAX_PHASE_DUR[5]:
            enter_phase(6, t, st)
        elif phase == 6 and elapsed >= MAX_PHASE_DUR[6]:
            enter_phase(7, t, st)

    # Total sim time
    total_sim_time = (PHASE_END[0] + PHASE_END[1]
                      + MAX_PHASE_DUR[2] + MAX_PHASE_DUR[3]
                      + MAX_PHASE_DUR[4] + MAX_PHASE_DUR[5]
                      + MAX_PHASE_DUR[6] + MAX_PHASE_DUR[7] + 2.0)

    def sim_loop(viewer_ctx):
        nonlocal mpc_step_counter, hover_des, theta_des, gripper_close, last_u0

        t0_wall = time.perf_counter()
        t0_sim  = mj_data.time

        while mj_data.time - t0_sim < total_sim_time:
            if viewer_ctx is not None and not viewer_ctx.is_running():
                return

            t  = mj_data.time - t0_sim
            st = get_grasp_state(mj_model, mj_data)

            # Phase transitions
            advance_phase(t, st)

            elapsed = t - phase_t0

            # ----------------------------------------------------------------
            # Phase 0: TAKEOFF — PID + folded arm
            # ----------------------------------------------------------------
            if phase == 0:
                ramp_dur = 4.0
                hover_des = smooth_ramp(elapsed, 0, ramp_dur,
                                        np.array([0.0, 0.0, GROUND_Z]),
                                        HOVER_PRE_MPC)
                theta_des     = np.zeros(2)
                gripper_close = False
                clear_mpc_forces(mj_model, mj_data)

                T, tau = pid.compute(st['pos'], st['vel'], st['quat'], st['omega'],
                                     hover_des, 0.0, sim_dt)
                apply_platform_control(mj_data, T, tau)
                apply_arm_pd(mj_data, theta_des, st['theta'], st['theta_dot'])
                apply_gripper(mj_data, close=False)

            # ----------------------------------------------------------------
            # Phase 1: HOVER — PID, approach setpoint; arm folds
            # ----------------------------------------------------------------
            elif phase == 1:
                ramp_dur = 3.0
                hover_des = smooth_ramp(elapsed, 0, ramp_dur,
                                        hover_prev, HOVER_PRE_MPC)
                theta_des     = np.zeros(2)
                gripper_close = False
                clear_mpc_forces(mj_model, mj_data)

                T, tau = pid.compute(st['pos'], st['vel'], st['quat'], st['omega'],
                                     hover_des, 0.0, sim_dt)
                apply_platform_control(mj_data, T, tau)
                apply_arm_pd(mj_data, theta_des, st['theta'], st['theta_dot'])
                apply_gripper(mj_data, close=False)

            # ----------------------------------------------------------------
            # Phase 2: MPC REACH — NMPC drives EE to grasp point
            # ----------------------------------------------------------------
            elif phase == 2:
                t_mpc = t - mpc_t0   # time within MPC trajectory

                # Run NMPC solve at dt_mpc rate
                if mpc_step_counter % mpc_steps_per_solve == 0:
                    x_now = pack_state(st)
                    try:
                        u0_new, info = mpc_ctrl.solve(x_now, t_mpc)
                        last_u0 = u0_new
                        solve_times.append(info['solve_time'])
                        if mpc_step_counter % (mpc_steps_per_solve * 20) == 0:
                            p_ee = ee_pos_from_state(am_model, st)
                            ee_err = np.linalg.norm(p_ee - BOX_TARGET)
                            ee_errors.append(ee_err)
                            print(f'  [t={t:.2f}s] ee_err={ee_err*1000:.1f} mm  '
                                  f'solve={info["solve_time"]*1000:.2f} ms  '
                                  f'status={info["status"]}')
                    except Exception as exc:
                        print(f'  [t={t:.2f}s] MPC solve failed: {exc}')
                        # Fall back to last valid u0

                mpc_step_counter += 1

                if last_u0 is not None:
                    apply_mpc_control(mj_model, mj_data, last_u0)
                else:
                    # Gravity compensation fallback while waiting for first solve
                    g   = 9.81
                    m   = am_model.total_mass
                    u_hover = np.zeros(8)
                    u_hover[2] = m * g
                    apply_mpc_control(mj_model, mj_data, u_hover)

                apply_gripper(mj_data, close=False)

            # ----------------------------------------------------------------
            # Phase 3: GRASP — hold last MPC u0, close gripper after 0.3 s
            # ----------------------------------------------------------------
            elif phase == 3:
                gripper_close = elapsed > 0.3

                # Keep NMPC hold (frozen terminal reference = grasp point)
                if last_u0 is not None and mpc_ctrl is not None:
                    if mpc_step_counter % mpc_steps_per_solve == 0:
                        x_now = pack_state(st)
                        t_mpc = mpc_ctrl.traj.T_f   # clamp at terminal
                        try:
                            u0_new, _ = mpc_ctrl.solve(x_now, t_mpc)
                            last_u0 = u0_new
                        except Exception:
                            pass
                    mpc_step_counter += 1
                    apply_mpc_control(mj_model, mj_data, last_u0)
                else:
                    clear_mpc_forces(mj_model, mj_data)
                    T, tau = pid.compute(st['pos'], st['vel'], st['quat'], st['omega'],
                                         HOVER_PRE_MPC, 0.0, sim_dt)
                    apply_platform_control(mj_data, T, tau)
                    apply_arm_pd(mj_data, np.zeros(2), st['theta'], st['theta_dot'])

                apply_gripper(mj_data, close=gripper_close)

            # ----------------------------------------------------------------
            # Phase 4: LIFT — PID raises platform; arm held via PD; gripper closed
            # ----------------------------------------------------------------
            elif phase == 4:
                ramp_dur = 3.0
                hover_des = smooth_ramp(elapsed, 0, ramp_dur,
                                        hover_prev,
                                        np.array([HOVER_PRE_MPC[0], 0.0, LIFT_Z]))
                clear_mpc_forces(mj_model, mj_data)

                T, tau = pid.compute(st['pos'], st['vel'], st['quat'], st['omega'],
                                     hover_des, 0.0, sim_dt)
                apply_platform_control(mj_data, T, tau)

                # Hold arm IK angles for box
                ik = arm_ik(st['pos'], BOX_TARGET)
                if ik is not None:
                    theta_des = np.array(ik)
                apply_arm_pd(mj_data, theta_des, st['theta'], st['theta_dot'])
                apply_gripper(mj_data, close=True)

            # ----------------------------------------------------------------
            # Phase 5: TRANSPORT — PID to drop-off; arm PD; gripper closed
            # ----------------------------------------------------------------
            elif phase == 5:
                ramp_dur = 4.0
                hover_des = smooth_ramp(elapsed, 0, ramp_dur,
                                        hover_prev, HOVER[6])   # HOVER[6] = drop-off
                clear_mpc_forces(mj_model, mj_data)

                T, tau = pid.compute(st['pos'], st['vel'], st['quat'], st['omega'],
                                     hover_des, 0.0, sim_dt)
                apply_platform_control(mj_data, T, tau)
                apply_arm_pd(mj_data, theta_des, st['theta'], st['theta_dot'])
                apply_gripper(mj_data, close=True)

            # ----------------------------------------------------------------
            # Phase 6: PLACE — descend; open gripper
            # ----------------------------------------------------------------
            elif phase == 6:
                ramp_dur = 1.5
                hover_des = smooth_ramp(elapsed, 0, ramp_dur,
                                        hover_prev, HOVER[7])
                clear_mpc_forces(mj_model, mj_data)

                T, tau = pid.compute(st['pos'], st['vel'], st['quat'], st['omega'],
                                     hover_des, 0.0, sim_dt)
                apply_platform_control(mj_data, T, tau)
                apply_arm_pd(mj_data, theta_des, st['theta'], st['theta_dot'])
                apply_gripper(mj_data, close=(elapsed < 0.5))

            # ----------------------------------------------------------------
            # Phase 7: RETRACT — fold arm, fly home
            # ----------------------------------------------------------------
            elif phase == 7:
                ramp_dur = 2.0
                hover_des = smooth_ramp(elapsed, 0, ramp_dur,
                                        hover_prev, np.array([0.0, 0.0, 1.5]))
                theta_des[0] = smooth_ramp(elapsed, 0, ramp_dur, _retract_start[0], 0.0)
                theta_des[1] = smooth_ramp(elapsed, 0, ramp_dur, _retract_start[1], 0.0)
                clear_mpc_forces(mj_model, mj_data)

                T, tau = pid.compute(st['pos'], st['vel'], st['quat'], st['omega'],
                                     hover_des, 0.0, sim_dt)
                apply_platform_control(mj_data, T, tau)
                apply_arm_pd(mj_data, theta_des, st['theta'], st['theta_dot'])
                apply_gripper(mj_data, close=False)

            # ----------------------------------------------------------------
            # Advance simulation
            # ----------------------------------------------------------------
            mujoco.mj_step(mj_model, mj_data)

            # Real-time sync
            sim_elapsed  = mj_data.time - t0_sim
            wall_elapsed = time.perf_counter() - t0_wall
            if sim_elapsed > wall_elapsed and viewer_ctx is not None:
                time.sleep(sim_elapsed - wall_elapsed)

            if viewer_ctx is not None:
                viewer_ctx.sync()

        # End of simulation
        box_pos = get_box_pos(mj_model, mj_data)
        st_final = get_grasp_state(mj_model, mj_data)
        p_ee_final = ee_pos_from_state(am_model, st_final)
        ee_final_err = np.linalg.norm(p_ee_final - BOX_TARGET)
        print(f'\n=== Simulation complete ===')
        print(f'Final box position  : {box_pos}')
        print(f'Final EE position   : {p_ee_final}')
        print(f'Final EE error      : {ee_final_err*1000:.1f} mm')
        if solve_times:
            print(f'MPC solve time stats: '
                  f'mean={np.mean(solve_times)*1000:.2f} ms  '
                  f'max={np.max(solve_times)*1000:.2f} ms')
        if viewer_ctx is not None:
            print('Close viewer to exit.')
            while viewer_ctx.is_running():
                time.sleep(0.05)

    # Launch
    if use_viewer:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            viewer.sync()
            sim_loop(viewer)
    else:
        sim_loop(None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='MPC Grasp Task')
    p.add_argument('--no-terminal-constraint', action='store_true',
                   help='Disable hard terminal equality constraint')
    p.add_argument('--rebuild', action='store_true',
                   help='Force recompile of acados C code')
    p.add_argument('--no-viewer', action='store_true',
                   help='Run headless')
    p.add_argument('--dt-mpc', type=float, default=0.05,
                   help='MPC shooting step size in seconds (default: 0.05)')
    p.add_argument('--horizon', type=int, default=20,
                   help='MPC prediction horizon steps (default: 20)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    print('=== MPC Grasp Task ===')
    run_mpc_grasp(
        dt_mpc=args.dt_mpc,
        N=args.horizon,
        rebuild=args.rebuild,
        enable_terminal_constraint=not args.no_terminal_constraint,
        use_viewer=not args.no_viewer,
    )
