"""Load, inspect, and basic-test the MuJoCo aerial manipulator model."""
import numpy as np
import mujoco


def load_model():
    import os
    xml_path = os.path.join(os.path.dirname(__file__), 'model', 'am_robot.xml')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    return model, data


def get_state(model, data):
    """Extract state in our convention: [p, v, q_xyzw, omega, theta, theta_dot]."""
    # Platform position and velocity
    p = data.qpos[:3].copy()
    v = data.qvel[:3].copy()
    # MuJoCo quaternion [w,x,y,z] → our [x,y,z,w]
    q_wxyz = data.qpos[3:7]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    # Angular velocity (MuJoCo gives it in world frame for free joints)
    omega = data.qvel[3:6].copy()
    # Joint angles and velocities
    theta = data.qpos[7:9].copy()
    theta_dot = data.qvel[6:8].copy()
    return np.concatenate([p, v, q_xyzw, omega, theta, theta_dot])


def set_state(model, data, x):
    """Set MuJoCo state from our convention vector."""
    data.qpos[:3] = x[:3]       # position
    data.qvel[:3] = x[3:6]      # velocity
    # Our [x,y,z,w] → MuJoCo [w,x,y,z]
    q_xyzw = x[6:10]
    data.qpos[3] = q_xyzw[3]    # w
    data.qpos[4] = q_xyzw[0]    # x
    data.qpos[5] = q_xyzw[1]    # y
    data.qpos[6] = q_xyzw[2]    # z
    data.qvel[3:6] = x[10:13]   # angular velocity
    data.qpos[7:9] = x[13:15]   # joint angles
    data.qvel[6:8] = x[15:17]   # joint velocities
    mujoco.mj_forward(model, data)


def apply_platform_wrench(data, F, tau):
    """Apply external force and torque to platform in world frame."""
    platform_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, 'base')
    data.xfrc_applied[platform_id, :3] = F
    data.xfrc_applied[platform_id, 3:] = tau


if __name__ == '__main__':
    model, data = load_model()

    print('=== Model Info ===')
    print(f'  Bodies: {model.nbody}')
    print(f'  Joints: {model.njnt}')
    print(f'  Actuators: {model.nu}')
    print(f'  qpos size: {model.nq}  (3 pos + 4 quat + 2 joints = 9)')
    print(f'  qvel size: {model.nv}  (3 vel + 3 angvel + 2 joints = 8)')
    print(f'  Timestep: {model.opt.timestep} s')

    # Body names and masses
    print('\n=== Bodies ===')
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        mass = model.body_mass[i]
        print(f'  [{i}] {name}: mass={mass:.3f} kg')
    total_mass = np.sum(model.body_mass)
    print(f'  Total mass: {total_mass:.3f} kg')

    # --- Test 1: Initial position ---
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    x0 = get_state(model, data)
    print('\n=== Initial State ===')
    print(f'  Position: {x0[:3]}')
    print(f'  Quaternion [x,y,z,w]: {x0[6:10]}')
    print(f'  Joints: {x0[13:15]}')

    # End-effector position
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
    print(f'  End-effector world pos: {data.site_xpos[ee_id]}')

    # --- Test 2: Free fall for 0.1 s (short, before ground contact) ---
    print('\n=== Free Fall Test (0.1 s) ===')
    mujoco.mj_resetData(model, data)
    # Start high enough to avoid ground contact
    data.qpos[2] = 2.0
    mujoco.mj_forward(model, data)
    t_fall = 0.1
    n_steps = int(t_fall / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    x_ff = get_state(model, data)
    z_expected = 2.0 + 0.5 * (-9.81) * t_fall**2
    vz_expected = -9.81 * t_fall
    print(f'  z = {x_ff[2]:.4f}  (expected {z_expected:.4f})')
    print(f'  vz = {x_ff[5]:.4f}  (expected {vz_expected:.4f})')
    assert abs(x_ff[2] - z_expected) < 0.01, f'Free fall z mismatch'
    assert abs(x_ff[5] - vz_expected) < 0.01, f'Free fall vz mismatch'
    print('  PASSED')

    # --- Test 3: Hover (apply mg upward) ---
    print('\n=== Hover Test (1 s) ===')
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 2.0  # start high, away from ground
    mujoco.mj_forward(model, data)
    mg = total_mass * 9.81
    n_hover = int(1.0 / model.opt.timestep)
    for _ in range(n_hover):
        apply_platform_wrench(data, [0, 0, mg], [0, 0, 0])
        mujoco.mj_step(model, data)
    x_hov = get_state(model, data)
    print(f'  Position after 1 s: {x_hov[:3]}')
    print(f'  Velocity: {x_hov[3:6]}')
    z_err = abs(x_hov[2] - 2.0)
    print(f'  z error: {z_err:.6f}')
    print(f'  (Small drift expected — force applied at platform COM, not system COM)')

    # --- Test 4: Viewer (if available) ---
    print('\n=== Viewer ===')
    try:
        import mujoco.viewer
        print('  Launching viewer... (close window to exit)')
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        mujoco.viewer.launch(model, data)
    except Exception as e:
        print(f'  Viewer not available: {e}')

    print('\nDone.')
