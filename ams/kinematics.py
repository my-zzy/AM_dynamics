import numpy as np
from .math_utils import quat_to_rotation_matrix, cross
from .model import AerialManipulatorModel

Z_AXIS = np.array([0.0, 0.0, 1.0])


def forward_kinematics(model, q_A, p_A, theta):
    """Compute world-frame rotations, positions, and COM positions for all frames.

    Frames: {0} arm base (fixed), {1} link 1 body, {2} link 2 body, {3} end-effector.

    Args:
        model: AerialManipulatorModel
        q_A:   platform quaternion [x,y,z,w] (4,)
        p_A:   platform position in world (3,)
        theta: joint angles [θ₁, θ₂] (n,)

    Returns:
        R:     list of world-frame rotation matrices, R[0]..R[3]
        p:     list of world-frame positions of frame origins, p[0]..p[3]
        p_com: list of world-frame COM positions for each link, p_com[0]=link1, p_com[1]=link2
    """
    n = model.n_joints
    R_A = quat_to_rotation_matrix(q_A)

    # Frame {0}: fixed arm base, rigidly attached to platform
    R_0 = R_A @ model.mount_rotation
    p_0 = p_A + R_A @ model.mount_offset

    # Local transforms: {0}→{1}, {1}→{2}, {2}→{3}
    R_local, p_local = model.compute_link_transforms(theta)

    n_frames = n + 2  # 4 frames: 0, 1, 2, 3
    R = [None] * n_frames
    p = [None] * n_frames
    R[0] = R_0
    p[0] = p_0

    for i in range(len(R_local)):
        R[i + 1] = R[i] @ R_local[i]
        p[i + 1] = p[i] + R[i] @ p_local[i]

    # COM positions in world frame
    p_com = [None] * n
    for i in range(n):
        # Link i+1 COM offset is defined in frame {i+1}
        p_com[i] = p[i + 1] + R[i + 1] @ model.links[i].com_offset

    return R, p, p_com


def velocity_recursion(model, omega_A, v_A, p_A, theta_dot, R, p, p_com):
    """Compute world-frame angular and linear velocities for all frames and COMs.

    Args:
        omega_A:   platform angular velocity in world frame (3,)
        v_A:       platform COM linear velocity in world frame (3,)
        p_A:       platform position in world (3,)
        theta_dot: joint velocities [θ̇₁, θ̇₂] (n,)
        R, p, p_com: from forward_kinematics

    Returns:
        omega: angular velocities for frames 0..3
        v:     linear velocities of frame origins 0..3
        v_com: COM linear velocities for links 1..2 (0-indexed)
    """
    n = model.n_joints
    n_frames = n + 2

    omega = [None] * n_frames
    v = [None] * n_frames

    # Frame {0}: rigidly attached to platform
    omega[0] = omega_A.copy()
    r_mount = p[0] - p_A
    v[0] = v_A + cross(omega_A, r_mount)

    # Joint rates for each transition: {0}→{1} uses θ̇₁, {1}→{2} uses θ̇₂, {2}→{3} has none
    theta_dot_ext = np.append(theta_dot, 0.0)

    for i in range(n_frames - 1):
        r = p[i + 1] - p[i]
        # ω_{i+1} = ω_i + R_{i+1} · θ̇ · z
        omega[i + 1] = omega[i] + R[i + 1] @ (theta_dot_ext[i] * Z_AXIS)
        # v_{i+1} = v_i + ω_i × (p_{i+1} - p_i)
        v[i + 1] = v[i] + cross(omega[i], r)

    # COM velocities
    v_com = [None] * n
    for i in range(n):
        r_com = p_com[i] - p[i + 1]
        v_com[i] = v[i + 1] + cross(omega[i + 1], r_com)

    return omega, v, v_com


def acceleration_recursion(model, alpha_A, a_A, p_A, theta_dot, theta_ddot,
                           omega, R, p, p_com):
    """Compute world-frame angular and linear accelerations for all frames and COMs.

    Args:
        alpha_A:    platform angular acceleration in world frame (3,)
        a_A:        platform COM linear acceleration in world frame (3,)
        p_A:        platform position in world (3,)
        theta_dot:  joint velocities [θ̇₁, θ̇₂] (n,)
        theta_ddot: joint accelerations [θ̈₁, θ̈₂] (n,)
        omega:      angular velocities from velocity_recursion
        R, p, p_com: from forward_kinematics

    Returns:
        alpha: angular accelerations for frames 0..3
        a:     linear accelerations of frame origins 0..3
        a_com: COM linear accelerations for links 1..2 (0-indexed)
    """
    n = model.n_joints
    n_frames = n + 2

    alpha = [None] * n_frames
    a = [None] * n_frames

    # Frame {0}: rigidly attached to platform
    r_mount = p[0] - p_A
    alpha[0] = alpha_A.copy()
    a[0] = a_A + cross(alpha_A, r_mount) + cross(omega[0], cross(omega[0], r_mount))

    theta_dot_ext = np.append(theta_dot, 0.0)
    theta_ddot_ext = np.append(theta_ddot, 0.0)

    for i in range(n_frames - 1):
        r = p[i + 1] - p[i]
        # α_{i+1} = α_i + ω_{i+1} × (R_{i+1} · θ̇ · z) + R_{i+1} · θ̈ · z
        joint_vel_axis = R[i + 1] @ (theta_dot_ext[i] * Z_AXIS)
        alpha[i + 1] = (alpha[i]
                        + cross(omega[i + 1], joint_vel_axis)
                        + R[i + 1] @ (theta_ddot_ext[i] * Z_AXIS))
        # a_{i+1} = a_i + α_i × r + ω_i × (ω_i × r)
        a[i + 1] = (a[i]
                    + cross(alpha[i], r)
                    + cross(omega[i], cross(omega[i], r)))

    # COM accelerations
    a_com = [None] * n
    for i in range(n):
        r_com = p_com[i] - p[i + 1]
        a_com[i] = (a[i + 1]
                    + cross(alpha[i + 1], r_com)
                    + cross(omega[i + 1], cross(omega[i + 1], r_com)))

    return alpha, a, a_com


if __name__ == '__main__':
    m = AerialManipulatorModel()
    q_id = np.array([0.0, 0.0, 0.0, 1.0])
    p_A = np.array([0.0, 0.0, 1.0])

    # --- FK zero config ---
    theta = np.zeros(2)
    R, p, pc = forward_kinematics(m, q_id, p_A, theta)
    assert np.allclose(p[0], [0, 0, 0.95])
    assert np.allclose(p[1], [0, 0, 0.95])
    assert np.allclose(p[2], [0, 0, 0.70])
    assert np.allclose(p[3], [0, 0, 0.50])
    assert np.allclose(pc[0], [0, 0, 0.825])
    assert np.allclose(pc[1], [0, 0, 0.60])
    print('FK zero config: PASSED')

    # --- FK theta1=90 deg ---
    R2, p2, pc2 = forward_kinematics(m, q_id, p_A, np.array([np.pi / 2, 0.0]))
    assert np.allclose(p2[1], [0, 0, 0.95], atol=1e-10)
    assert np.allclose(p2[2], [0.25, 0, 0.95], atol=1e-10)
    assert np.allclose(p2[3], [0.45, 0, 0.95], atol=1e-10)
    print('FK theta1=90: PASSED')

    # --- Velocity at rest ---
    omega_A = np.zeros(3)
    v_A = np.zeros(3)
    omega, v, vc = velocity_recursion(m, omega_A, v_A, p_A, np.zeros(2), R, p, pc)
    assert all(np.allclose(w, 0) for w in omega)
    assert all(np.allclose(vi, 0) for vi in v)
    print('Velocity at rest: PASSED')

    # --- Joint 1 spinning at 1 rad/s ---
    td = np.array([1.0, 0.0])
    omega3, v3, vc3 = velocity_recursion(m, omega_A, v_A, p_A, td, R, p, pc)
    assert np.allclose(R[1] @ Z_AXIS, [0, -1, 0])
    assert np.allclose(omega3[0], [0, 0, 0])
    assert np.allclose(omega3[1], [0, -1, 0])
    assert np.allclose(v3[1], [0, 0, 0], atol=1e-10)
    assert np.allclose(v3[2], [0.25, 0, 0], atol=1e-10)
    assert np.allclose(v3[3], [0.45, 0, 0], atol=1e-10)
    assert np.allclose(vc3[0], [0.125, 0, 0], atol=1e-10)
    print('Velocity joint1 spinning: PASSED')

    # --- Centripetal acceleration ---
    alpha3, a3, ac3 = acceleration_recursion(
        m, np.zeros(3), np.zeros(3), p_A, td, np.zeros(2), omega3, R, p, pc)
    assert np.allclose(a3[1], [0, 0, 0], atol=1e-10)
    assert np.allclose(a3[2], [0, 0, 0.25], atol=1e-10)
    print('Centripetal acceleration: PASSED')

    # --- Platform yaw at 1 rad/s ---
    oy, vy, vcy = velocity_recursion(
        m, np.array([0.0, 0.0, 1.0]), v_A, p_A, np.zeros(2), R, p, pc)
    for i in range(4):
        assert np.allclose(oy[i], [0, 0, 1])
    assert np.allclose(vy[0], [0, 0, 0], atol=1e-10)
    print('Platform yaw: PASSED')

    # --- Tilted platform (90 deg pitch, quaternion test) ---
    a = np.pi / 4
    q_pitch = np.array([0, np.sin(a), 0, np.cos(a)])
    Rp, pp, pcp = forward_kinematics(m, q_pitch, p_A, np.zeros(2))
    assert np.allclose(pp[0], [-0.05, 0, 1.0], atol=1e-10)
    assert np.allclose(pp[3], [-0.50, 0, 1.0], atol=1e-10)
    print('Tilted platform (quaternion): PASSED')

    print()
    print('All kinematics tests passed!')
