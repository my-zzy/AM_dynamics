import numpy as np
from .math_utils import quat_to_rotation_matrix, cross
from .model import AerialManipulatorModel
from .kinematics import forward_kinematics, velocity_recursion, acceleration_recursion

Z_AXIS = np.array([0.0, 0.0, 1.0])


def backward_recursion(model, omega, alpha, a_com, R, p, p_com):
    """Newton-Euler backward recursion on the manipulator links.

    Computes constraint forces/torques at each joint, working backward
    from the end-effector (zero external load) to the arm base.

    Args:
        model: AerialManipulatorModel
        omega: angular velocities for frames 0..n+1
        alpha: angular accelerations for frames 0..n+1
        a_com: COM linear accelerations for links 0..n-1
        R, p, p_com: from forward_kinematics

    Returns:
        joint_torques: motor torque at each joint (n,)
        f_base: constraint force at arm base (from base on link 1) (3,)
        tau_base: constraint torque at arm base (from base on link 1) (3,)
    """
    n = model.n_joints
    g = model.gravity
    f_tip = np.zeros(3)
    tau_tip = np.zeros(3)
    joint_torques = np.zeros(n)

    for j in reversed(range(n)):
        link = model.links[j]
        I_w = R[j + 1] @ link.inertia @ R[j + 1].T

        r_in = p[j + 1] - p_com[j]
        r_out = p[j + 2] - p_com[j]

        # Force balance: m * a_com = f - f_tip + m * g
        f = link.mass * a_com[j] - link.mass * g + f_tip

        # Torque balance about COM
        tau = (I_w @ alpha[j + 1]
               + cross(omega[j + 1], I_w @ omega[j + 1])
               - cross(r_in, f)
               + tau_tip
               + cross(r_out, f_tip))

        # Project onto joint axis (z of parent frame; equals z of child when α=0)
        z_w = R[j + 1] @ Z_AXIS
        joint_torques[j] = tau @ z_w

        f_tip = f
        tau_tip = tau

    return joint_torques, f_tip, tau_tip


def _eval_id(model, R_A, p_A, omega_A, theta_dot,
             a_A, alpha_A, theta_ddot,
             omega, R, p, p_com):
    """Evaluate inverse dynamics reusing shared FK and velocity results.

    Returns concatenated [F_ext, tau_ext, joint_torques] vector.
    """
    alpha, a, a_com = acceleration_recursion(
        model, alpha_A, a_A, p_A, theta_dot, theta_ddot,
        omega, R, p, p_com)

    jt, f1, tau1 = backward_recursion(model, omega, alpha, a_com, R, p, p_com)

    I_A_w = R_A @ model.platform_inertia @ R_A.T
    r_mount = p[0] - p_A

    F_ext = model.platform_mass * a_A + f1 - model.platform_mass * model.gravity
    tau_ext = (I_A_w @ alpha_A
               + cross(omega_A, I_A_w @ omega_A)
               + tau1
               + cross(r_mount, f1))

    return np.concatenate([F_ext, tau_ext, jt])


def inverse_dynamics(model, q_A, p_A, omega_A, v_A, theta, theta_dot,
                     a_A, alpha_A, theta_ddot):
    """Compute forces/torques required to produce given accelerations.

    Args:
        model:      AerialManipulatorModel
        q_A:        platform quaternion [x,y,z,w] (4,)
        p_A:        platform position (3,)
        omega_A:    platform angular velocity in world frame (3,)
        v_A:        platform linear velocity in world frame (3,)
        theta:      joint angles (n,)
        theta_dot:  joint velocities (n,)
        a_A:        desired platform linear acceleration (3,)
        alpha_A:    desired platform angular acceleration (3,)
        theta_ddot: desired joint accelerations (n,)

    Returns:
        F_ext:        required external force on platform (3,)
        tau_ext:      required external torque on platform (3,)
        joint_torques: required joint torques (n,)
    """
    R_A = quat_to_rotation_matrix(q_A)
    R, p, p_com = forward_kinematics(model, q_A, p_A, theta)
    omega, v, v_com = velocity_recursion(
        model, omega_A, v_A, p_A, theta_dot, R, p, p_com)

    result = _eval_id(model, R_A, p_A, omega_A, theta_dot,
                      a_A, alpha_A, theta_ddot, omega, R, p, p_com)
    n = model.n_joints
    return result[:3], result[3:6], result[6:]


def forward_dynamics(model, q_A, p_A, omega_A, v_A, theta, theta_dot,
                     F_ext, tau_ext, joint_torques):
    """Compute accelerations from applied forces/torques.

    Builds the mass matrix via repeated inverse dynamics evaluations,
    then solves M * qddot = u - h.

    Args:
        model:         AerialManipulatorModel
        q_A:           platform quaternion [x,y,z,w] (4,)
        p_A:           platform position (3,)
        omega_A:       platform angular velocity in world frame (3,)
        v_A:           platform linear velocity in world frame (3,)
        theta:         joint angles (n,)
        theta_dot:     joint velocities (n,)
        F_ext:         external force on platform (3,)
        tau_ext:       external torque on platform (3,)
        joint_torques: applied joint torques (n,)

    Returns:
        a_A:        platform linear acceleration (3,)
        alpha_A:    platform angular acceleration (3,)
        theta_ddot: joint accelerations (n,)
    """
    n = model.n_joints
    n_acc = 6 + n

    R_A = quat_to_rotation_matrix(q_A)

    # Shared forward pass (position + velocity, independent of accelerations)
    R, p, p_com = forward_kinematics(model, q_A, p_A, theta)
    omega, v, v_com = velocity_recursion(
        model, omega_A, v_A, p_A, theta_dot, R, p, p_com)

    # Bias forces (Coriolis + gravity, with zero acceleration)
    h = _eval_id(model, R_A, p_A, omega_A, theta_dot,
                 np.zeros(3), np.zeros(3), np.zeros(n),
                 omega, R, p, p_com)

    # Mass matrix column by column: M[:, k] = ID(e_k) - h
    M = np.zeros((n_acc, n_acc))
    for k in range(n_acc):
        e = np.zeros(n_acc)
        e[k] = 1.0
        M[:, k] = _eval_id(model, R_A, p_A, omega_A, theta_dot,
                           e[:3], e[3:6], e[6:],
                           omega, R, p, p_com) - h

    u = np.concatenate([F_ext, tau_ext, joint_torques])
    qddot = np.linalg.solve(M, u - h)

    return qddot[:3], qddot[3:6], qddot[6:]


if __name__ == '__main__':
    m = AerialManipulatorModel()
    q_id = np.array([0.0, 0.0, 0.0, 1.0])
    p_A = np.array([0.0, 0.0, 1.0])
    z3 = np.zeros(3)
    z2 = np.zeros(2)
    total_mass = m.platform_mass + sum(l.mass for l in m.links)

    # --- Static hover (zero config, arm hanging down) ---
    F, tau, jt = inverse_dynamics(m, q_id, p_A, z3, z3, z2, z2, z3, z3, z2)
    assert np.allclose(F, [0, 0, total_mass * 9.81]), f'F={F}'
    assert np.allclose(tau, 0, atol=1e-10), f'tau={tau}'
    assert np.allclose(jt, 0, atol=1e-10), f'jt={jt}'
    print('Static hover (zero config): PASSED')

    # --- Free fall (no external forces) ---
    a, alpha, tdd = forward_dynamics(m, q_id, p_A, z3, z3, z2, z2,
                                     z3, z3, z2)
    assert np.allclose(a, m.gravity, atol=1e-10), f'a={a}'
    assert np.allclose(alpha, 0, atol=1e-10), f'alpha={alpha}'
    assert np.allclose(tdd, 0, atol=1e-10), f'tdd={tdd}'
    print('Free fall: PASSED')

    # --- ID/FD round trip (arbitrary state) ---
    rng = np.random.default_rng(42)
    theta_t = rng.standard_normal(2) * 0.5
    td_t = rng.standard_normal(2) * 0.3
    a_t = rng.standard_normal(3)
    al_t = rng.standard_normal(3)
    tdd_t = rng.standard_normal(2)
    F_id, tau_id, jt_id = inverse_dynamics(
        m, q_id, p_A, z3, z3, theta_t, td_t, a_t, al_t, tdd_t)
    a_fd, al_fd, tdd_fd = forward_dynamics(
        m, q_id, p_A, z3, z3, theta_t, td_t, F_id, tau_id, jt_id)
    assert np.allclose(a_fd, a_t, atol=1e-10), f'a mismatch: {a_fd} vs {a_t}'
    assert np.allclose(al_fd, al_t, atol=1e-10), f'alpha mismatch'
    assert np.allclose(tdd_fd, tdd_t, atol=1e-10), f'tdd mismatch'
    print('ID/FD round trip: PASSED')

    # --- Mass matrix symmetry and positive definiteness ---
    R_A = quat_to_rotation_matrix(q_id)
    R, p_fk, pc = forward_kinematics(m, q_id, p_A, theta_t)
    om, _, _ = velocity_recursion(m, z3, z3, p_A, td_t, R, p_fk, pc)
    n_acc = 8
    h = _eval_id(m, R_A, p_A, z3, td_t, z3, z3, z2, om, R, p_fk, pc)
    M = np.zeros((n_acc, n_acc))
    for k in range(n_acc):
        e = np.zeros(n_acc)
        e[k] = 1.0
        M[:, k] = _eval_id(m, R_A, p_A, z3, td_t,
                           e[:3], e[3:6], e[6:], om, R, p_fk, pc) - h
    assert np.allclose(M, M.T, atol=1e-10), 'M not symmetric'
    eig = np.linalg.eigvalsh(M)
    assert np.all(eig > 0), f'M not positive definite: {eig}'
    print('Mass matrix symmetric & positive definite: PASSED')

    # --- Static hover at theta1=90 deg ---
    theta90 = np.array([np.pi / 2, 0.0])
    F90, tau90, jt90 = inverse_dynamics(m, q_id, p_A, z3, z3, theta90, z2,
                                        z3, z3, z2)
    assert np.allclose(F90, [0, 0, total_mass * 9.81], atol=1e-6), f'F90={F90}'
    exp_jt1 = 0.15 * 9.81 * 0.125 + 0.12 * 9.81 * 0.35
    exp_jt2 = 0.12 * 9.81 * 0.10
    assert np.allclose(jt90[0], exp_jt1, atol=1e-4), f'jt1={jt90[0]}, exp={exp_jt1}'
    assert np.allclose(jt90[1], exp_jt2, atol=1e-4), f'jt2={jt90[1]}, exp={exp_jt2}'
    print('Static hover (theta1=90): PASSED')

    # --- ID/FD round trip with nonzero platform velocity ---
    omega_t = rng.standard_normal(3) * 0.5
    v_t = rng.standard_normal(3) * 0.3
    q_t = np.array([0.1, 0.2, 0.05, 1.0])
    q_t /= np.linalg.norm(q_t)
    F_id2, tau_id2, jt_id2 = inverse_dynamics(
        m, q_t, p_A, omega_t, v_t, theta_t, td_t, a_t, al_t, tdd_t)
    a_fd2, al_fd2, tdd_fd2 = forward_dynamics(
        m, q_t, p_A, omega_t, v_t, theta_t, td_t, F_id2, tau_id2, jt_id2)
    assert np.allclose(a_fd2, a_t, atol=1e-8), f'a mismatch (tilted)'
    assert np.allclose(al_fd2, al_t, atol=1e-8), f'alpha mismatch (tilted)'
    assert np.allclose(tdd_fd2, tdd_t, atol=1e-8), f'tdd mismatch (tilted)'
    print('ID/FD round trip (tilted, moving): PASSED')

    print()
    print('All dynamics tests passed!')
