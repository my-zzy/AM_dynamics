"""CasADi symbolic dynamics for the Aerial Manipulator System.

Mirrors ams/dynamics.py, ams/kinematics.py, ams/math_utils.py and
ams/simulator.py using CasADi MX symbolic expressions so that the
full Newton-Euler pipeline can be handed to acados for NMPC.

Main entry point
----------------
    f, ee_pos_func = build_ca_dynamics(model)

    f          : CasADi Function  f(x[17], u[8]) -> x_dot[17]
    ee_pos_func: CasADi Function  ee_pos(x[17]) -> p_ee[3]

State layout (matches ams/state.py)
------------------------------------
    x[0:3]   p_A      platform position (world)
    x[3:6]   v_A      platform linear velocity (world)
    x[6:10]  q_A      platform quaternion [x,y,z,w]
    x[10:13] omega_A  platform angular velocity (body)
    x[13:15] theta    joint angles
    x[15:17] theta_dot joint velocities

Control input
-------------
    u[0:3]   F_ext        external force on platform (body frame; Fx=Fy=0 → pure thrust along body-z)
    u[3:6]   tau_ext      external torque on platform (body frame; roll/pitch/yaw)
    u[6:8]   joint_torques
"""

import casadi as ca
import numpy as np

from .model import AerialManipulatorModel

# World-frame z-axis (constant DM column vector)
_Z = ca.DM([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def ca_skew(v):
    """Skew-symmetric matrix from a 3-vector (MX or DM)."""
    return ca.vertcat(
        ca.horzcat(   0,   -v[2],  v[1]),
        ca.horzcat( v[2],     0,  -v[0]),
        ca.horzcat(-v[1],  v[0],     0),
    )


def ca_quat_to_rotmat(q):
    """Quaternion [x,y,z,w] -> 3x3 rotation matrix (CasADi MX)."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)),
        ca.horzcat(  2*(x*y + z*w),   1 - 2*(x*x + z*z),      2*(y*z - x*w)),
        ca.horzcat(  2*(x*z - y*w),     2*(y*z + x*w),    1 - 2*(x*x + y*y)),
    )


def ca_quat_deriv(q, omega):
    """Quaternion derivative: q_dot = 0.5 * Omega(omega) @ q.

    q:     [x,y,z,w] (4,1)
    omega: angular velocity in body frame [wx,wy,wz] (3,1)
    """
    wx, wy, wz = omega[0], omega[1], omega[2]
    Omega = ca.vertcat(
        ca.horzcat( 0,   -wz,  wy,  wx),
        ca.horzcat(wz,    0,  -wx,  wy),
        ca.horzcat(-wy,  wx,   0,   wz),
        ca.horzcat(-wx, -wy,  -wz,   0),
    )
    return 0.5 * ca.mtimes(Omega, q)


# ---------------------------------------------------------------------------
# DH transform helpers
# ---------------------------------------------------------------------------

def _ca_dh_transform(link, theta_i):
    """CasADi DH rotation and translation for one link joint angle.

    Matches LinkParams.dh_transform but accepts a symbolic theta_i.

    Returns:
        R: (3,3) CasADi expression
        p: (3,1) CasADi expression
    """
    ct = ca.cos(theta_i)
    st = ca.sin(theta_i)
    ca_ = float(np.cos(link.alpha))   # numeric (alpha is constant)
    sa  = float(np.sin(link.alpha))

    R = ca.vertcat(
        ca.horzcat(ct,       -st,      0.0),
        ca.horzcat(st * ca_, ct * ca_, -sa),
        ca.horzcat(st * sa,  ct * sa,  ca_),
    )
    p = ca.vertcat(
        ca.MX(link.a),
        ca.MX(-link.d * sa),
        ca.MX(link.d * ca_),
    )
    return R, p


def _ca_compute_link_transforms(model, theta):
    """CasADi version of AerialManipulatorModel.compute_link_transforms.

    theta: (n,1) symbolic joint angles

    Returns:
        R_local: list of (3,3) CasADi expressions  [{0}→{1}, {1}→{2}, {2}→{3}]
        p_local: list of (3,1) CasADi expressions
    """
    n = model.n_joints
    R_local = []
    p_local = []

    # {0} → {1}: pure rotation θ₁ around z, origins coincide
    ct0 = ca.cos(theta[0])
    st0 = ca.sin(theta[0])
    R_local.append(ca.vertcat(
        ca.horzcat(ct0, -st0, 0.0),
        ca.horzcat(st0,  ct0, 0.0),
        ca.horzcat(0.0,  0.0, 1.0),
    ))
    p_local.append(ca.DM.zeros(3, 1))

    # {i} → {i+1} for i = 1 … n-1  (uses link[i-1] geometry, joint angle theta[i])
    # theta[1] has a -π/2 offset: DH convention sets θ=0 as collinear (straight),
    # but the XML zero config is the L-shape (link2 horizontal), so θ₂_DH = θ₂_XML - π/2.
    for i in range(1, n):
        offset = np.pi / 2 if i == 1 else 0.0
        R, p = _ca_dh_transform(model.links[i - 1], theta[i] - offset)
        R_local.append(R)
        p_local.append(p)

    # {n} → {n+1}: end-effector frame, last link geometry, no joint
    R_ee, p_ee = _ca_dh_transform(model.links[-1], ca.DM(0.0))
    R_local.append(R_ee)
    p_local.append(p_ee)

    return R_local, p_local


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def ca_forward_kinematics(model, q_A, p_A, theta):
    """CasADi forward kinematics.

    Returns:
        R:     list of (3,3) world-frame rotations   R[0]..R[3]
        p:     list of (3,1) world-frame origins     p[0]..p[3]
        p_com: list of (3,1) world-frame COM positions  p_com[0]=link1, [1]=link2
    """
    n = model.n_joints
    R_A = ca_quat_to_rotmat(q_A)

    mount_R = ca.DM(model.mount_rotation)
    mount_t = ca.DM(model.mount_offset.reshape(3, 1))

    R0 = ca.mtimes(R_A, mount_R)
    p0 = p_A + ca.mtimes(R_A, mount_t)

    R_local, p_local = _ca_compute_link_transforms(model, theta)

    n_frames = n + 2   # 4 frames: {0},{1},{2},{3}
    R = [None] * n_frames
    p = [None] * n_frames
    R[0] = R0
    p[0] = p0

    for i in range(len(R_local)):
        R[i + 1] = ca.mtimes(R[i], R_local[i])
        p[i + 1] = p[i] + ca.mtimes(R[i], p_local[i])

    p_com = []
    for i in range(n):
        com_off = ca.DM(model.links[i].com_offset.reshape(3, 1))
        p_com.append(p[i + 1] + ca.mtimes(R[i + 1], com_off))

    return R, p, p_com


# ---------------------------------------------------------------------------
# Velocity recursion
# ---------------------------------------------------------------------------

def ca_velocity_recursion(model, omega_A, v_A, p_A, theta_dot, R, p, p_com):
    """CasADi velocity recursion.

    omega_A, v_A, p_A: (3,1) MX
    theta_dot:         (n,1) MX  (or list of n scalar MX)

    Returns:
        omega: list of (3,1) angular velocities   omega[0]..omega[3]
        v:     list of (3,1) linear velocities    v[0]..v[3]
        v_com: list of (3,1) COM velocities       v_com[0]=link1, [1]=link2
    """
    n = model.n_joints
    n_frames = n + 2

    omega = [None] * n_frames
    v     = [None] * n_frames

    omega[0] = omega_A
    r_mount  = p[0] - p_A
    v[0]     = v_A + ca.cross(omega_A, r_mount)

    # Extend joint velocities with a trailing zero (EE frame, no joint)
    td_ext = [theta_dot[i] for i in range(n)] + [ca.DM(0.0)]

    for i in range(n_frames - 1):
        r = p[i + 1] - p[i]
        omega[i + 1] = omega[i] + ca.mtimes(R[i + 1], td_ext[i] * _Z)
        v[i + 1]     = v[i] + ca.cross(omega[i], r)

    v_com = []
    for i in range(n):
        r_com = p_com[i] - p[i + 1]
        v_com.append(v[i + 1] + ca.cross(omega[i + 1], r_com))

    return omega, v, v_com


# ---------------------------------------------------------------------------
# Acceleration recursion
# ---------------------------------------------------------------------------

def ca_acceleration_recursion(model, alpha_A, a_A, p_A,
                               theta_dot, theta_ddot,
                               omega, R, p, p_com):
    """CasADi acceleration recursion.

    Returns:
        alpha: list of (3,1) angular accelerations  alpha[0]..alpha[3]
        a:     list of (3,1) linear accelerations   a[0]..a[3]
        a_com: list of (3,1) COM linear accelerations
    """
    n = model.n_joints
    n_frames = n + 2

    alpha = [None] * n_frames
    a     = [None] * n_frames

    r_mount  = p[0] - p_A
    alpha[0] = alpha_A
    a[0]     = (a_A
                + ca.cross(alpha_A, r_mount)
                + ca.cross(omega[0], ca.cross(omega[0], r_mount)))

    td_ext  = [theta_dot[i]   for i in range(n)] + [ca.DM(0.0)]
    tdd_ext = [theta_ddot[i]  for i in range(n)] + [ca.DM(0.0)]

    for i in range(n_frames - 1):
        r = p[i + 1] - p[i]
        joint_vel_axis  = ca.mtimes(R[i + 1], td_ext[i]  * _Z)
        joint_acc_axis  = ca.mtimes(R[i + 1], tdd_ext[i] * _Z)
        alpha[i + 1] = (alpha[i]
                        + ca.cross(omega[i + 1], joint_vel_axis)
                        + joint_acc_axis)
        a[i + 1]     = (a[i]
                        + ca.cross(alpha[i], r)
                        + ca.cross(omega[i], ca.cross(omega[i], r)))

    a_com = []
    for i in range(n):
        r_com = p_com[i] - p[i + 1]
        a_com.append(a[i + 1]
                     + ca.cross(alpha[i + 1], r_com)
                     + ca.cross(omega[i + 1], ca.cross(omega[i + 1], r_com)))

    return alpha, a, a_com


# ---------------------------------------------------------------------------
# Backward (Newton-Euler) recursion
# ---------------------------------------------------------------------------

def _ca_backward_recursion(model, omega, alpha, a_com, R, p, p_com):
    """CasADi Newton-Euler backward recursion.

    Returns:
        jt_list: list of n scalar MX joint torques (index 0 = joint 1)
        f_base:  (3,1) constraint force at arm base
        tau_base:(3,1) constraint torque at arm base
    """
    n   = model.n_joints
    g   = ca.DM(model.gravity.reshape(3, 1))

    f_tip   = ca.DM.zeros(3, 1)
    tau_tip = ca.DM.zeros(3, 1)
    jt_list = [None] * n

    for j in reversed(range(n)):
        link = model.links[j]
        I_body = ca.DM(link.inertia)
        I_w    = ca.mtimes(R[j + 1], ca.mtimes(I_body, R[j + 1].T))

        r_in  = p[j + 1] - p_com[j]
        r_out = p[j + 2] - p_com[j]

        # Force balance: m*a_com = f - f_tip + m*g
        f = link.mass * a_com[j] - link.mass * g + f_tip

        # Torque balance about COM
        tau = (ca.mtimes(I_w, alpha[j + 1])
               + ca.cross(omega[j + 1], ca.mtimes(I_w, omega[j + 1]))
               - ca.cross(r_in,  f)
               + tau_tip
               + ca.cross(r_out, f_tip))

        # Project onto joint z-axis (world frame)
        z_w = ca.mtimes(R[j + 1], _Z)
        jt_list[j] = ca.dot(tau, z_w)

        f_tip   = f
        tau_tip = tau

    return jt_list, f_tip, tau_tip


# ---------------------------------------------------------------------------
# Internal: evaluate inverse dynamics (reusing shared FK + velocity pass)
# ---------------------------------------------------------------------------

def _ca_eval_id(model, R_A, p_A, omega_A, theta_dot,
                a_A, alpha_A, theta_ddot,
                omega, R, p, p_com):
    """Evaluate inverse dynamics symbolically, reusing pre-computed omega/R/p.

    Returns:
        (8,1) MX vector: [F_ext(3), tau_ext(3), joint_torques(n)]
    """
    n = model.n_joints

    alpha, a, a_com = ca_acceleration_recursion(
        model, alpha_A, a_A, p_A,
        theta_dot, theta_ddot,
        omega, R, p, p_com)

    jt_list, f1, tau1 = _ca_backward_recursion(model, omega, alpha, a_com, R, p, p_com)

    I_A_body = ca.DM(model.platform_inertia)
    I_A_w    = ca.mtimes(R_A, ca.mtimes(I_A_body, R_A.T))
    r_mount  = p[0] - p_A
    g        = ca.DM(model.gravity.reshape(3, 1))

    F_ext   = model.platform_mass * a_A + f1 - model.platform_mass * g
    tau_ext = (ca.mtimes(I_A_w, alpha_A)
               + ca.cross(omega_A, ca.mtimes(I_A_w, omega_A))
               + tau1
               + ca.cross(r_mount, f1))

    jt = ca.vertcat(*jt_list)
    return ca.vertcat(F_ext, tau_ext, jt)   # (6+n, 1)


# ---------------------------------------------------------------------------
# Forward dynamics (mass-matrix via repeated ID)
# ---------------------------------------------------------------------------

def ca_forward_dynamics(model, q_A, p_A, omega_A, v_A,
                        theta, theta_dot,
                        F_ext, tau_ext, joint_torques):
    """CasADi forward dynamics.

    Builds the (n_acc x n_acc) mass matrix symbolically via n_acc+1
    evaluations of the inverse dynamics (one bias + one per column),
    then solves  M * qddot = u - h  using ca.solve.

    Returns:
        a_A        : (3,1) platform linear acceleration
        alpha_A    : (3,1) platform angular acceleration
        theta_ddot : (n,1) joint accelerations
    """
    n     = model.n_joints
    n_acc = 6 + n                        # 8 for 2-joint arm

    R_A = ca_quat_to_rotmat(q_A)

    # Shared FK + velocity pass
    R, p, p_com = ca_forward_kinematics(model, q_A, p_A, theta)
    omega, _, _ = ca_velocity_recursion(model, omega_A, v_A, p_A, theta_dot, R, p, p_com)

    # Bias vector h (Coriolis + gravity, zero accelerations)
    z3 = ca.DM.zeros(3, 1)
    zn = ca.DM.zeros(n, 1)
    h  = _ca_eval_id(model, R_A, p_A, omega_A, theta_dot,
                     z3, z3, zn,
                     omega, R, p, p_com)

    # Mass matrix: column k = ID(e_k) - h
    M_cols = []
    for k in range(n_acc):
        e = ca.DM.zeros(n_acc, 1)
        e[k] = 1.0
        col = _ca_eval_id(model, R_A, p_A, omega_A, theta_dot,
                          e[:3], e[3:6], e[6:],
                          omega, R, p, p_com) - h
        M_cols.append(col)

    M     = ca.horzcat(*M_cols)                      # (n_acc, n_acc)
    u_vec = ca.vertcat(F_ext, tau_ext, joint_torques) # (n_acc, 1)
    qddot = ca.solve(M, u_vec - h)                   # (n_acc, 1)

    return qddot[:3], qddot[3:6], qddot[6:]


# ---------------------------------------------------------------------------
# Full state derivative
# ---------------------------------------------------------------------------

def _ca_state_derivative_expr(model, x, u):
    """Build the state-derivative expression f(x,u) -> x_dot (symbolic).

    Does NOT create a ca.Function; returns the raw MX expression so that
    acados can embed it in a larger expression graph if needed.
    """
    # Unpack state
    p_A       = x[0:3]
    v_A       = x[3:6]
    q_A       = x[6:10]
    omega_A   = x[10:13]
    theta     = x[13:15]
    theta_dot = x[15:17]

    # Unpack input (body-frame: thrust along body-z, torques as roll/pitch/yaw)
    F_body   = u[0:3]
    tau_body = u[3:6]
    jt       = u[6:8]

    # Rotate to world frame so that tilting the drone creates horizontal force.
    # This is the standard quadrotor coupling: F_world = R_body @ F_body.
    R_A = ca_quat_to_rotmat(q_A)
    F_world   = ca.mtimes(R_A, F_body)
    tau_world = ca.mtimes(R_A, tau_body)

    a_A, alpha_A, theta_ddot = ca_forward_dynamics(
        model, q_A, p_A, omega_A, v_A,
        theta, theta_dot,
        F_world, tau_world, jt)

    q_dot = ca_quat_deriv(q_A, omega_A)

    return ca.vertcat(
        v_A,          # p_dot   = v
        a_A,          # v_dot   = a
        q_dot,        # q_dot
        alpha_A,      # omega_dot = alpha
        theta_dot,    # theta_dot
        theta_ddot,   # theta_ddot
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ca_dynamics(model=None):
    """Build CasADi symbolic functions for the aerial manipulator dynamics.

    Args:
        model: AerialManipulatorModel instance (default: default parameters)

    Returns:
        f          : ca.Function  f(x[17], u[8]) -> x_dot[17]
        ee_pos_func: ca.Function  ee_pos(x[17]) -> p_ee[3]
    """
    if model is None:
        model = AerialManipulatorModel()

    x = ca.MX.sym('x', 17)
    u = ca.MX.sym('u', 8)

    x_dot_expr = _ca_state_derivative_expr(model, x, u)
    f = ca.Function('f', [x, u], [x_dot_expr],
                    ['x', 'u'], ['x_dot'])

    # EE position function (depends only on state)
    p_A     = x[0:3]
    q_A     = x[6:10]
    theta   = x[13:15]
    R, p, _ = ca_forward_kinematics(model, q_A, p_A, theta)
    p_ee    = p[3]                        # world-frame EE position (3,1)
    ee_pos_func = ca.Function('ee_pos', [x], [p_ee],
                              ['x'], ['p_ee'])

    return f, ee_pos_func


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model=None, seed=42, atol=1e-6):
    """Compare CasADi f(x,u) output with numpy state_derivative for random inputs.

    Prints pass/fail for each check and raises AssertionError on mismatch.
    """
    from .simulator import state_derivative
    from .state import SystemState

    if model is None:
        model = AerialManipulatorModel()

    rng = np.random.default_rng(seed)

    def _random_state():
        p   = rng.standard_normal(3)
        v   = rng.standard_normal(3) * 0.5
        q   = rng.standard_normal(4)
        q  /= np.linalg.norm(q)
        om  = rng.standard_normal(3) * 0.3
        th  = rng.standard_normal(2) * 0.4
        td  = rng.standard_normal(2) * 0.3
        return np.concatenate([p, v, q, om, th, td])

    def _random_input():
        return rng.standard_normal(8) * np.array([5, 5, 5, 0.5, 0.5, 0.5, 1, 1])

    f, ee_pos_func = build_ca_dynamics(model)

    for trial in range(5):
        x_np = _random_state()
        u_np = _random_input()

        # Numpy reference
        xdot_np = state_derivative(model, x_np, u_np)

        # CasADi evaluation (convert to numpy)
        xdot_ca = np.array(f(x_np, u_np)).flatten()

        if not np.allclose(xdot_ca, xdot_np, atol=atol):
            max_err = np.max(np.abs(xdot_ca - xdot_np))
            raise AssertionError(
                f'Trial {trial}: state_derivative mismatch  max_err={max_err:.2e}\n'
                f'  numpy : {xdot_np}\n'
                f'  casadi: {xdot_ca}')
        print(f'Trial {trial+1}: state_derivative match  '
              f'(max_err={np.max(np.abs(xdot_ca - xdot_np)):.2e})  PASSED')

    # EE position check
    x_np = _random_state()
    from .kinematics import forward_kinematics
    from .math_utils import quat_to_rotation_matrix
    q_A_np = x_np[6:10]
    p_A_np = x_np[0:3]
    th_np  = x_np[13:15]
    _, p_np, _ = forward_kinematics(model, q_A_np, p_A_np, th_np)
    p_ee_np = p_np[3]
    p_ee_ca = np.array(ee_pos_func(x_np)).flatten()
    if not np.allclose(p_ee_ca, p_ee_np, atol=atol):
        raise AssertionError(
            f'EE position mismatch: numpy={p_ee_np}  casadi={p_ee_ca}')
    print(f'EE position match  (max_err={np.max(np.abs(p_ee_ca - p_ee_np)):.2e})  PASSED')

    print('\nAll CasADi dynamics validation checks PASSED.')
    return f, ee_pos_func


if __name__ == '__main__':
    validate()
