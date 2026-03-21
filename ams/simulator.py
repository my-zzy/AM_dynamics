import numpy as np
from .math_utils import quat_derivative, quat_normalize
from .model import AerialManipulatorModel
from .dynamics import forward_dynamics
from .state import SystemState, N_JOINTS


def state_derivative(model, x, u):
    """Compute x_dot = f(x, u).

    Args:
        model: AerialManipulatorModel
        x:     state vector (17,)  [p, v, q, omega, theta, theta_dot]
        u:     input vector (8,)   [F_ext(3), tau_ext(3), joint_torques(n)]

    Returns:
        x_dot: state derivative vector (17,)
    """
    s = SystemState.from_vector(x)
    n = N_JOINTS

    F_ext = u[:3]
    tau_ext = u[3:6]
    jt = u[6:]

    a_A, alpha_A, theta_ddot = forward_dynamics(
        model,
        s.platform.quaternion,
        s.platform.position,
        s.platform.angular_velocity,
        s.platform.velocity,
        s.manipulator.joint_angles,
        s.manipulator.joint_velocities,
        F_ext, tau_ext, jt)

    q_dot = quat_derivative(s.platform.quaternion, s.platform.angular_velocity)

    return np.concatenate([
        s.platform.velocity,              # p_dot   = v
        a_A,                              # v_dot   = a
        q_dot,                            # q_dot
        alpha_A,                          # omega_dot = alpha
        s.manipulator.joint_velocities,   # theta_dot
        theta_ddot,                       # theta_ddot
    ])


def rk4_step(model, x, u, dt):
    """Single RK4 integration step with quaternion normalization.

    Args:
        model: AerialManipulatorModel
        x:     state vector (17,)
        u:     input vector (8,), held constant over the step
        dt:    time step (s)

    Returns:
        x_next: state vector after one step (17,)
    """
    k1 = state_derivative(model, x, u)
    k2 = state_derivative(model, x + 0.5 * dt * k1, u)
    k3 = state_derivative(model, x + 0.5 * dt * k2, u)
    k4 = state_derivative(model, x + dt * k3, u)

    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Normalize quaternion
    x_next[6:10] = quat_normalize(x_next[6:10])

    return x_next


def simulate(model, x0, u_func, t_span, dt):
    """Simulate the system over a time span.

    Args:
        model:  AerialManipulatorModel
        x0:     initial state vector (17,)
        u_func: callable(t) -> input vector (8,)
        t_span: (t_start, t_end)
        dt:     time step (s)

    Returns:
        t_hist: time array (N+1,)
        x_hist: state history (N+1, 17)
    """
    t_start, t_end = t_span
    steps = int(np.ceil((t_end - t_start) / dt))
    t_hist = np.linspace(t_start, t_start + steps * dt, steps + 1)
    x_hist = np.zeros((steps + 1, len(x0)))
    x_hist[0] = x0

    x = x0.copy()
    for i in range(steps):
        u = u_func(t_hist[i])
        x = rk4_step(model, x, u, dt)
        x_hist[i + 1] = x

    return t_hist, x_hist


if __name__ == '__main__':
    from .dynamics import inverse_dynamics

    m = AerialManipulatorModel()
    total_mass = m.total_mass

    # --- Hover hold: constant thrust = total weight ---
    s0 = SystemState()
    s0.platform.position[:] = [0, 0, 1.0]
    x0 = s0.to_vector()

    hover_F = np.array([0, 0, total_mass * 9.81])
    u_hover = np.concatenate([hover_F, np.zeros(3 + N_JOINTS)])

    t, x = simulate(m, x0, lambda _t: u_hover, (0, 1.0), 0.01)
    assert np.allclose(x[-1, :3], [0, 0, 1.0], atol=1e-5), f'pos={x[-1,:3]}'
    assert np.allclose(x[-1, 3:6], 0, atol=1e-5), f'vel={x[-1,3:6]}'
    assert np.allclose(x[-1, 6:10], [0, 0, 0, 1], atol=1e-5), f'q={x[-1,6:10]}'
    print('Hover hold (1 s): PASSED')

    # --- Free fall for 0.5 s ---
    u_zero = np.zeros(6 + N_JOINTS)
    t2, x2 = simulate(m, x0, lambda _t: u_zero, (0, 0.5), 0.01)
    z_expected = 1.0 + 0.5 * (-9.81) * 0.5**2
    vz_expected = -9.81 * 0.5
    assert np.allclose(x2[-1, 2], z_expected, atol=1e-4), f'z={x2[-1,2]}, exp={z_expected}'
    assert np.allclose(x2[-1, 5], vz_expected, atol=1e-4), f'vz={x2[-1,5]}'
    assert np.allclose(x2[-1, :2], 0, atol=1e-10), 'xy drift'
    print('Free fall (0.5 s): PASSED')

    # --- Hover with arm at theta1=90, gravity compensation ---
    s90 = SystemState()
    s90.platform.position[:] = [0, 0, 1.0]
    s90.manipulator.joint_angles[:] = [np.pi / 2, 0.0]
    x90 = s90.to_vector()
    F90, tau90, jt90 = inverse_dynamics(
        m, s90.platform.quaternion, s90.platform.position,
        np.zeros(3), np.zeros(3), s90.manipulator.joint_angles,
        np.zeros(N_JOINTS), np.zeros(3), np.zeros(3), np.zeros(N_JOINTS))
    u90 = np.concatenate([F90, tau90, jt90])
    t3, x3 = simulate(m, x90, lambda _t: u90, (0, 1.0), 0.01)
    assert np.allclose(x3[-1, :3], [0, 0, 1.0], atol=1e-4), f'pos={x3[-1,:3]}'
    assert np.allclose(x3[-1, 13:15], [np.pi / 2, 0], atol=1e-4), f'theta={x3[-1,13:15]}'
    print('Hover with arm at 90 deg (1 s): PASSED')

    # --- Quaternion norm preserved over long sim ---
    q_norms = np.linalg.norm(x[:, 6:10], axis=1)
    assert np.allclose(q_norms, 1.0, atol=1e-10), f'quat norm range: {q_norms.min()}-{q_norms.max()}'
    print('Quaternion norm preserved: PASSED')

    # --- Coast: no gravity, constant platform velocity, arm at rest ---
    import copy
    m_nograv = copy.deepcopy(m)
    m_nograv.gravity[:] = 0.0
    s_coast = SystemState()
    s_coast.platform.position[:] = [0, 0, 1.0]
    s_coast.platform.velocity[:] = [1.0, 0.0, 0.0]
    x_coast = s_coast.to_vector()
    t4, x4 = simulate(m_nograv, x_coast, lambda _t: np.zeros(6 + N_JOINTS),
                       (0, 1.0), 0.01)
    assert np.allclose(x4[-1, 0], 1.0, atol=1e-4), f'x={x4[-1,0]}'
    assert np.allclose(x4[-1, 3], 1.0, atol=1e-4), f'vx={x4[-1,3]}'
    assert np.allclose(x4[-1, 2], 1.0, atol=1e-4), f'z={x4[-1,2]}'
    print('Coast (no gravity, 1 s): PASSED')

    print()
    print('All simulator tests passed!')
