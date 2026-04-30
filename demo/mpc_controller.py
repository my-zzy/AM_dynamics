"""Nonlinear MPC controller for the Aerial Manipulator grasp task.

Uses acados (with CasADi symbolic dynamics from ams/casadi_dynamics.py)
to set up and solve the OCP at real-time rates via SQP-RTI.

Quick start
-----------
    from ams.model import AerialManipulatorModel
    from demo.mpc_trajectory import EETrajectory
    from demo.mpc_controller import MPCController

    model = AerialManipulatorModel()
    traj  = EETrajectory(p_start, p_end, T_f=4.0, dt=0.05)
    ctrl  = MPCController(traj, model)        # compiles acados C code once

    u0    = ctrl.solve(x_current, t_current)  # call every control step

OCP formulation
---------------
    State  x ∈ ℝ¹⁷   [p_A(3), v_A(3), q_A(4), ω_A(3), θ(2), θ̇(2)]
    Input  u ∈ ℝ⁸    [F_body(3), τ_body(3), τ_j(2)]
                     F_body = [0, 0, T] (thrust along body-z); rotated to world in OCP dynamics.
                     τ_body = roll/pitch/yaw body torques; rotated to world in OCP dynamics.
    Horizon N steps, step size dt  →  prediction window = N·dt seconds

Stage cost (NONLINEAR_LS, residual y ∈ ℝ²⁷):
    y = [p_EE(x)(3), v_A(3), ω_A(3), θ(2), u(8), Δu(8)]
    Δu = u_k − u_{k-1}   (control rate; u_{k-1} passed as a model parameter)
    ℓ = ½ (y - y_ref)ᵀ W (y - y_ref)

Terminal cost (residual y_e ∈ ℝ⁹):
    y_e = [p_EE(x)(3), v_A(3), ω_A(3)]

Terminal constraints (hard equality):
    p_EE(x_N)  = p_EE_ref_final          (3)
    v_A(x_N)   = 0                        (3)
    θ̇(x_N)    = 0                        (2)

Path constraints:
    ‖q_A‖² = 1   (quaternion unit-norm, soft via large penalty or hard)
    joint limits, force/torque limits
"""

import os
import sys
import numpy as np
import casadi as ca

# ---------------------------------------------------------------------------
# Default weights and limits
# ---------------------------------------------------------------------------

# Stage cost weight matrix diagonal (ny = 17)
# [p_EE(3), v_A(3), omega_A(3), u(8)]
_W_STAGE_DIAG = np.array([
    20.0, 20.0, 20.0,    # EE position tracking
      2.0,   2.0,   2.0,    # platform linear velocity damping
      2.0,   2.0,   2.0,    # platform angular velocity damping
      0.02,   0.02,   0.02,    # F_ext effort
      0.02,   0.02,   0.02,    # tau_ext effort
      0.05,   0.05,         # joint torque effort (high: discourages arm motion)
])

# Control rate (delta-u) weight diagonal (n_delta_u = 8)
# Penalises u_k - u_{k-1} to smooth the input trajectory
# [F_ext(3), tau_ext(3), tau_j(2)]
_W_RATE_DIAG = np.array([
    3.0, 3.0, 3.0,   # F_ext rate
    3.0, 3.0, 3.0,   # tau_ext rate
    5.0, 5.0,        # joint torque rate (discourages jerky arm motion)
])

# Joint angle (theta) cost diagonal — penalises arm deflection from zero.
# High weight forces the optimizer to fly the drone rather than bend the arm.
# [theta1, theta2]
_W_JOINT_ANGLE_DIAG = np.array([5.0, 5.0])

# Terminal cost weight diagonal (ny_e = 9)
# [p_EE(3), v_A(3), omega_A(3)]
_W_TERMINAL_DIAG = np.array([
    80.0, 80.0, 80.0,  # terminal EE position (heavy)
      10.0,   10.0,   10.0,  # terminal velocity
      10.0,   10.0,   10.0,  # terminal angular velocity
])

# Input bounds  [F_ext(3), tau_ext(3), tau_j(2)]
# Quadrotor thrust is body-z only: F_ext[0]=F_ext[1]=0, F_ext[2] >= 0
_U_MIN = np.array([ 0.0,  0.0,  0.0,  -2.0, -2.0, -2.0,  -0.5, -0.5])
_U_MAX = np.array([ 0.0,  0.0, 30.0,   2.0,  2.0,  2.0,   0.5,  0.5])

# State bounds (only joint angles and velocities are tightly bounded)
# Format: lower/upper for all 17 states
# Use ±1e9 for unconstrained
_INF = 1e9
_X_MIN = np.array([
    -_INF, -_INF, 0.0,          # p_A  (z >= 0 : no underground)
    -_INF, -_INF, -_INF,        # v_A
    -1.0, -1.0, -1.0, -1.0,    # q_A  (unit sphere, soft via dynamics)
    -_INF, -_INF, -_INF,        # omega_A
    -np.deg2rad(20), -np.deg2rad(20),  # theta joint limits (±20° — nearly fixed)
    -1.0, -1.0,                         # theta_dot velocity limits (slow)
])
_X_MAX = np.array([
    _INF, _INF, _INF,
    _INF, _INF, _INF,
    1.0, 1.0, 1.0, 1.0,
    _INF, _INF, _INF,
    np.deg2rad(20), np.deg2rad(20),
    1.0, 1.0,
])


# ---------------------------------------------------------------------------
# Helper: build acados OCP description
# ---------------------------------------------------------------------------

def _build_ocp(traj, model, N, dt,
               w_stage=None, w_terminal=None, w_rate=None, w_joint_angle=None,
               u_min=None, u_max=None,
               x_min=None, x_max=None,
               enable_terminal_constraint=True,
               code_export_dir=None):
    """Construct the AcadosOcp object (does not compile yet).

    Parameters
    ----------
    traj    : EETrajectory   used only for the terminal reference p_EE_ref
    model   : AerialManipulatorModel
    N       : int            prediction horizon (shooting nodes)
    dt      : float          time step (s)
    w_stage, w_terminal : optional diagonal weight arrays
    u_min, u_max        : optional input bound arrays
    x_min, x_max        : optional state bound arrays
    enable_terminal_constraint : bool  add hard terminal equality constraints
    code_export_dir     : str   directory for generated C code
    """
    try:
        from acados_template import AcadosOcp, AcadosModel
    except ImportError as e:
        raise ImportError(
            'acados_template not found. '
            'Install acados and set ACADOS_SOURCE_DIR. '
            'See https://docs.acados.org/installation/') from e

    import sys, os
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), '..'))
    from ams.casadi_dynamics import (
        build_ca_dynamics, _ca_state_derivative_expr,
        ca_forward_kinematics,
    )

    w_stage       = _W_STAGE_DIAG        if w_stage       is None else w_stage
    w_terminal    = _W_TERMINAL_DIAG     if w_terminal    is None else w_terminal
    w_rate        = _W_RATE_DIAG         if w_rate        is None else w_rate
    w_joint_angle = _W_JOINT_ANGLE_DIAG  if w_joint_angle is None else w_joint_angle
    u_min         = _U_MIN               if u_min         is None else u_min
    u_max         = _U_MAX               if u_max         is None else u_max
    x_min         = _X_MIN               if x_min         is None else x_min
    x_max         = _X_MAX               if x_max         is None else x_max

    nx = 17
    nu = 8
    # y = [p_EE(3), v_A(3), omega_A(3), theta(2), u(8), delta_u(8)] = 27
    ny   = 3 + 3 + 3 + 2 + nu + nu   # 27  stage residual
    ny_e = 3 + 3 + 3                  # 9   terminal residual

    # ----------------------------------------------------------------
    # Symbolic model
    # ----------------------------------------------------------------
    x_sym = ca.MX.sym('x', nx)
    u_sym = ca.MX.sym('u', nu)

    x_dot_expr = _ca_state_derivative_expr(model, x_sym, u_sym)

    # Model parameter: previous control input (for control-rate cost Δu = u - u_prev)
    u_prev_sym = ca.MX.sym('u_prev', nu)

    acados_model = AcadosModel()
    acados_model.name         = 'am_mpc'
    acados_model.x            = x_sym
    acados_model.u            = u_sym
    acados_model.p            = u_prev_sym
    acados_model.xdot         = ca.MX.sym('xdot', nx)
    acados_model.f_expl_expr  = x_dot_expr     # explicit ODE

    # EE position expression (shared)
    p_A_sym   = x_sym[0:3]
    q_A_sym   = x_sym[6:10]
    theta_sym = x_sym[13:15]
    R_sym, p_sym, _ = ca_forward_kinematics(model, q_A_sym, p_A_sym, theta_sym)
    p_ee_expr = p_sym[3]          # (3,1)

    # ----------------------------------------------------------------
    # Cost — NONLINEAR_LS
    # ----------------------------------------------------------------
    v_A_sym    = x_sym[3:6]
    omega_A_sym= x_sym[10:13]

    # Stage residual y = [p_ee, v_A, omega_A, theta, u, Δu]
    # theta ref = 0 (arm folded): penalises arm deflection, forces drone to fly
    # Δu = u - u_prev  where u_prev is passed as a model parameter per node
    delta_u_expr = u_sym - u_prev_sym
    y_expr   = ca.vertcat(p_ee_expr, v_A_sym, omega_A_sym, theta_sym, u_sym, delta_u_expr)
    # Terminal residual y_e = [p_ee, v_A, omega_A]
    y_e_expr = ca.vertcat(p_ee_expr, v_A_sym, omega_A_sym)

    acados_model.cost_y_expr   = y_expr
    acados_model.cost_y_expr_e = y_e_expr

    # Quaternion norm constraint removed — ERK integration keeps |q|≈1
    # sufficiently well for short horizons; the nonlinear path constraint
    # was causing SQP line-search failures (ACADOS_MINSTEP).

    # Terminal constraints: p_EE = p_final, v_A = 0, theta_dot = 0
    if enable_terminal_constraint:
        theta_dot_sym = x_sym[15:17]
        con_h_e = ca.vertcat(p_ee_expr, v_A_sym, theta_dot_sym)
        acados_model.con_h_expr_e = con_h_e   # shape (8,)

    # ----------------------------------------------------------------
    # OCP object
    # ----------------------------------------------------------------
    ocp = AcadosOcp()
    ocp.model = acados_model

    # Dimensions
    ocp.dims.N = N

    # Cost type
    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    # W layout: [w_stage[0:9]=EE+vel+omega, w_joint_angle=theta, w_stage[9:17]=u, w_rate=delta_u]
    ocp.cost.W   = np.diag(np.concatenate([w_stage[:9], w_joint_angle, w_stage[9:], w_rate]))
    ocp.cost.W_e = np.diag(w_terminal)

    # Initial references (updated at solve time)
    ocp.cost.yref   = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # Initial parameter values (u_prev = 0; updated at solve time)
    ocp.parameter_values = np.zeros(nu)

    # ----------------------------------------------------------------
    # Constraints
    # ----------------------------------------------------------------

    # Terminal constraints
    if enable_terminal_constraint:
        p_ee_final = traj._p(traj.T_f)  # (3,)
        # [p_EE(3), v_A(3), theta_dot(2)]
        lh_e = np.concatenate([p_ee_final, np.zeros(3), np.zeros(2)])
        uh_e = lh_e.copy()
        tol_e = 1e-2                     # 1 cm / 0.01 rad-s^-1 tolerance
        lh_e[:3] -= tol_e
        uh_e[:3] += tol_e
        lh_e[3:] -= tol_e
        uh_e[3:] += tol_e
        ocp.constraints.lh_e = lh_e
        ocp.constraints.uh_e = uh_e

    # Input bounds
    ocp.constraints.lbu   = u_min
    ocp.constraints.ubu   = u_max
    ocp.constraints.idxbu = np.arange(nu, dtype=int)

    # State bounds (joint angles + velocities)
    # Indices: 13,14 (theta), 15,16 (theta_dot), 2 (z >= 0)
    idx_sbx = np.array([2, 13, 14, 15, 16], dtype=int)
    ocp.constraints.lbx   = np.array([x_min[i] for i in idx_sbx])
    ocp.constraints.ubx   = np.array([x_max[i] for i in idx_sbx])
    ocp.constraints.idxbx = idx_sbx

    # Initial state constraint (set at runtime via set(0,'lbx'/'ubx'))
    ocp.constraints.x0 = np.zeros(nx)

    # ----------------------------------------------------------------
    # Solver options
    # ----------------------------------------------------------------
    ocp.solver_options.tf                        = N * dt
    ocp.solver_options.integrator_type           = 'ERK'
    ocp.solver_options.sim_method_num_stages     = 4      # RK4
    ocp.solver_options.sim_method_num_steps      = 1
    ocp.solver_options.nlp_solver_type           = 'SQP'    # SQP_RTI
    ocp.solver_options.qp_solver                 = 'FULL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx            = 'GAUSS_NEWTON'
    ocp.solver_options.nlp_solver_max_iter       = 50     # full SQP for diagnosis
    ocp.solver_options.qp_solver_cond_N          = N
    ocp.solver_options.print_level               = 0

    if code_export_dir is not None:
        ocp.code_export_directory = code_export_dir

    return ocp, ny, ny_e


# ---------------------------------------------------------------------------
# MPCController
# ---------------------------------------------------------------------------

class MPCController:
    """Real-time NMPC controller for the aerial manipulator.

    Parameters
    ----------
    traj    : EETrajectory
        Pre-computed EE reference trajectory.
    model   : AerialManipulatorModel, optional
        Robot model (default parameters if None).
    N       : int, optional
        Prediction horizon in steps (default 20).
    dt      : float, optional
        Shooting step size in seconds (default 0.05 s).
    rebuild : bool, optional
        Force recompilation of acados C code (default False: reuse if found).
    enable_terminal_constraint : bool, optional
        Activate hard terminal equality for p_EE and velocities (default True).
    **weight_kwargs :
        Override weight / bound arrays:
        ``w_stage``, ``w_terminal``, ``u_min``, ``u_max``, ``x_min``, ``x_max``.
    """

    def __init__(self, traj, model=None, N=20, dt=0.05,
                 rebuild=False, enable_terminal_constraint=True,
                 **weight_kwargs):
        try:
            from acados_template import AcadosOcpSolver
        except ImportError as e:
            raise ImportError(
                'acados_template not found. '
                'Install acados and set ACADOS_SOURCE_DIR.') from e

        if model is None:
            from ams.model import AerialManipulatorModel
            model = AerialManipulatorModel()

        self.traj  = traj
        self.model = model
        self.N     = N
        self.dt    = dt
        self.nx    = 17
        self.nu    = 8

        # Code export directory alongside this file
        code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'acados_generated_am_mpc')

        ocp, self._ny, self._ny_e = _build_ocp(
            traj, model, N, dt,
            enable_terminal_constraint=enable_terminal_constraint,
            code_export_dir=code_dir,
            **weight_kwargs,
        )

        json_file = os.path.join(code_dir, 'acados_ocp_am_mpc.json')
        build_flag = rebuild or not os.path.exists(json_file)

        self._solver = AcadosOcpSolver(
            ocp,
            json_file=json_file,
            build=build_flag,
            generate=build_flag,
        )

        # Gravity compensation torques at L-shape (θ₁=θ₂=0, XML zero config).
        # At this config:
        #   - link1 COM is directly below joint1 → zero moment arm → τ_j1 from link1 = 0
        #   - link2+EE COM is at +com_x along world +x from joint2
        #     → torque about z₀=[0,+1,0]_world = m₂·g·com_x about both joints
        # Required motor torque (positive counteracts gravity for +θ₂ = dip down):
        _m2  = model.links[1].mass
        _cx2 = model.links[1].com_offset[0]   # 0.12727 m
        _tau = _m2 * 9.81 * _cx2              # ≈ 0.274 Nm magnitude
        # At L-shape (θ=0 XML), link2+EE COM is at x_com=0.127m forward of mount.
        # This creates a gravity pitch torque on the platform of −m·g·x_com about world +y.
        # Both the joint motors AND the platform rotors must compensate:
        #   τ_j (both joints) = −0.274 Nm  (hold arm up)
        #   τ_ext[1] (platform pitch) = −0.274 Nm  (counter arm moment on drone body)
        self._tau_grav     = np.array([-_tau, -_tau])        # [τ_j1, τ_j2]
        self._tau_ext_grav = np.array([0.0, -_tau, 0.0])     # platform pitch compensation

        # Warm-start arrays  (N+1 states, N inputs)
        self._x_init = np.zeros((N + 1, self.nx))
        self._u_init = np.zeros((N,     self.nu))

        # Gravity-compensating warm-start: hover force + pitch compensation + arm torques
        self._u_init[:, 2]   = model.total_mass * 9.81  # F_ext_z
        self._u_init[:, 3:6] = self._tau_ext_grav        # tau_ext: platform pitch hold
        self._u_init[:, 6:8] = self._tau_grav            # tau_j1, tau_j2

        self._initialized = False
        self._enable_tc   = enable_terminal_constraint

        # Last applied control — used as u_prev at node 0 for the rate cost.
        # Initialised to the hover warm-start so the first solve starts smooth.
        self._u_prev = self._u_init[0].copy()

    # ------------------------------------------------------------------
    # Reference update helpers
    # ------------------------------------------------------------------

    def _update_references(self, t_current):
        """Push the sliding EE reference window into the solver."""
        # Terminal cost reference: end of the current horizon window, not T_f.
        # This ensures the terminal cost pulls toward an achievable point
        # instead of the 4s-away final target, which causes a cost spike.
        t_horizon_end = t_current + self.N * self.dt
        p_ee_ref_final = self.traj._p(t_horizon_end)

        # Set per-node u_prev parameter for the control rate cost term.
        # Node 0 uses the last actually-applied control; later nodes use the
        # warm-started previous control from the shifted solution array.
        self._solver.set(0, 'p', self._u_prev)
        for k in range(1, self.N):
            self._solver.set(k, 'p', self._u_init[k - 1])

        for k in range(self.N):
            t_k   = t_current + k * self.dt
            p_ref = self.traj._p(t_k)          # (3,)

            # Stage reference  [p_EE(3), v_A(3), omega_A(3)=0, theta(2)=0, u(8)=hover, delta_u(8)=0]
            # y layout: p_EE[0:3], v_A[3:6], omega[6:9], theta[9:11], u[11:19], du[19:27]
            # v_A feedforward ≈ v_EE (valid for near-hover, small arm angles).
            v_ref           = self.traj._v(t_k)             # (3,) EE velocity feedforward
            yref_k          = np.zeros(self._ny)
            yref_k[0:3]     = p_ref
            yref_k[3:6]     = v_ref                         # platform velocity feedforward
            # theta ref = 0 (arm folded) — already zero
            yref_k[13]      = self.model.total_mass * 9.81  # u[2]=F_ext_z hover  (11+2=13)
            yref_k[15]      = self._tau_ext_grav[1]         # u[4]=tau_ext_y      (11+4=15)
            yref_k[17:19]   = self._tau_grav                # u[6:8]=tau_j        (11+6:8=17:19)
            self._solver.cost_set(k, 'yref', yref_k)

        # Terminal reference  [p_EE_final(3), v_A=0, omega_A=0]
        yref_e      = np.zeros(self._ny_e)
        yref_e[0:3] = p_ee_ref_final
        self._solver.cost_set(self.N, 'yref', yref_e)

        # Update hard terminal constraint bounds if enabled
        if self._enable_tc:
            p_ee_final = self.traj._p(self.traj.T_f)
            tol_e = 1e-2
            lh_e = np.concatenate([p_ee_final - tol_e,
                                    -tol_e * np.ones(3),
                                    -tol_e * np.ones(2)])
            uh_e = np.concatenate([p_ee_final + tol_e,
                                    tol_e * np.ones(3),
                                    tol_e * np.ones(2)])
            self._solver.constraints_set(self.N, 'lh', lh_e)
            self._solver.constraints_set(self.N, 'uh', uh_e)

    # ------------------------------------------------------------------
    # Warm-start initialisation
    # ------------------------------------------------------------------

    def _init_warm_start(self, x0, t_current):
        """Fill x_init by interpolating platform position along the EE trajectory.

        Rather than rolling out with hover inputs (which leaves the drone
        stationary), we shift the platform position proportionally to the EE
        trajectory motion.  This gives the SQP a physically plausible initial
        guess even when the trajectory requires large translations.
        """
        p_A_start = x0[0:3].copy()

        # EE displacement over the prediction horizon
        p_ee_now = self.traj._p(t_current)
        t_end    = t_current + self.N * self.dt
        p_ee_end = self.traj._p(t_end)
        delta_ee = p_ee_end - p_ee_now   # how much EE needs to move

        # Platform moves by the same vector (arm angles stay at zero)
        p_A_end  = p_A_start + delta_ee
        v_A_mean = delta_ee / (self.N * self.dt)   # constant velocity hint

        self._x_init[0] = x0.copy()
        for k in range(1, self.N + 1):
            alpha = k / self.N
            xi       = x0.copy()
            xi[0:3]  = (1.0 - alpha) * p_A_start + alpha * p_A_end
            xi[3:6]  = v_A_mean          # smooth constant velocity
            self._x_init[k] = xi

        for k in range(self.N + 1):
            self._solver.set(k, 'x', self._x_init[k])
        for k in range(self.N):
            self._solver.set(k, 'u', self._u_init[k])

    def _shift_warm_start(self):
        """Shift solution arrays by one step for the next solve call."""
        self._x_init[:-1] = self._x_init[1:]
        self._u_init[:-1] = self._u_init[1:]
        # Last node: repeat final value
        self._x_init[-1]  = self._x_init[-2]
        self._u_init[-1]  = self._u_init[-2]

    # ------------------------------------------------------------------
    # Main solve method
    # ------------------------------------------------------------------

    def solve(self, x_current, t_current):
        """Solve one SQP-RTI step and return the first optimal control action.

        Parameters
        ----------
        x_current : array_like, shape (17,)
            Current system state vector.
        t_current : float
            Current time within the trajectory (seconds from traj start).

        Returns
        -------
        u0 : ndarray, shape (8,)
            Optimal first input to apply.
        info : dict
            ``status`` (int), ``cost`` (float), ``solve_time`` (float s).
        """
        x0 = np.asarray(x_current, dtype=float).flatten()

        # Renormalise quaternion in the initial state — ERK integration does
        # not enforce unit norm, so small drift accumulates over time.
        q_norm = np.linalg.norm(x0[6:10])
        if q_norm > 1e-8:
            x0[6:10] /= q_norm

        # Initialise warm-start on the first call
        if not self._initialized:
            self._init_warm_start(x0, t_current)
            self._initialized = True
        else:
            self._shift_warm_start()
            # Load shifted warm-start into solver
            for k in range(self.N + 1):
                self._solver.set(k, 'x', self._x_init[k])
            for k in range(self.N):
                self._solver.set(k, 'u', self._u_init[k])

        # Set current initial state constraint
        self._solver.set(0, 'lbx', x0)
        self._solver.set(0, 'ubx', x0)

        # Update sliding reference window
        self._update_references(t_current)

        # Solve
        status = self._solver.solve()

        # Extract solution for warm-starting next iteration.
        # Only update from a successful solve to avoid propagating NaN states.
        if status in (0, 2):  # 0=success, 2=max_iter (acceptable for RTI)
            for k in range(self.N + 1):
                xk = self._solver.get(k, 'x')
                # Renormalise quaternion to prevent drift-induced ill-conditioning.
                q_norm = np.linalg.norm(xk[6:10])
                if q_norm > 1e-8:
                    xk[6:10] /= q_norm
                self._x_init[k] = xk
            for k in range(self.N):
                self._u_init[k] = self._solver.get(k, 'u')

        u0 = self._solver.get(0, 'u')

        # Update u_prev so the next solve's rate cost references the
        # input we are about to apply.
        self._u_prev = u0.copy()

        cost       = self._solver.get_cost()
        solve_time = self._solver.get_stats('time_tot')

        if status not in (0, 2):   # 0=success, 2=max_iter (acceptable for RTI)
            import warnings
            warnings.warn(f'acados solver returned status {status} at t={t_current:.3f}s')

        return u0, {'status': status, 'cost': cost, 'solve_time': solve_time}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset(self):
        """Reset warm-start flag (e.g. after a large state jump)."""
        self._initialized = False
        self._u_prev = self._u_init[0].copy()

    def ee_position(self, x):
        """Compute EE world position from state vector using CasADi function."""
        if not hasattr(self, '_ee_pos_func'):
            from ams.casadi_dynamics import build_ca_dynamics
            _, self._ee_pos_func = build_ca_dynamics(self.model)
        return np.array(self._ee_pos_func(x)).flatten()


# ---------------------------------------------------------------------------
# Quick build test (does not require acados to pass; checks imports/shapes)
# ---------------------------------------------------------------------------

def _smoke_test():
    """Check that the OCP description is built without errors (no compilation)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from ams.model import AerialManipulatorModel
    from demo.mpc_trajectory import EETrajectory

    model = AerialManipulatorModel()
    p0    = np.array([0.0, 0.0, 0.85])
    pf    = np.array([0.25, 0.0, 0.70])
    traj  = EETrajectory(p0, pf, T_f=4.0, dt=0.05)

    N, dt = 20, 0.05

    ocp, ny, ny_e = _build_ocp(traj, model, N, dt,
                                enable_terminal_constraint=True,
                                code_export_dir='/tmp/am_mpc_smoke')

    assert ny   == 17, f'ny={ny}'
    assert ny_e == 9,  f'ny_e={ny_e}'
    assert ocp.cost.W.shape    == (17, 17)
    assert ocp.cost.W_e.shape  == (9, 9)
    assert ocp.constraints.lbu.shape == (8,)
    assert ocp.constraints.lbx.shape == (5,)   # z, theta1, theta2, td1, td2
    assert len(ocp.constraints.lh)   == 1      # quat norm
    if ocp.model.con_h_expr_e is not None:
        assert ocp.constraints.lh_e.shape == (8,)   # p_ee, v_A, theta_dot

    print('OCP description build: PASSED')
    print(f'  ny={ny}, ny_e={ny_e}, N={N}, dt={dt}')
    print(f'  Horizon window: {N*dt:.2f} s')
    print('Smoke test PASSED (no acados compilation required).')


if __name__ == '__main__':
    _smoke_test()
