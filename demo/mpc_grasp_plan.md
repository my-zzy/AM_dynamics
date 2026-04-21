# MPC Grasp Task — Design & Implementation Plan

## 1. Overview

**Goal:** Replace the decoupled PID approach in `grasp_task.py` with a unified Nonlinear MPC (NMPC) controller that treats the aerial-manipulator system as one coupled plant, drives the end-effector along a prescribed trajectory, and stops exactly at the target grasp pose.

**Solver:** [acados](https://docs.acados.org/) via the Python interface (`acados_template`), with CasADi for symbolic dynamics.

**Key improvement over PID:** The MPC sees the full coupled 17-DOF Newton-Euler model, so arm-induced reaction forces are anticipated rather than corrected after the fact.

---

## 2. System Summary

### State vector $x \in \mathbb{R}^{17}$

| Slice | Symbol | Description |
|---|---|---|
| 0:3 | $p_A$ | Platform position (world) |
| 3:6 | $v_A$ | Platform linear velocity |
| 6:10 | $q_A$ | Platform quaternion $[x,y,z,w]$ |
| 10:13 | $\omega_A$ | Platform angular velocity (body frame) |
| 13:15 | $\theta$ | Joint angles $[\theta_1,\theta_2]$ |
| 15:17 | $\dot\theta$ | Joint velocities |

### Control input $u \in \mathbb{R}^8$

| Slice | Symbol | Description |
|---|---|---|
| 0:3 | $F_{ext}$ | Net external force on platform (world) |
| 3:6 | $\tau_{ext}$ | Net external torque on platform (body) |
| 6:8 | $\tau_j$ | Joint torques $[\tau_1, \tau_2]$ |

### End-effector position

$$p_{EE} = p_{EE}(p_A,\, q_A,\, \theta) \in \mathbb{R}^3$$

computed by the forward kinematics chain in `ams/kinematics.py`.

---

## 3. Problem Formulation

### 3.1 Trajectory

Generate a smooth **minimum-jerk polynomial** trajectory from the initial EE pose to the grasp point in $T_f$ seconds (e.g. 4 s):

$$p_{EE}^{ref}(t),\quad \dot{p}_{EE}^{ref}(t), \quad t \in [0, T_f]$$

Boundary conditions:
- $t=0$: current EE position, zero velocity
- $t=T_f$: grasp position, **zero velocity** (hard terminal constraint)

The trajectory is sampled once at the MPC time-step $\Delta t$ to produce a reference sequence $\{p_k^{ref}\}_{k=0}^{N}$.

### 3.2 Optimal Control Problem (OCP)

At each real-time step, solve:

$$\min_{\{u_k\}_{k=0}^{N-1}} \sum_{k=0}^{N-1} \ell(x_k, u_k) + \ell_f(x_N)$$

subject to:

$$x_{k+1} = F_{RK4}(x_k, u_k, \Delta t)$$
$$\|q_{A,k}\| = 1 \quad (\text{quaternion norm constraint})$$
$$x_{min} \leq x_k \leq x_{max}$$
$$u_{min} \leq u_k \leq u_{max}$$
$$\|q_{A,N}\| = 1,\quad p_{EE}(x_N) = p_{EE}^{ref}(T_f),\quad v_A(x_N) = 0,\quad \dot\theta(x_N) = 0$$

### 3.3 Stage Cost

$$\ell(x_k, u_k) = \underbrace{(p_{EE,k} - p_k^{ref})^T Q_{ee} (p_{EE,k} - p_k^{ref})}_{\text{EE tracking}} + \underbrace{u_k^T R\, u_k}_{\text{effort}} + \underbrace{\Delta u_k^T R_{\Delta}\, \Delta u_k}_{\text{rate}} + w_v \|v_A\|^2 + w_\omega \|\omega_A\|^2$$

### 3.4 Terminal Cost / Constraint

**Hard terminal equality** (acados `ocp.constraints.lh_e`):

$$p_{EE}(x_N) = p_{EE}^{ref}(T_f), \quad v_A(x_N) = 0, \quad \dot\theta(x_N) = 0$$

This enforces exact stopping at the grasp point.

---

## 4. Implementation Plan

### Phase 1 — CasADi Symbolic Dynamics (`ams/casadi_dynamics.py`)

**Goal:** Re-express the full `forward_dynamics` pipeline symbolically so acados can auto-differentiate it.

| Step | Task |
|---|---|
| 1.1 | Port `math_utils.py` to CasADi: `ca_quat_to_rotmat`, `ca_skew`, `ca_quat_deriv` |
| 1.2 | Port `model.py` DH parameters — pure numeric, no porting needed |
| 1.3 | Port `kinematics.py` FK, velocity, acceleration recursions using `ca.MX` |
| 1.4 | Port `dynamics.py` backward recursion + mass-matrix construction |
| 1.5 | Expose `ca_state_derivative(x, u) -> x_dot` as a CasADi function |
| 1.6 | Wrap the EE position as a CasADi expression `ca_ee_pos(x)` |
| 1.7 | Validate: compare numeric output of `ca_state_derivative` vs `state_derivative` for random $(x,u)$ |

> **Quaternion handling:** Use acados' built-in quaternion norm constraint support (`ocp.model.con_h_expr` with `norm_sq(q) - 1 = 0`) rather than re-parametrizing the state. This keeps the state definition consistent with the existing simulator.

### Phase 2 — Trajectory Generator (`demo/mpc_trajectory.py`)

| Step | Task |
|---|---|
| 2.1 | Implement `min_jerk_traj(p0, pf, T_f, dt)` returning sampled arrays |
| 2.2 | Wrap as a class `EETrajectory` with `get_ref(t)` method |
| 2.3 | Verify zero-velocity boundary conditions at both ends |
| 2.4 | Optional: support waypoints for avoid-obstacle paths |

### Phase 3 — acados OCP Setup (`demo/mpc_controller.py`)

| Step | Task |
|---|---|
| 3.1 | Define `AcadosModel` using `ca_state_derivative`, state name `x`, control name `u` |
| 3.2 | Set `ocp.dims`: $n_x=17$, $n_u=8$, $N$=prediction horizon (start with $N=20$, $\Delta t = 0.05$ s) |
| 3.3 | Define cost using `NONLINEAR_LS` with residual functions for EE tracking + effort |
| 3.4 | Add quaternion norm equality constraint via nonlinear path constraint |
| 3.5 | Add state/input box constraints (joint limits $\pm\pi/2$, thrust bounds, torque limits) |
| 3.6 | Add terminal constraints: `lh_e`/`uh_e` for $p_{EE}$, $v_A$, $\dot\theta$ |
| 3.7 | Configure solver: `FULL_CONDENSING_HPIPM`, SQP-RTI for real-time iteration |
| 3.8 | Generate C code and compile |
| 3.9 | Wrap in a `MPCController` class with `solve(x_current, t_current)` → `u` |

**Suggested initial weights:**

| Weight | Value | Tuning note |
|---|---|---|
| $Q_{ee}$ | $100 \cdot I_3$ | EE tracking, reduce if infeasible |
| $R$ | $\text{diag}(0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 1, 1)$ | Torques more expensive |
| $R_\Delta$ | $0.01 \cdot I_8$ | Rate penalty for smoothness |
| $w_v, w_\omega$ | 5, 5 | Damp platform oscillation |
| Terminal $Q_{ee,f}$ | $1000 \cdot I_3$ | Heavy terminal EE cost |

### Phase 4 — Closed-Loop Integration (`demo/mpc_grasp_task.py`)

| Step | Task |
|---|---|
| 4.1 | Initialize MuJoCo, load `grasp_scene.xml` |
| 4.2 | On takeoff completion, activate MPC |
| 4.3 | At each MuJoCo step: read state → `get_grasp_state` → pack `x` → `MPCController.solve` → apply $u_0$ |
| 4.4 | Convert $F_{ext}, \tau_{ext}$ to MuJoCo rotor commands (inverse allocation matrix) |
| 4.5 | Apply $\tau_j$ directly to joint actuators |
| 4.6 | Detect terminal condition: $\|p_{EE} - p_{EE}^{ref}(T_f)\| < \epsilon$ and $\|v_{EE}\| < \epsilon_v$ → trigger gripper close |
| 4.7 | After grasp, switch to a hold MPC (frozen terminal reference) or PID for lift |

### Phase 5 — Testing & Validation

| Step | Task |
|---|---|
| 5.1 | Open-loop check: simulate OCP solution on `ams/simulator.py` (no MuJoCo) |
| 5.2 | Verify EE reaches target with zero velocity |
| 5.3 | Closed-loop sim in MuJoCo with perfect state feedback |
| 5.4 | Check coupling compensation: compare EE tracking error vs PID baseline |
| 5.5 | Tune weights and horizon until tracking error < 5 mm at terminal time |
| 5.6 | Test robustness: add 5% mass uncertainty, ±0.01 m position noise |

---

## 5. File Structure

```
AM_dynamics/
├── ams/
│   ├── casadi_dynamics.py      ← NEW: symbolic dynamics for acados
│   └── ... (existing files unchanged)
├── demo/
│   ├── mpc_trajectory.py       ← NEW: minimum-jerk trajectory generator
│   ├── mpc_controller.py       ← NEW: acados OCP setup + MPCController class
│   ├── mpc_grasp_task.py       ← NEW: closed-loop MuJoCo integration
│   └── ... (existing files unchanged)
└── README.md
```

---

## 6. Key Technical Notes

### Quaternion in acados
acados supports quaternion states directly. Define the quaternion norm as a nonlinear constraint:
```python
q = x[6:10]
ocp.model.con_h_expr = ca.dot(q, q) - 1.0   # = 0 at every stage
```
This avoids re-parametrization and is consistent with the existing state definition.

### SQP-RTI for real-time
Use `ocp_solver.options.nlp_solver_type = 'SQP_RTI'` for a single SQP iteration per MPC step (≈1 ms on modern hardware for this problem size), suitable for MuJoCo's 2 ms timestep.

### Warm-starting
At each time step, shift the previous solution by one step and use it as the initial guess for the next SQP-RTI call. acados handles this via `ocp_solver.set_flat('x_init', x_init_shifted)`.

### Input mapping
The `u = [F_ext, tau_ext, tau_j]` control input corresponds to the fully-actuated platform model. In MuJoCo, apply:
- **Platform forces/torques:** inject as body external wrench via `data.xfrc_applied` for the platform body
- **Joint torques:** write to `data.ctrl[0:2]`

---

## 7. Dependencies

```
acados (with Python interface, acados_template)
casadi >= 3.6
numpy
mujoco
```

Install acados: follow https://docs.acados.org/installation/index.html and set `ACADOS_SOURCE_DIR`.

---

## 8. Success Criteria

| Metric | Target |
|---|---|
| Terminal EE position error | < 5 mm |
| Terminal EE velocity | < 0.01 m/s |
| Platform attitude deviation during approach | < 5° |
| Solve time per step | < 2 ms (SQP-RTI) |
| Successful grasp rate in MuJoCo | ≥ 90% over 10 runs |
