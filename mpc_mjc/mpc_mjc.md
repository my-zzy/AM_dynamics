# MPC Controller with MuJoCo Simulation

## Overview

Build a nonlinear MPC controller for the aerial manipulator using CasADi dynamics
and validate it in MuJoCo simulation.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   MuJoCo    │────▶│     MPC     │────▶│   MuJoCo    │
│  (state x)  │     │  (solve u*) │     │ (apply u*)  │
└─────────────┘     └─────────────┘     └─────────────┘
        ▲                                      │
        └──────────────────────────────────────┘
```

## Architecture

```
mpc_mjc/
├── model/
│   ├── am_robot.xml          # MuJoCo MJCF model of the aerial manipulator
│   └── meshes/               # (optional) visual meshes
├── casadi_dynamics.py        # CasADi symbolic dynamics (port from ams/)
├── mpc_solver.py             # OCP formulation + CasADi/acados solver
├── mujoco_sim.py             # MuJoCo simulation loop + MPC interface
├── mpc_config.py             # All tuning parameters in one place
└── main.py                   # Entry point: run MPC + MuJoCo
```

---

## Step 1: MuJoCo Model (MJCF)

Build the aerial manipulator as a MuJoCo XML model.

- **Platform**: free-joint body (6 DOF), mass 1.5 kg, inertia diag(0.015, 0.015, 0.02)
- **Arm base**: fixed to platform at [0, 0, -0.05] with rotated frame
- **Link 1**: hinge joint (axis = z_0), length 0.25 m, mass 0.15 kg
- **Link 2**: hinge joint (axis = z_1), length 0.20 m, mass 0.12 kg
- **Actuators**:
  - 6 force/torque actuators on the platform free joint (or 4 rotors modeled explicitly)
  - 2 torque actuators on the arm joints
- **Gravity**: [0, 0, -9.81]

Key decisions:
- Use `<freejoint>` for the platform (MuJoCo gives quaternion [w,x,y,z] — need to convert to our [x,y,z,w])
- Match all masses, inertias, and geometry exactly to our `model.py`

**Deliverable**: `am_robot.xml` that loads in MuJoCo and matches our dynamics.

**Validation**: Drop the robot in MuJoCo with no control, compare free-fall and joint swing
trajectories against our `simulator.py` output.

---

## Step 2: CasADi Dynamics

Port the Newton-Euler dynamics from NumPy to CasADi symbolic expressions.

Port these functions (same math, `ca.SX` instead of `np.ndarray`):
1. `quat_to_rotation_matrix(q)` → `ca.SX` 3×3
2. `forward_kinematics(q_A, p_A, theta)` → symbolic R, p, p_com
3. `velocity_recursion(...)` → symbolic omega, v
4. `acceleration_recursion(...)` → symbolic alpha, a, a_com
5. `backward_recursion(...)` → symbolic f, tau at each joint
6. `forward_dynamics(x, u)` → symbolic a_A, alpha_A, theta_ddot

Final output: `ca.Function('f', [x, u], [x_dot])` — the continuous-time dynamics.

Tips:
- CasADi doesn't have `for i in reversed(range(n))` issues — loops over 2 links unroll fine
- Use `ca.cross()` instead of our `cross()`
- Use `ca.mtimes()` instead of `@`
- `np.linalg.solve(M, b)` → `ca.solve(M, b)` or explicitly invert (small 8×8 matrix)

**Validation**: Evaluate the CasADi function numerically at several test states
and compare against our NumPy `state_derivative()` output. Must match to ~1e-10.

---

## Step 3: MPC Formulation

Define the optimal control problem (OCP).

### State and Input

```
x = [p(3), v(3), q(4), ω(3), θ(2), θ̇(2)]     → 17 dimensions
u = [F(3), τ(3), τ_joint(2)]                    → 8 dimensions
```

### OCP

```
min   Σₖ (xₖ − xref)ᵀ Q (xₖ − xref) + uₖᵀ R uₖ  +  (xₙ − xref)ᵀ Qf (xₙ − xref)
s.t.  xₖ₊₁ = discrete_dynamics(xₖ, uₖ)       ∀ k = 0..N-1
      x₀ = x_current
      u_min ≤ uₖ ≤ u_max                        (actuator limits)
      θ_min ≤ θₖ ≤ θ_max                        (joint limits)
      ‖qₖ‖ = 1                                  (quaternion constraint)
```

### Tuning parameters (mpc_config.py)

| Parameter | Suggested start |
|-----------|----------------|
| Horizon N | 20 |
| dt | 0.02 s (50 Hz) |
| MPC rate | 50 Hz (solve every step) |
| Q (position) | diag(100, 100, 100) |
| Q (velocity) | diag(10, 10, 10) |
| Q (quaternion) | diag(50, 50, 50, 50) |
| Q (angular vel) | diag(10, 10, 10) |
| Q (joint angles) | diag(20, 20) |
| Q (joint vel) | diag(1, 1) |
| R (force) | diag(0.01, 0.01, 0.01) |
| R (torque) | diag(0.1, 0.1, 0.1) |
| R (joint torque) | diag(0.1, 0.1) |
| F_max | 30 N per axis |
| τ_max | 2 N·m per axis |
| τ_joint_max | 5 N·m |

### Solver choice

**Option A — acados (recommended for real-time)**
- RTI (Real-Time Iteration) scheme
- SQP with Gauss-Newton Hessian
- Generates C code → very fast
- Python interface: `AcadosOcpSolver`

**Option B — CasADi Opti (simpler, slower)**
- Direct multiple shooting
- IPOPT solver
- Good for prototyping, too slow for real-time

### Quaternion handling

Two options:
1. **Quaternion error in cost**: Use `q_err = q_ref⁻¹ ⊗ q` and penalize the vector part
2. **Ignore norm constraint**: Let the integrator handle it, just normalize after each MPC solve

Start with option 2 (simpler), switch to 1 if drift is a problem.

---

## Step 4: MuJoCo–MPC Loop

```python
# main.py pseudocode

model = mujoco.MjModel.from_xml_path('model/am_robot.xml')
data = mujoco.MjData(model)
solver = MPCSolver(casadi_dynamics, config)

set_initial_state(data)
reference = generate_trajectory()

while running:
    # 1. Read state from MuJoCo
    x = get_state_from_mujoco(data)       # convert MuJoCo quat [w,x,y,z] → [x,y,z,w]

    # 2. Solve MPC
    u_opt = solver.solve(x, reference)

    # 3. Apply control to MuJoCo
    data.ctrl[:] = u_opt

    # 4. Step MuJoCo
    mujoco.mj_step(model, data)

    # 5. Render (optional)
    viewer.sync()
```

Key details:
- **Quaternion convention mismatch**: MuJoCo uses [w,x,y,z], we use [x,y,z,w] → convert at the interface
- **Actuator mapping**: Map our `u = [F, τ, τ_joint]` to MuJoCo `data.ctrl`
- **Timing**: MuJoCo timestep can be smaller than MPC dt (e.g., MuJoCo at 1000 Hz, MPC at 50 Hz)

---

## Step 5: Validation & Testing

### 5.1 Dynamics match
- Run same initial conditions in our NumPy simulator and MuJoCo
- Compare trajectories — should match closely (small differences from integrator)

### 5.2 Static hover
- Set reference = hover at [0, 0, 1], arm hanging down
- MPC should find u ≈ [0, 0, mg, 0, 0, 0, 0, 0] and hold position

### 5.3 Position step
- Step reference from [0,0,1] to [1,0,1]
- Platform should move smoothly, arm should stay stable

### 5.4 Arm trajectory
- Move end-effector to a target while keeping platform stable
- Requires coupling compensation — this is why we need full dynamics in MPC

### 5.5 Disturbance rejection
- Apply external force in MuJoCo mid-flight
- MPC should recover

---

## Execution Order

| # | Task | Depends on | Est. complexity |
|---|------|------------|-----------------|
| 1 | MuJoCo MJCF model | model.py | Medium |
| 2 | CasADi dynamics | ams/*.py | Medium-High |
| 3 | Validate CasADi vs NumPy | 2 | Low |
| 4 | MPC solver (start with CasADi Opti) | 2 | Medium |
| 5 | MuJoCo simulation loop | 1 | Low |
| 6 | Connect MPC ↔ MuJoCo | 4, 5 | Low |
| 7 | Hover test | 6 | Low |
| 8 | Tune Q, R weights | 7 | Iterative |
| 9 | Trajectory tracking tests | 8 | Medium |
| 10 | (Optional) Switch to acados for speed | 4 | Medium |

Start with steps 1 + 2 in parallel, then integrate.

---

## Dependencies

```
pip install mujoco casadi numpy
# Optional for acados:
pip install acados_template
```
