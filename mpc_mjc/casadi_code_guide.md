# CasADi Dynamics Port — Code Guide (Option A: Reuse DH Code)

## Strategy

Reuse the verified Newton-Euler recursion from `ams/` by porting it to CasADi
symbolic expressions. Same math, same frame conventions — only the numerical
backend changes (`ca.SX` instead of `np.ndarray`) and the model parameters
update to match the MuJoCo robot.

A thin **interface layer** handles the convention mismatch between MuJoCo and DH.

---

## 1. What Is Reused As-Is (Same Logic)

These files contain the core Newton-Euler math. The **logic, structure, and
equations** are reused verbatim — only `numpy` calls become `casadi` calls.

| ams/ file | Functions to port | CasADi equivalent |
|-----------|-------------------|-------------------|
| `math_utils.py` | `quat_to_rotation_matrix(q)` | `ca.SX` 3×3, same formula |
| | `quat_derivative(q, omega)` | `ca.SX` Omega matrix, same formula |
| `kinematics.py` | `forward_kinematics(model, q_A, p_A, theta)` | Unrolled loop (2 links) |
| | `velocity_recursion(...)` | Same recursion |
| | `acceleration_recursion(...)` | Same recursion |
| `dynamics.py` | `backward_recursion(...)` | Same recursion |
| | `_eval_id(...)` | Same logic |
| | `forward_dynamics(...)` | Mass matrix via repeated ID |
| `simulator.py` | `state_derivative(model, x, u)` | Becomes `ca.Function('f', [x, u], [x_dot])` |
| `state.py` | State layout `[p, v, q, ω, θ, θ̇]` | Same indexing (17 states, 8 inputs) |

### NumPy → CasADi translation table

| NumPy | CasADi |
|-------|--------|
| `np.array([...])` | `ca.vertcat(...)` or `ca.SX(...)` |
| `A @ B` | `ca.mtimes(A, B)` |
| `np.cross(a, b)` | `ca.cross(a, b)` |
| `np.linalg.solve(M, b)` | `ca.solve(M, b)` |
| `np.cos(x)` | `ca.cos(x)` |
| `np.sin(x)` | `ca.sin(x)` |
| `np.zeros(3)` | `ca.SX.zeros(3)` |
| `np.eye(3)` | `ca.SX.eye(3)` |
| `np.diag([a,b,c])` | `ca.diag(ca.vertcat(a, b, c))` |
| `np.concatenate([a, b])` | `ca.vertcat(a, b)` |
| `v.copy()` | not needed (SX is symbolic) |
| `a[0:3]` | `a[0:3]` (same slicing) |

---

## 2. What Needs New Parameters

The MuJoCo robot has **different geometry** from the original `ams/model.py`.
The DH code is parameterized by the model, so we just need a new parameter set.

### 2.1 Geometry differences

| Property | ams/model.py | MuJoCo XML | Notes |
|----------|-------------|------------|-------|
| Link 1 length | 0.25 m | 0.12 m | DH `a₁ = 0.12` |
| Link 2 length | 0.20 m | 0.16 m | DH `a₂ = 0.16` |
| Link 1 COM | (0.125, 0, 0) in {1} | (0, 0, −0.06) in MuJoCo link1 | → (0.06, 0, 0) in DH {1} |
| Link 2 COM | (0.10, 0, 0) in {2} | (0.08, 0, 0) in MuJoCo link2 | → (0.08, 0, 0) in DH {2} |
| Link 1 inertia | diag(0.0001, 0.0008, 0.0008) | diag(0.0002, 0.0002, 0.00005) in MuJoCo | → diag(0.00005, 0.0002, 0.0002) in DH {1} |
| Link 2 inertia | diag(0.0001, 0.0004, 0.0004) | diag(0.00015, 0.00015, 0.00003) in MuJoCo | → diag(0.00003, 0.00015, 0.00015) in DH {2} |
| EE mass | 0 | 0.1 kg | Fold into link 2 |
| Zero config | Both links hang down | Link1 down, link2 forward (90° bend) | Handled by θ₂ offset |

### 2.2 End-effector mass

The MuJoCo `ee` body has mass 0.1 kg at 0.16 m from joint 2 (= link2 tip).
**Fold it into link 2** to avoid changing N_JOINTS:

```
m₂_combined = 0.12 + 0.10 = 0.22 kg
COM₂_combined = (0.12 × 0.08 + 0.10 × 0.16) / 0.22 = 0.1164 m  along x₂
```

For combined inertia, use the parallel axis theorem (shift each original inertia
to the new combined COM, then add). For the MPC, approximate values are fine —
we can tune later.

### 2.3 The 90° bend (θ₂ offset)

At zero joint angles, the MuJoCo arm has a structural 90° bend: link1 points
down, link2 points forward. In DH terms (all α=0, planar), both links point
along x at zero angle — i.e., both hang straight down.

**Solution**: a constant θ₂ offset. The DH code receives:
```
θ₂_DH = θ₂_converted + π/2
```
where the π/2 rotates the DH zero-config from "link2 pointing down"
to "link2 pointing forward," matching MuJoCo's geometry.

This offset is built into the interface layer (see §3), NOT into the DH code.

### 2.4 Unchanged parameters

| Property | Value | Same? |
|----------|-------|-------|
| Platform mass | 1.5 kg | ✓ |
| Platform inertia | diag(0.008, 0.008, 0.015) | ✓ |
| Mount offset | (0, 0, −0.05) in platform frame | ✓ |
| Mount rotation | [[0,1,0],[0,0,−1],[−1,0,0]] | ✓ |
| Gravity | (0, 0, −9.81) | ✓ |
| N_JOINTS | 2 | ✓ |

---

## 3. New Code: Interface Layer

The DH arm base frame {0} has z₀ = (0, −1, 0) in platform coords.
MuJoCo joints rotate about y_platform = (0, 1, 0). These are the **same physical
axis but with opposite sign**. Plus joint 2 has the π/2 structural offset.

### 3.1 State conversion: MuJoCo → DH

```python
def mjc_to_dh_joints(theta_mjc, theta_dot_mjc):
    """Convert MuJoCo joint states to DH convention."""
    theta_dh = np.array([
        -theta_mjc[0],              # sign flip (z₀ = -y_plat)
        -theta_mjc[1] + np.pi/2,   # sign flip + 90° structural offset
    ])
    theta_dot_dh = np.array([
        -theta_dot_mjc[0],
        -theta_dot_mjc[1],          # offset drops out in derivative
    ])
    return theta_dh, theta_dot_dh
```

Platform states (position, velocity, quaternion, angular velocity) need **no
conversion** — both use world frame, and our quaternion order [x,y,z,w] is
already handled in `test_model.py`'s `get_state()`.

### 3.2 Torque conversion: DH → MuJoCo

```python
def dh_to_mjc_torques(tau_dh):
    """Convert DH joint torques to MuJoCo convention."""
    return np.array([
        -tau_dh[0],   # same sign flip
        -tau_dh[1],
    ])
```

### 3.3 Platform force/torque

The DH code computes `F_ext` and `tau_ext` in **world frame** — exactly what
`apply_platform_wrench()` expects. No conversion needed.

However, the current MPC plan uses thrust + body torques `(T, τ_body)` as the
platform input. Two options:
- **Option 1**: Use full 6-DOF wrench `[F, τ]` in world frame (8 total inputs =
  3 force + 3 torque + 2 joints). Matches `inverse_dynamics()` output directly.
- **Option 2**: Use `(T, τ_body)` (6 total = 1 thrust + 3 body torques + 2 joints).
  Need `T * R_A @ [0,0,1] → F_world` and `R_A @ τ_body → τ_world` in the
  dynamics (adds one rotation).

**Recommendation**: Start with Option 1 (world-frame wrench, 8 inputs) for the
CasADi dynamics — it matches the existing code perfectly. Convert to
thrust+body-torques in the MPC cost/constraints if desired.

---

## 4. CasADi Porting Approach

### 4.1 File structure

```
mpc_mjc/
├── casadi_dynamics.py     ← NEW: all CasADi symbolic dynamics
├── casadi_model.py        ← NEW: MuJoCo-matched parameters (DH convention)
├── mjc_interface.py       ← NEW: sign flip + offset conversions
└── validate_dynamics.py   ← NEW: compare CasADi vs NumPy outputs
```

### 4.2 casadi_model.py — Parameter set

A dataclass or dict holding the MuJoCo-matched DH parameters:

```python
# All values in DH frame convention
PLATFORM_MASS     = 1.5
PLATFORM_INERTIA  = diag(0.008, 0.008, 0.015)
MOUNT_OFFSET      = [0, 0, -0.05]
MOUNT_ROTATION    = [[0,1,0],[0,0,-1],[-1,0,0]]
GRAVITY           = [0, 0, -9.81]

# Link 1 (DH frame {1})
LINK1_MASS        = 0.15
LINK1_INERTIA     = diag(0.00005, 0.0002, 0.0002)
LINK1_COM         = [0.06, 0, 0]
LINK1_ALPHA       = 0.0
LINK1_A           = 0.12
LINK1_D           = 0.0

# Link 2 (DH frame {2}) — includes folded ee mass
LINK2_MASS        = 0.22          # 0.12 + 0.10
LINK2_INERTIA     = diag(...)     # computed via parallel axis theorem
LINK2_COM         = [0.1164, 0, 0]  # combined COM
LINK2_ALPHA       = 0.0
LINK2_A           = 0.16
LINK2_D           = 0.0
```

### 4.3 casadi_dynamics.py — Porting order

Port in this order (each builds on the previous):

```
1.  ca_quat_to_rotation_matrix(q)         # math_utils → ca.SX
2.  ca_quat_derivative(q, omega)          # math_utils → ca.SX
3.  ca_dh_transform(alpha, a, d, theta)   # model → ca.SX
4.  ca_forward_kinematics(q_A, p_A, theta_dh)
5.  ca_velocity_recursion(...)
6.  ca_acceleration_recursion(...)
7.  ca_backward_recursion(...)
8.  ca_eval_id(...)                        # single ID evaluation
9.  ca_forward_dynamics(x_dh, u)          # mass matrix via 9 ID calls
10. ca_state_derivative(x_dh, u)          # full x_dot = f(x, u)
```

Important CasADi notes:
- Loops over 2 links are fine — CasADi unrolls them into the expression graph
- No `if/else` on symbolic values — but we don't need any (fixed 2-link structure)
- `ca.solve(M, b)` for the 8×8 mass matrix solve
- The final output is: `f = ca.Function('f_dynamics', [x, u], [x_dot])`

### 4.4 State vector convention in CasADi

The CasADi dynamics operates in **DH convention**:
```
x_dh = [p(3), v(3), q_xyzw(4), ω(3), θ_dh(2), θ̇_dh(2)]   → 17 dims
u    = [F_world(3), τ_world(3), τ_joint_dh(2)]              → 8 dims
```

The interface layer converts MuJoCo states to `x_dh` and CasADi torques back
to MuJoCo torques before each MPC solve / control application.

---

## 5. Validation (Step 3 from mpc_mjc.md)

### 5.1 CasADi vs NumPy (must match to ~1e-10)

```python
# validate_dynamics.py
import numpy as np
import casadi as ca

# 1. Pick random state and input
# 2. Convert to DH convention
# 3. Evaluate NumPy state_derivative (ams/simulator.py)
# 4. Evaluate CasADi state_derivative (casadi_dynamics.py)
# 5. Compare — should match to machine precision
```

Test at several states:
- Zero config, hover input (F = [0,0,mg], τ = 0, joints = 0)
- Random state + random input
- Arm at 45° / 90°

### 5.2 CasADi vs MuJoCo (should be close, not exact)

Even after validation against NumPy, the CasADi dynamics won't perfectly match
MuJoCo because:
- MuJoCo uses its own integrator (semi-implicit Euler by default)
- Damping model differs (`damping="0.02"` in XML vs. none in our code)
- Contact forces (landing legs)

This is expected and acceptable — the MPC handles model mismatch via feedback.

---

## 6. Conversion Cheat Sheet

```
MuJoCo → DH (state):
    θ₁_dh = -θ₁_mjc
    θ₂_dh = -θ₂_mjc + π/2
    θ̇₁_dh = -θ̇₁_mjc
    θ̇₂_dh = -θ̇₂_mjc
    (everything else: no change)

DH → MuJoCo (torques):
    τ₁_mjc = -τ₁_dh
    τ₂_mjc = -τ₂_dh
    F_world, τ_world: no change (both in world frame)

Frame axis mapping:
    DH x₀ = (0,0,-1)_plat = down
    DH y₀ = (1,0,0)_plat  = forward
    DH z₀ = (0,-1,0)_plat = right   ← joint axis
    MuJoCo joint axis = (0,1,0)_plat = left = -z₀
```
