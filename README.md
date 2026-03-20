# AM_dynamics
Dynamics of an aerial manipulator: a quadrotor with a 2-DOF planar arm mounted underneath.

## System Overview

```
          ┌─────────────┐
          │  Quadrotor   │   {A} platform frame
          │   Platform   │   (x_A = forward, y_A = left, z_A = up)
          └──────┬───────┘
                 │  mount offset [0, 0, -0.05] m
                 │
          Frame {0} ── Fixed arm base (does NOT rotate)
          Frame {1} ── Joint 1 body frame (same origin, rotated by θ₁ about z)
                 │
           Link 1 │ 0.25 m along x_1
                 │  COM at +0.125 along x_1
                 │
          Frame {2} ── Joint 2 body frame (rotated by θ₂ about z)
                 │
           Link 2 │ 0.20 m along x_2
                 │  COM at +0.10 along x_2
                 │
          Frame {3} ── End-effector
```

## Coordinate Frames

### World Frame {W}
- **Origin**: Fixed in inertial space
- **Axes**: x = East, y = North, z = Up (ENU convention)
- All dynamics equations are expressed in this frame (superscript `⁰`)

### Platform Frame {A}
- **Origin**: Center of mass of the quadrotor
- **Axes**: x_A = forward, y_A = left, z_A = up (FLU body convention)
- **Orientation**: Represented by unit quaternion q_A = [q_x, q_y, q_z, q_w]^T
- **Relation to {W}**: R_A = rotation matrix from quaternion, p_A = platform position in world

### Frame {0} (Fixed arm base)
- **Origin**: Bottom center of platform, 5 cm below platform COM (mount point)
- **Axes** (in platform frame coordinates):
  - x_0 = [0, 0, -1]_A (pointing down)
  - y_0 = [1, 0, 0]_A  (pointing forward, same as x_A)
  - z_0 = [0, -1, 0]_A (pointing right — joint rotation axis)
- **Relation to {A}**: Fixed transform (does NOT rotate with the arm)
  - p_mount = [0, 0, -0.05]^T in {A}
  - R_mount maps {0} axes to {A} axes
- **Why this orientation**: z_0 is horizontal → joints rotate in a vertical plane. x_0 = down → θ = 0 means arm hangs straight down.

### Frame {1} (Link 1 body frame)
- **Origin**: Same as frame {0} (coincident origins)
- **Axes**: Rotated from {0} by θ₁ about z_0
  - x_1 along link 1 (at θ₁=0: same as x_0, pointing down)
  - z_1 = z_0 (joint axis preserved, α = 0)
- **Body frame**: Rotates with link 1
- **COM of link 1**: [+0.125, 0, 0] in frame {1} (midpoint of link 1, positive along x_1)

### Frame {2} (Link 2 body frame)
- **Origin**: End of link 1, 0.25 m along x_1 from frame {1}
- **Axes**: Rotated from {1} by θ₂ about z_1
  - x_2 along link 2 (at θ₂=0: same as x_1)
  - z_2 = z_1 (joint axis preserved, α = 0)
- **Body frame**: Rotates with link 2
- **COM of link 2**: [+0.10, 0, 0] in frame {2} (midpoint of link 2, positive along x_2)

### Frame {3} (End-effector)
- **Origin**: End of link 2, 0.20 m along x_2 from frame {2}
- **Axes**: Same as frame {2} (no joint, fixed extension)

### DH Parameters (Modified DH / Craig convention)

Transforms are from frame {i} to {i+1}:

| Transform | α | a (m) | d (m) | θ | Notes |
|-----------|---|-------|-------|---|-------|
| {0}→{1} | 0 | 0 | 0 | θ₁ | Pure rotation, origins coincide |
| {1}→{2} | 0 | 0.25 | 0 | θ₂ | Link 1 geometry (a = L₁) |
| {2}→{3} | 0 | 0.20 | 0 | 0 | Link 2 geometry (a = L₂), no joint |

All α = 0 → all joint axes are parallel → planar arm motion.

COM offsets (in body frame {i}):
- Link 1: [0.125, 0, 0] in {1}
- Link 2: [0.100, 0, 0] in {2}

## Frame Transform Chain

World → Platform → Frame {0} → Frame {1} → Frame {2} → Frame {3}

```
⁰T₃ = ⁰T_A · T_mount · ⁰T₁(θ₁) · ¹T₂(θ₂) · ²T₃
```

Each DH transform (since α = d = 0):

```
         ┌ cosθ  -sinθ  0  a ┐
ⁱTᵢ₊₁ = │ sinθ   cosθ  0  0 │
         │  0      0     1  0 │
         └  0      0     0  1 ┘
```

Note: {0}→{1} has a = 0 (pure rotation), {2}→{3} has θ = 0 (pure translation).

## Zero Configuration (all θ = 0)

With identity platform orientation and platform at p_A = [0, 0, h]:

| Point | Frame | World Position |
|-------|-------|---------------|
| Platform COM | {A} | [0, 0, h] |
| Arm base | {0} | [0, 0, h − 0.05] |
| Link 1 body frame | {1} | [0, 0, h − 0.05] (same as {0}) |
| Link 1 COM | — | [0, 0, h − 0.175] |
| Link 2 body frame (joint 2) | {2} | [0, 0, h − 0.30] |
| Link 2 COM | — | [0, 0, h − 0.40] |
| End-effector | {3} | [0, 0, h − 0.50] |

The arm hangs straight down along −z_W.

## Sign Conventions

- **+θ₁**: Swings link 1 from vertical (down) toward forward (like raising an excavator boom)
- **+θ₂**: Swings link 2 relative to link 1 in the same rotation sense
- **Quaternion**: Hamilton convention, [x, y, z, w] storage order
- **Angular velocity**: World-frame ⁰ω_A

## Code Structure

```
ams/
├── state.py       — State dataclasses (platform + manipulator), vector conversion
├── math_utils.py  — Pure math: quaternion ops, skew matrix, cross product
├── model.py       — Physical parameters, DH table, link properties
├── kinematics.py  — Forward kinematics, velocity/acceleration recursion
├── dynamics.py    — Newton-Euler forward/backward pass
└── simulator.py   — x_dot = f(x, u) integration wrapper
```
