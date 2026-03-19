# AM_dynamics
Dynamics of an aerial manipulator: a quadrotor with a 2-DOF planar arm mounted underneath.

## System Overview

```
          ┌─────────────┐
          │  Quadrotor  │   {A} platform frame
          │   Platform  │   (x_A = forward, y_A = left, z_A = up)
          └──────┬──────┘
                 │  mount offset [0, 0, -0.05] m
                 │
          ┌──────┴───────┐
          │  Arm Base {0}│   (x_0 = down, y_0 = forward, z_0 = joint axis)
          └──────┬───────┘
                 │  Joint 1 (θ₁, rotates about z_0)
                 │
           Link 1 │ 0.25 m along x_1
                 │
          ┌──────┴───────┐
          │  Frame {1}   │
          └──────┬───────┘
                 │  Joint 2 (θ₂, rotates about z_1)
                 │
           Link 2 │ 0.20 m along x_2
                 │
              [End-effector]
              Frame {2}
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

### Arm Base Frame {0}
- **Origin**: Bottom center of platform, 5 cm below platform COM
- **Axes** (in platform frame coordinates):
  - x_0 = [0, 0, -1]_A (pointing down)
  - y_0 = [1, 0, 0]_A  (pointing forward, same as x_A)
  - z_0 = [0, -1, 0]_A (pointing right)
- **Relation to {A}**: Fixed transform
  - p_mount = [0, 0, -0.05]^T in {A}
  - R_mount maps {0} axes to {A} axes
- **Why this orientation**: Joint axes (z_0) are horizontal. x_0 = down means θ = 0 → arm hangs straight down. +θ swings the arm forward.

### Joint Frames {1}, {2} (Craig DH convention)

| Frame | Origin | x-axis | z-axis |
|-------|--------|--------|--------|
| {1} | Joint 2 location | Along link 1 (at θ₁=0: same as x_0) | Parallel to z_0 (joint 2 axis) |
| {2} | End-effector | Along link 2 (at θ₂=0: same as x_1) | Parallel to z_1 |

DH parameters (Craig convention):

| Joint | α | a (m) | d (m) | θ |
|-------|---|-------|-------|---|
| 1 | 0 | 0.25 | 0 | θ₁ |
| 2 | 0 | 0.20 | 0 | θ₂ |

Both α = 0 → all joint axes are parallel → planar arm motion.

## Frame Transform Chain

World → Platform → Arm base → Link 1 → Link 2

```
⁰T₂ = ⁰T_A · T_mount · ⁰T₁(θ₁) · ¹T₂(θ₂)
```

Each DH transform (since α = d = 0):

```
         ┌ cosθ  -sinθ  0  a ┐
ⁱ⁻¹Tᵢ =  │ sinθ   cosθ  0  0 │
         │  0      0    1  0 │
         └  0      0    0  1 ┘
```

## Zero Configuration (all θ = 0)

With identity platform orientation and platform at p_A = [0, 0, h]:

| Point | World Position |
|-------|---------------|
| Platform COM | [0, 0, h] |
| Arm base (frame {0}) | [0, 0, h − 0.05] |
| Joint 2 (frame {1}) | [0, 0, h − 0.30] |
| End-effector (frame {2}) | [0, 0, h − 0.50] |

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
