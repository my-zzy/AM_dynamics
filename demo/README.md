# AM Demo — Model Reference & Zero-Configuration Fix

This document replaces the outdated top-level README.md.  All geometry is
derived from `basic/model/am_robot.xml` and cross-checked with MuJoCo.

---

## 1. XML Model Geometry

### Body tree (positions are in the *parent* body frame)

```
base  (free joint)
└── link1   pos="0 0 -0.05"    joint1: hinge axis="0 1 0"  (local y)
    └── link2   pos="0 0 -0.12"   joint2: hinge axis="0 1 0"  (local y)
        └── ee      pos="0.16 0 0"   (no joint — rigid)
            ├── finger_left   pos="0.02  0.026 0"  (gripper_left_joint, slide y)
            └── finger_right  pos="0.02 -0.026 0"  (gripper_right_joint, slide y)
```

### Geom extents (visual + collision, in each body's own frame)

| Body   | Geom type | from → to (local frame)   | Length   |
|--------|-----------|---------------------------|----------|
| link1  | capsule   | (0,0,0) → (0,0,−0.12)    | 0.12 m   |
| link2  | capsule   | (0,0,0) → (0.16,0,0)     | 0.16 m   |
| ee     | box palm  | —                         | —        |
| —      | site `end_effector` | (0.078, 0, 0) in ee | — |

Total arm reach from joint 2 to fingertip site: **0.16 + 0.078 = 0.238 m**.

### Inertial properties (from XML + parallel-axis theorem)

| Body                        | Mass (kg) | Diagonal inertia [Ixx, Iyy, Izz] (kg·m²) |
|-----------------------------|-----------|-------------------------------------------|
| base (quadrotor platform)   | 1.500     | [0.00800, 0.00800, 0.01500]               |
| link1                       | 0.150     | [0.00020, 0.00020, 0.00005]               |
| link2 + ee + fingers (lump) | 0.220     | [0.000254, 0.000821, 0.000908] †          |

† Computed by `ams/inertia_check.py` via parallel-axis theorem (bodies: link2, ee,
finger_left, finger_right), rotated into DH frame {2}.  Run
`python ams/inertia_check.py` to reproduce.

---

## 2. XML Zero Configuration

At `joint1 = joint2 = 0`, level platform at world position `(0, 0, h)`:

| Point                      | World position          |
|----------------------------|-------------------------|
| Platform COM               | (0, 0, h)               |
| Joint 1 / link1 origin     | (0, 0, h − 0.05)        |
| Joint 2 / link2 origin     | (0, 0, h − 0.17)        |
| EE body origin             | (0.16, 0, h − 0.17)     |
| `end_effector` site        | (0.238, 0, h − 0.17)    |

**Shape: L — link1 hangs straight down, link2 extends horizontally forward.**

```
       base  ──────── (h)
         │  0.05 m
       joint1         (h−0.05)
         │  0.12 m  (link1, −z direction)
       joint2─────────────────── EE site
               0.238 m  (+x direction, link2 + gripper)
                         (h−0.17)
```

### XML joint sign convention (right-hand rule about local y-axis = [0,1,0])

Using **R_y(θ)** applied to the link zero-angle direction:

- **+θ₁**: link1 direction `[0,0,−1]` → `[−sinθ₁, 0, −cosθ₁]`.
  Small positive θ₁ swings link1 **backward** (−x world).
- **+θ₂**: link2 direction `[1,0,0]` → `[cosθ₂, 0, −sinθ₂]`.
  Small positive θ₂ dips link2 **downward** (−z world).

---

## 3. Zero-Configuration Inconsistency (old model.py vs XML)

The original `ams/model.py` defined θ = 0 as **arm straight down** (both links
collinear, pointing −z).  The XML's θ = 0 is the **L-shape** above.  Two
independent errors cause the mismatch:

| Issue | Old model.py | XML |
|-------|-------------|-----|
| Joint rotation axis (z₀) | `[0,−1,0]_A` (right) | `[0,+1,0]_A` (left) — sign flip |
| θ₂ = 0 geometry | link2 points **down** (collinear) | link2 points **forward** (−π/2 offset) |

Because of the sign flip, old model positive θ means **forward** while XML positive θ means **backward**.

---

## 4. Fix Options

### Option 1 — Add `ref` to joint2 in XML *(XML only)*

```xml
<joint name="joint2" type="hinge" axis="0 1 0" damping="0.02" ref="-1.5708"/>
```

MuJoCo reports `qpos[joint2] = 0` when actual angle = `ref = −π/2`, i.e. arm
straight down.  No Python code changes.

**Pros:** minimal change.  
**Cons:** visual default pose changes; sign of θ₁ still opposite to old model;
does not fix the z₀ sign flip.

---

### Option 2 — Software angle offset in state read/write *(no XML change)*

Apply a conversion layer wherever raw qpos is read or written:

```python
# reading from MuJoCo:
theta_model[0] = -theta_xml[0]          # sign flip (z₀ reversal)
theta_model[1] = -theta_xml[1] - pi/2   # sign flip + 90° offset
```

**Pros:** XML and model.py definitions stay untouched.  
**Cons:** hidden conversion layer, easy to forget; must be applied consistently.

---

### Option 3 — Pre-rotate link2 body in XML *(XML + geom update)*

Add an `euler` rotation to the `link2` body definition so that at joint2 = 0
the link already points down:

```xml
<body name="link2" pos="0 0 -0.12" euler="0 90 0">
```

All child geom positions must be re-expressed in the new body frame.

**Pros:** geometrically explicit.  
**Cons:** requires careful re-checking of all geom and site positions; visual
default pose changes.

---

### Option 4 — Update model.py DH to match XML *(Python only, **selected**)*

Redefine the mount rotation and add a θ₂ offset in `model.py` so that the FK
accepts raw MuJoCo qpos values and returns correct world positions.

No XML, no MuJoCo changes.  See full derivation in the next section.

**Pros:** single source of truth (XML); no hidden conversions.  
**Cons:** model.py's intuitive "down = zero" convention is replaced by the XML
L-shape convention.

---

## 5. Option 4 Derivation

### 5.1 Root cause decomposition

**Issue A — z₀ sign.**  
The DH joint rotation axis is z of frame {0}.  In `model.py` (old):

```
z₀_old = [0, −1, 0]_A   →  world [0, −1, 0]  (right at level hover)
XML joint1 axis          →  world [0, +1, 0]  (left)
```

They are antiparallel → positive θ has opposite sense.  Fix: set `z₀ = [0,+1,0]_A`.

**Issue B — θ₂ offset.**  
In DH (Craig), at θ₂ = 0 the x₂ axis equals x₁ (link1 direction = down at θ₁=0).
But the XML link2 is **horizontal** (forward) at joint2 = 0.  The angle between
"down" and "forward" is −π/2 (rotation about +y from down to forward by −90°).
Fix: use DH angle `θ₂_DH = θ₂_XML − π/2`.

### 5.2 New mount_rotation

Frame {0} axes expressed in platform frame {A} (FLU: x=forward, y=left, z=up):

| Axis | Direction in {A} | World (level hover) | Reason |
|------|-----------------|---------------------|--------|
| x₀   | `[0, 0, −1]`    | down                | link1 direction at θ₁=0 |
| y₀   | `[−1, 0, 0]`    | backward            | completes right-hand frame |
| z₀   | `[0, +1, 0]`    | left                | joint rotation axis (= XML) |

Verification: `det([x₀|y₀|z₀]) = det([[0,−1,0],[0,0,1],[−1,0,0]]) = +1` ✓

```python
# ams/model.py  mount_rotation  (NEW for Option 4)
mount_rotation = np.array([
    [ 0.0, -1.0,  0.0],   # row: A_x component of each {0} axis
    [ 0.0,  0.0,  1.0],   # row: A_y component
    [-1.0,  0.0,  0.0],   # row: A_z component
])
# Columns: x₀=[0,0,−1]_A,  y₀=[−1,0,0]_A,  z₀=[0,+1,0]_A
```

### 5.3 θ₂ offset derivation

We need x₂ = +x_W = `[1,0,0]` at θ₁=0, θ₂_XML=0.

In DH: x₂ = R_mount · Rz(θ₁=0) · Rz(θ₂_DH) · [1,0,0]
           = R_mount · [cosθ₂_DH, sinθ₂_DH, 0]ᵀ

Setting this equal to `[1,0,0]`:

```
[cosθ₂_DH, sinθ₂_DH, 0]ᵀ = R_mountᵀ [1,0,0]ᵀ = [0, −1, 0]ᵀ
→  cosθ₂_DH = 0,  sinθ₂_DH = −1
→  θ₂_DH = −π/2
```

Relationship: **θ₂_DH = θ₂_XML − π/2**

In code, the `dh_transform` call for link 2 uses `theta[1] - pi/2` instead
of `theta[1]`.

### 5.4 Verification

At level hover platform at origin, θ_XML = [0, 0]:

```
p[0] = (0, 0, −0.05)                          mount
p[1] = p[0] + R_mount·[0,0,0] = (0,0,−0.05)  joint1 (coincident)
p[2] = p[1] + R_mount·[0.12,0,0]
     = (0,0,−0.05) + 0.12·[0,0,−1]
     = (0, 0, −0.17)  ✓  joint2

R₂ = R_mount · Rz(−π/2) = [[1,0,0],[0,0,−1],[0,1,0]]  →  x₂=[1,0,0] ✓

p[3] = p[2] + R₂·[0.16,0,0]
     = (0,0,−0.17) + 0.16·[1,0,0]
     = (0.16, 0, −0.17)  ✓  EE body

EE site = p[2] + R₂·[0.238,0,0] = (0.238, 0, −0.17)  ✓  (matches MuJoCo)
```

---

## 6. Updated Coordinate Frames (Option 4)

### World Frame {W}
- z up, x East (forward at zero yaw), y North (left)

### Platform Frame {A}
- FLU body: x=forward, y=left, z=up
- Quaternion stored as `[qx, qy, qz, qw]`

### Frame {0} — Fixed arm base

| Property | Value |
|----------|-------|
| Origin   | p_A + R_A · [0, 0, −0.05] (5 cm below platform COM) |
| x₀       | R_A · [0, 0, −1] (down at level hover) |
| y₀       | R_A · [−1, 0, 0] (backward at level hover) |
| z₀       | R_A · [0, +1, 0] (left = joint rotation axis) |

### Frame {1} — Link 1 body frame

- **Origin**: same as {0}
- **Axes**: {0} rotated by θ₁ about z₀
- At θ₁=0: identical to {0} (x₁ = down, z₁ = left)
- **Link 1 COM**: `+0.06 m` along x₁ from frame {1} origin

### Frame {2} — Link 2 body frame

- **Origin**: 0.12 m along x₁ from {1} (end of link 1)
- **Axes**: {1} rotated by (θ₂ − π/2) about z₁
- At θ₁=0, θ₂=0 (XML): x₂ = [1,0,0]_W (forward), z₂ = [0,1,0]_W (left)
- **Link 2 + EE lump COM**: `+0.127 m` along x₂ from frame {2} origin

### Frame {3} — End-effector

- **Origin**: 0.16 m along x₂ from {2} (end of link 2 body)
- Same orientation as {2} (no joint)
- **Fingertip site**: additional 0.078 m along x₂

---

## 7. Updated DH Parameters (Option 4, Craig convention)

Transform from frame {i} to {i+1}: Rz(θ) · Tz(d) · Tx(a) · Rx(α)

| Transform  | α   | a (m) | d (m) | θ                 | Notes                             |
|------------|-----|-------|-------|-------------------|-----------------------------------|
| {0} → {1} | 0   | 0     | 0     | θ₁                | Pure rotation; origins coincide   |
| {1} → {2} | 0   | 0.12  | 0     | **θ₂ − π/2**     | −π/2 offset encodes L-shape       |
| {2} → {3} | 0   | 0.16  | 0     | 0                 | Pure translation (no joint)       |

All α = 0 → all joint axes parallel (z₀ ∥ z₁ ∥ z₂) → planar arm in the xz world plane.

COM offsets in body frame {i}:
- Link 1: `[+0.06, 0, 0]` in {1}
- Link 2 + EE: `[+0.127, 0, 0]` in {2}

---

## 8. Updated Zero Configuration (option 4, θ_XML = [0, 0])

With identity platform orientation and platform at p_A = (0, 0, h):

| Point                  | Frame       | World position           |
|------------------------|-------------|--------------------------|
| Platform COM           | {A}         | (0, 0, h)                |
| Arm mount / Joint 1    | {0} = {1}   | (0, 0, h − 0.05)         |
| Link 1 COM             | —           | (0, 0, h − 0.11)         |
| Joint 2 / link2 origin | {2}         | (0, 0, h − 0.17)         |
| Link 2 + EE lump COM   | —           | (0.127, 0, h − 0.17)     |
| EE body origin         | {3}         | (0.16, 0, h − 0.17)      |
| Fingertip site         | —           | (0.238, 0, h − 0.17)     |

Shape: link1 vertical along −z, link2 horizontal along +x → **L-shape**.

---

## 9. Sign Conventions (Option 4)

| Symbol | Positive direction | Physical effect |
|--------|--------------------|-----------------|
| +θ₁    | right-hand about z₀ = +y_W | link1 swings **backward** (−x world) and upward |
| +θ₂    | right-hand about z₁ = z₀  | link2 dips **downward** (−z world) from horizontal |

These match the raw MuJoCo qpos sign conventions directly.

---

## 10. Required model.py Changes (Option 4 implementation)

1. **`mount_rotation`** — replace with:
   ```python
   mount_rotation = np.array([
       [ 0.0, -1.0,  0.0],
       [ 0.0,  0.0,  1.0],
       [-1.0,  0.0,  0.0],
   ])
   ```

2. **`compute_link_transforms`** — apply θ₂ offset when calling `dh_transform`
   for link 2:
   ```python
   # Replace the loop body for i=1 (link2):
   R, p = self.links[0].dh_transform(theta[1] - np.pi / 2)
   ```
   (Or add a `theta_offset` field to `LinkParams`.)

3. **All call sites** — pass raw MuJoCo `qpos[7:9]` directly as `theta`;
   no manual remapping needed.

---

## 11. Code Structure

```
ams/
├── model.py        — physical parameters, DH table, mount_rotation
├── kinematics.py   — FK / velocity / acceleration recursion
├── dynamics.py     — Newton-Euler forward/backward pass
├── casadi_dynamics.py — CasADi symbolic dynamics (for MPC)
├── simulator.py    — ẋ = f(x, u) integration wrapper
└── inertia_check.py — MuJoCo vs model.py mass/inertia comparison

demo/
├── mpc_controller.py   — Acados nonlinear MPC (SQP)
├── mpc_trajectory.py   — minimum-jerk EE trajectory
├── mpc_reach_test.py   — hover → reach → hold test
└── mpc_single_step_debug.py — one-shot MPC diagnostic
```
