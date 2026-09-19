# Grasp Task – Design Plan

## Scene recap

| Item | Value |
|---|---|
| Drone hover setpoint | `(bx, 0, 1.5)` m – adjustable via PID |
| Box position (world) | `(0.40, 0, 1.125)` m (centre) on pedestal |
| Box size (full) | 26 × 26 × 50 mm |
| Gripper open inner gap | 36 mm (5 mm clearance each side) |
| Gripper close travel | ±24 mm per finger → closes to ~4 mm gap |
| Arm joints | joint1, joint2 – both y-axis hinges (xz-plane) |
| Arm reach (l1, l2) | l1 = 0.12 m (link1), l2 ≈ 0.238 m (link2 + palm + EE site) |
| ctrl indices | 0=joint1, 1=joint2, 2=gripper_left, 3=gripper_right |

---

## Kinematics (2-D inverse kinematics)

Both joints rotate about the y-axis, so the arm moves entirely in the **robot-body xz-plane**.
With the drone level and at position `(bx, 0, bz)`, the EE world position is:

```
ex = bx − l1·sin(θ1) + l2·cos(θ1 + θ2)
ez = bz − 0.05 − l1·cos(θ1) − l2·sin(θ1 + θ2)
```

> Joint-zero pose is the **L-configuration**: link1 hangs straight down (−z), link2 points forward (+x).
> **Negative θ1** swings link1 tip toward +x (box side); **positive θ2** rotates link2 downward (−z in link1 frame).

To reach a desired `(ex, ez)` from drone base `(bx, bz)`, define relative target:

```
dx = ex − bx
dz = ez − bz + 0.05        # absorb link1 mount offset
```

Two-link sine-rule IK (note: `sin` not `cos` due to the L-config zero pose):

```
sin(θ2) = (dx² + dz² − l1² − l2²) / (2·l1·l2)
θ2 = arcsin(...)            # elbow-down ⇒ positive θ2
θ1 = atan2(−dz, dx) − atan2(l1 + l2·sin(θ2), l2·cos(θ2))
```

**Numerical example** (box at `(0.30, 0, 1.30)`, base at `(0.0, 0, 1.5)`):
- `dx=0.30`, `dz=1.30−1.5+0.05=−0.15`
- `sin(θ2)=(0.09+0.0225−0.0144−0.0566)/(2·0.12·0.238)=0.727`
- Solved: `θ1 ≈ −0.60 rad (−34°)`, `θ2 ≈ +0.81 rad (+47°)` *(elbow-down)*

---

## Task phases

```
Phase 0  TAKEOFF      t = 0  – 5 s    ramp from ground (z=0.22) to hover setpoint
Phase 1  INIT         t = 5  – 9 s    stabilise hover at (0, 0, 1.5)
Phase 2  ARM READY    t = 9  – 13 s   hold arm at initial angles, open gripper
Phase 3  APPROACH     t = 13 – 17 s   translate drone to (0.25, 0, 1.45) + live IK
Phase 4  GRASP        t = 17 – 19 s   close gripper
Phase 5  LIFT         t = 19 – 23 s   raise hover setpoint Δz = +0.20 m
Phase 6  TRANSPORT    t = 23 – 28 s   translate to drop-off location
Phase 7  PLACE        t = 28 – 30 s   descend, open gripper
Phase 8  RETRACT      t = 30 – 32 s   retract arm to rest pose
```

### Phase 0 – TAKEOFF
- Start on ground: drone base at `z = 0.22 m` (feet touching floor).
- Cosine-ramp hover setpoint from `(0, 0, 0.22)` → `(0, 0, 1.5)` over 4 s.
- Arm: `θ1=0°`, `θ2=0°` (L-config), gripper open.

### Phase 1 – INIT
- Hover PID setpoint: `(0.0, 0, 1.5)`.
- Arm: `θ1=0°`, `θ2=0°` (L-config), fingers open.
- Let drone stabilise.

### Phase 2 – ARM READY
- Keep drone at `(0.0, 0, 1.5)`.
- Hold arm at initial (L-config) angles: `θ1=0°`, `θ2=0°`.
- Gripper: fully open (`ctrl[2]=0`, `ctrl[3]=0`).
- Note: box at (0.40, 0, 1.125) is **unreachable** from this hover position (distance 0.515 m > 0.358 m max).

### Phase 3 – APPROACH
- Shift hover setpoint to `(0.25, 0, 1.45)` to get drone closer in x and z.
- Recompute IK live for current base position → arm tracks box target.
- Final EE target: `(0.40, 0, 1.125)` (box centre height, centred in y-gap).
- Required reach from APPROACH setpoint: `dx=0.15, dz=−0.275` → distance ≈ 0.313 m (within 0.358 m max).
- Verify `|ee_pos − box_pos|_y < 0.013` (box half-width) before continuing.

### Phase 4 – GRASP
- Apply closing force: `ctrl[2] = −20 N`, `ctrl[3] = +20 N`.
- Hold for ~2 s until fingers reach contact (joint velocity ≈ 0).
- Detect grip: check finger slide displacement `|q_finger| > 0.010 m`.

### Phase 5 – LIFT
- Hold gripper closed.
- Ramp hover setpoint z: `1.45 → 1.65 m` (lifts box 0.20 m off pedestal).
- Monitor box z-position to confirm it rises with the drone.

### Phase 6 – TRANSPORT
- Hold gripper closed.
- Smooth-ramp hover setpoint to drop-off location, e.g. `(−0.5, 0, 1.7)`.

### Phase 7 – PLACE
- Descend hover setpoint to drop height, e.g. `(−0.5, 0, 1.3)`.
- Open gripper: `ctrl[2]=0`, `ctrl[3]=0`.

### Phase 8 – RETRACT
- Smooth-ramp arm back to rest: `θ1=0°`, `θ2=0°`.
- Return hover setpoint to `(0, 0, 1.5)`.

---

## Control signals summary

| Phase | hover_pos | θ1 [°] | θ2 [°] | gripper_left | gripper_right |
|---|---|---|---|---|---|
| 0 TAKEOFF     | (0.0, 0, 0.22)→0 (0,0,1.5) |   0  |   0  |   0 |   0 |
| 1 INIT        | (0.0, 0, 1.5)    |   0  |   0  |   0 |   0 |
| 2 ARM READY   | (0.0, 0, 1.5)    |   0  |   0  |   0 |   0 |
| 3 APPROACH    | (0.25, 0, 1.45)  |  IK  |  IK  |   0 |   0 |
| 4 GRASP       | (0.25, 0, 1.45)  | hold | hold | −20 | +20 |
| 5 LIFT        | (0.25, 0, 1.65)  | hold | hold | −20 | +20 |
| 6 TRANSPORT   | (−0.5, 0, 1.7)  | hold | hold | −20 | +20 |
| 7 PLACE       | (−0.5, 0, 1.3)  | hold | hold |   0 |   0 |
| 8 RETRACT     | (0.0, 0, 1.5)    |   0  |   0  |   0 |   0 |

---

## Expected disturbances on base (from arm effect study)
- Arm motion induces **pitch** and **x-position** disturbance on the base.
- PID position controller compensates, but residual oscillation ~±0.03 m is expected.
- Gripper closing adds a small impulsive force on contact; PID damps it.
- Total arm+gripper mass ≈ 0.23 kg → CoM shift when fully extended ≈ 0.04 m forward.

---

## Implementation file
→ `demo/grasp_task.py` – load `grasp_scene.xml`, implement the phase state machine above.
