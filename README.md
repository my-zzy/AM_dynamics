# Aerial Manipulator Dynamics

A research codebase for modelling, simulation, and nonlinear MPC control of a
quadrotor aerial manipulator (AM): a quadrotor platform carrying a 2-DOF planar
robot arm with a parallel-jaw gripper.

The dynamics are expressed analytically via a Newton-Euler recursive algorithm
(`ams/`) and also in CasADi symbolic form for use inside an acados NMPC
(`demo/`).  MuJoCo (`basic/model/am_robot.xml`) serves as the ground-truth
simulator and physics validator throughout.

---

## Contents

- [Repository layout](#repository-layout)
- [Environment setup](#environment-setup)
- [Useful scripts and usage](#useful-scripts-and-usage)
  - [Validation](#validation)
  - [Basic experiments](#basic-experiments)
  - [MPC demo](#mpc-demo)
- [Model reference](#model-reference)
  - [XML geometry](#xml-geometry)
  - [Zero configuration](#zero-configuration)
  - [DH parameters](#dh-parameters-option-4-craig-convention)
  - [Gravity compensation](#gravity-compensation-at-zero-config)
- [State and input layout](#state-and-input-layout)

---

## Repository layout

```
ams/                        Core analytical model (pure Python + CasADi)
├── model.py                Physical parameters, DH table, mount_rotation
├── kinematics.py           FK / velocity / acceleration recursion
├── dynamics.py             Newton-Euler forward/backward pass
├── casadi_dynamics.py      CasADi symbolic dynamics (used by acados MPC)
├── simulator.py            ẋ = f(x, u) + RK4 integrator
├── state.py                State vector helper (pack/unpack)
├── math_utils.py           Quaternion utilities, skew-symmetric, etc.
└── inertia_check.py        Validation: model.py vs MuJoCo XML mass/inertia

basic/                      MuJoCo experiments (PID baseline, model inspection)
├── model/
│   ├── am_robot.xml        Full aerial manipulator MuJoCo model (ground truth)
│   └── quad_only.xml       Quadrotor body only (no arm)
├── test_model.py           Load and inspect the MuJoCo model
├── arm_test.py             Joint tracking test (fixed base)
├── arm_effect.py           Arm-swing effect on free-floating drone
├── pid_controller.py       PID controller implementation
├── pid_demo.py             Real-time PID position demo with waypoints
├── pid_tuning.py           PID tuning (full AM model, p2p / figure-8)
└── pid_tuning_quad.py      PID tuning (quad-only model, no arm)

demo/                       Nonlinear MPC (acados)
├── mpc_controller.py       Acados OCP setup + SQP solver (MPCController class)
├── mpc_trajectory.py       Minimum-jerk EE trajectory generator
├── mpc_reach_test.py       Main MPC reach test: hover → reach → hold
├── mpc_single_step_debug.py  One-shot MPC diagnostic + plots
├── mpc_grasp_task.py       Full pick-and-place with NMPC
├── grasp_task.py           Pick-and-place with decoupled PID + arm-PD
└── grasp_scene.xml         MuJoCo scene with target object

adrc/                       ADRC controller notes (design documents)
README.md                   This file
```

---


## Useful scripts and usage

All commands are run from the **workspace root** (`AM_dynamics/`).


### MPC demo

> Requires `conda activate gz` and acados installed with `ACADOS_SOURCE_DIR` set.

#### `demo/mpc_single_step_debug.py`  ← **start here**
Runs **one** MPC solver call from the nominal hover state and produces a 9-panel
diagnostic plot (`demo/mpc_single_step_debug.png`): predicted EE trajectory,
per-stage cost breakdown, joint angles, drone position, quaternion norm.

```bash
python demo/mpc_single_step_debug.py           # reuse existing compiled code
python demo/mpc_single_step_debug.py --rebuild # force recompile acados C code
```

Solver status 0 = OK.  Use this script to verify any model/weight change before
running a full test.

---

#### `demo/mpc_reach_test.py`
Full MPC reach test: drone hovers at `HOVER_START`, MPC drives the EE along a
minimum-jerk trajectory to `EE_TARGET`, then holds at the target.

```bash
python demo/mpc_reach_test.py
python demo/mpc_reach_test.py --rebuild          # recompile acados
python demo/mpc_reach_test.py --no-viewer        # headless
python demo/mpc_reach_test.py --no-tc            # disable hard terminal constraint
python demo/mpc_reach_test.py --dt-mpc=0.05 --horizon=20 --traj-dur=4.0
```

---

#### `demo/mpc_grasp_task.py`
Full pick-and-place using the NMPC controller. PID handles take-off; NMPC
takes over for EE reaching, hold at grasp point, and arm retraction.

```bash
python demo/mpc_grasp_task.py
```

---

### Basic experiments

#### `basic/arm_test.py`
Joint tracking test with the drone base **fixed** in space. Drives both joints
through step, sine, or ramp references via a PD controller and plots
position/velocity tracking.

```bash
python basic/arm_test.py          # default: sine mode
```

Edit `TEST_MODE = 'step' | 'sine' | 'ramp'` at the top of the file.

---

#### `basic/arm_effect.py`
Free-float demo: constant hover thrust applied as `xfrc_applied`, arm swings
sinusoidally. Shows how arm motion causes base drift.  Opens the MuJoCo viewer
and saves `basic/arm_effect.png` on exit.

```bash
python basic/arm_effect.py
```

---

#### `basic/pid_demo.py`
Real-time MuJoCo viewer demo. PID position controller flies the AM through a
waypoint sequence (take-off → forward → sideways → climb → return → land).

```bash
python basic/pid_demo.py
```

---

#### `basic/pid_tuning.py`
Interactive PID tuning loop for the full AM model. Runs a trajectory (p2p or
figure-8), opens the viewer, saves position/attitude plots when the viewer
closes. Edit `GAINS` at the top of the file to tune.

```bash
python basic/pid_tuning.py --mode p2p
python basic/pid_tuning.py --mode figure8
```

---

#### `basic/pid_tuning_quad.py`
Same as above but loads `quad_only.xml` — useful for tuning the drone gains
in isolation before adding the arm.

```bash
python basic/pid_tuning_quad.py --mode p2p
python basic/pid_tuning_quad.py --mode figure8
```

---

#### `demo/grasp_task.py`
Full pick-and-place demo using a **decoupled** PID (platform) + PD (arm)
controller. Eight phases: take-off → arm ready → approach → grasp → lift →
transport → place → retract.

```bash
python demo/grasp_task.py
```

---

### Validation

#### `ams/inertia_check.py`
Compares mass, inertia, and EE forward kinematics between `ams/model.py` and
the MuJoCo XML.  Run this whenever model parameters change.

```bash
python ams/inertia_check.py
```

Expected output: all checks print `OK`; FK EE position error < 1 mm.

---

#### `basic/test_model.py`
Loads `am_robot.xml`, prints joint names/indices, body tree, and geom extents.
Useful for verifying the MuJoCo model structure.

```bash
python basic/test_model.py
```

---

## Model reference

All geometry is derived from `basic/model/am_robot.xml`.

### XML geometry

**Body tree** (positions in parent body frame):

```
base  (free joint)
└── link1   pos="0 0 -0.05"   joint1: hinge axis="0 1 0"
    └── link2   pos="0 0 -0.12"  joint2: hinge axis="0 1 0"
        └── ee      pos="0.16 0 0"  (rigid)
            ├── finger_left   pos="0.02  0.026 0"  (slide joint, y)
            └── finger_right  pos="0.02 -0.026 0"  (slide joint, y)
```

| Body                        | Mass (kg) | Inertia diag [Ixx, Iyy, Izz] (kg·m²) |
|-----------------------------|-----------|---------------------------------------|
| base (quadrotor platform)   | 1.500     | [0.00800, 0.00800, 0.01500]           |
| link1                       | 0.150     | [0.00020, 0.00020, 0.00005]           |
| link2 + ee + fingers (lump) | 0.220     | [0.000254, 0.000821, 0.000908] †      |

† Computed via MuJoCo parallel-axis theorem in `ams/inertia_check.py`.

---

### Zero configuration

At `joint1 = joint2 = 0`, level platform at height `h`:

```
       platform  (0, 0, h)
           │  0.05 m
         joint1  (0, 0, h−0.05)
           │  0.12 m  (link1, −z)
         joint2 ──────────────── EE site  (0.238, 0, h−0.17)
               0.16 m (link2, +x)  +  0.078 m (gripper site)
```

**Shape: L — link1 hangs straight down, link2 extends horizontally forward.**

Sign convention (right-hand rule about local y = `[0,1,0]`):

| Joint | Positive direction | Effect |
|-------|--------------------|--------|
| +θ₁   | about +y           | link1 swings **backward** (−x world) |
| +θ₂   | about +y           | link2 dips **downward** (−z world) |

These match raw MuJoCo `qpos` directly — no remapping needed.

---

### DH parameters (Option 4, Craig convention)

| Transform  | α | a (m) | d (m) | θ             |
|------------|---|-------|-------|---------------|
| {0} → {1} | 0 | 0     | 0     | θ₁            |
| {1} → {2} | 0 | 0.12  | 0     | θ₂ − π/2      |
| {2} → {3} | 0 | 0.16  | 0     | 0 (EE, fixed) |

The `−π/2` offset on θ₂ encodes the L-shape zero configuration so that raw
MuJoCo `qpos` can be passed directly to `compute_link_transforms`.

Mount rotation (platform → arm base frame {0}, columns = {0} axes in {A}):

```python
mount_rotation = np.array([
    [ 0.0, -1.0,  0.0],   # x₀ = [0,0,−1]_A  (down)
    [ 0.0,  0.0,  1.0],   # y₀ = [−1,0,0]_A  (backward)
    [-1.0,  0.0,  0.0],   # z₀ = [0,+1,0]_A  (left = joint axis)
])
```

See [demo/README.md](demo/README.md) for the full derivation.

---

### Gravity compensation at zero config

To hold the L-shape hover statically, three inputs must be set to their
non-zero equilibrium values:

| Input       | Value         | Reason |
|-------------|---------------|--------|
| `F_ext[2]`  | `m_total · g ≈ 18.34 N` | Upward thrust |
| `tau_ext[1]`| `−m₂ · g · x_com ≈ −0.274 Nm` | Platform pitch (arm moment) |
| `tau_j[0]`  | `−0.274 Nm`   | Joint 1 hold |
| `tau_j[1]`  | `−0.274 Nm`   | Joint 2 hold |

Computed via inverse dynamics in `ams/dynamics.py`.

---

## State and input layout

**State** `x ∈ ℝ¹⁷`:

| Indices | Symbol     | Description                         |
|---------|-----------|-------------------------------------|
| 0:3     | `p_A`     | Platform position (world frame)      |
| 3:6     | `v_A`     | Platform linear velocity (world)     |
| 6:10    | `q_A`     | Platform quaternion `[x,y,z,w]`      |
| 10:13   | `ω_A`     | Platform angular velocity (body)     |
| 13:15   | `θ`       | Joint angles `[θ₁, θ₂]`             |
| 15:17   | `θ̇`      | Joint velocities                     |

**Input** `u ∈ ℝ⁸`:

| Indices | Symbol      | Description                          |
|---------|------------|--------------------------------------|
| 0:3     | `F_ext`    | External force on platform (world)   |
| 3:6     | `τ_ext`    | External torque on platform (body)   |
| 6:8     | `τ_j`      | Joint torques `[τ_j1, τ_j2]`        |

Note: `F_ext[0] = F_ext[1] = 0` (body-z thrust only); bounds enforced by MPC.

---
