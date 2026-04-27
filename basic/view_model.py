"""View the aerial manipulator XML model in the MuJoCo interactive viewer.

Run:
    conda activate mjc
    python basic/view_model.py

Viewer keyboard shortcuts (press inside the viewer window):
    J       — toggle joint axes display
    A       — toggle actuator display (shows actuator frames)
    F       — toggle contact forces
    C       — toggle contact points
    V       — toggle transparent rendering (see inertia ellipsoids)
    I       — toggle inertia ellipsoids
    T       — toggle tendons
    Space   — pause / resume simulation
    Arrows  — rotate camera
    Scroll  — zoom
    Right-click drag — pan
    Left-click drag  — rotate

To inspect a specific joint axis:
    1. Press J to show all joint axes as coloured arrows.
       Red = local x, Green = local y (rotation axis for hinge), Blue = local z.
    2. Double-click a body to select it and read its name in the status bar.
"""

import sys
import os
import numpy as np
import mujoco
import mujoco.viewer

_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _ROOT)

XML_PATH = os.path.join(os.path.dirname(__file__), 'model', 'am_robot.xml')


def print_joint_info(model):
    print("\n=== Joint info ===")
    for i in range(model.njnt):
        name  = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jtype = model.jnt_type[i]
        axis  = model.jnt_axis[i]          # axis in LOCAL body frame
        pos   = model.jnt_pos[i]           # joint origin in LOCAL body frame
        body  = model.jnt_bodyid[i]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        type_names = {0: 'free', 1: 'ball', 2: 'slide', 3: 'hinge'}
        print(f"  [{i}] {name:30s}  type={type_names.get(jtype,'?'):6s}"
              f"  body={bname:15s}"
              f"  axis={np.round(axis,4)}  pos={np.round(pos,4)}")

    print("\n=== Body world poses at zero config ===")
    mj_data = mujoco.MjData(model)
    mujoco.mj_resetData(model, mj_data)
    # Put drone at a visible height
    mj_data.qpos[2] = 1.5
    mujoco.mj_forward(model, mj_data)
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        pos  = mj_data.xpos[i]
        R    = mj_data.xmat[i].reshape(3, 3)
        print(f"  {name:20s}  pos={np.round(pos,4)}  x_axis={np.round(R[:,0],3)}")


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    print_joint_info(model)

    # Place drone at a comfortable hover height
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.5        # z = 1.5 m
    data.qpos[3] = 1.0        # quaternion w = 1 (level)
    mujoco.mj_forward(model, data)

    print("\nOpening viewer — press J to toggle joint axes.")
    with mujoco.viewer.launch_passive(model, data) as v:
        # Enable joint axis visualisation by default
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        v.sync()
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()


if __name__ == '__main__':
    main()
