"""Compare mass / inertia between ams/model.py and the MuJoCo XML.

Extracts body mass and inertia from am_robot.xml via MuJoCo and
prints a side-by-side comparison with AerialManipulatorModel values,
highlighting discrepancies.

Run from workspace root:
    python ams/inertia_check.py
"""

import sys, os
import numpy as np

_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'basic'))

import mujoco
from ams.model import AerialManipulatorModel

XML_PATH = os.path.join(_ROOT, 'basic', 'model', 'am_robot.xml')

TOL_MASS   = 1e-4   # kg
TOL_INERTIA = 1e-5  # kg·m²

# ---------------------------------------------------------------------------
# MuJoCo extraction
# ---------------------------------------------------------------------------

def get_body_props(mj_model, name):
    """Return (mass, inertia_diag_principal) for a named body."""
    bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f'Body "{name}" not found in XML')
    mass = float(mj_model.body_mass[bid])
    # body_inertia stores [Ixx, Iyy, Izz] in the principal frame (already diagonal)
    inertia = mj_model.body_inertia[bid].copy()
    return mass, inertia

# ---------------------------------------------------------------------------
# Print helper
# ---------------------------------------------------------------------------

def compare(label, model_val, xml_val, tol, unit=''):
    model_val = np.atleast_1d(np.asarray(model_val, dtype=float))
    xml_val   = np.atleast_1d(np.asarray(xml_val,   dtype=float))
    err = np.abs(model_val - xml_val)
    ok  = np.all(err < tol)
    tag = '  OK' if ok else '  *** MISMATCH ***'
    print(f'  {label:30s}  model={np.round(model_val,6)}  xml={np.round(xml_val,6)}{tag}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    am = AerialManipulatorModel()

    print('=' * 70)
    print('Mass / Inertia comparison: model.py  vs  am_robot.xml')
    print('=' * 70)

    # ── Platform (base body) ─────────────────────────────────────────────────
    print('\n[Platform / base]')
    m_xml, I_xml = get_body_props(mj_model, 'base')
    compare('mass [kg]',       am.platform_mass,              m_xml, TOL_MASS, 'kg')
    compare('inertia [kg·m²]', np.diag(am.platform_inertia), I_xml, TOL_INERTIA)

    # ── Link 1 ──────────────────────────────────────────────────────────────
    print('\n[Link 1]')
    m_xml, I_xml = get_body_props(mj_model, 'link1')
    compare('mass [kg]',       am.links[0].mass,              m_xml, TOL_MASS)
    compare('inertia [kg·m²]', np.diag(am.links[0].inertia), I_xml, TOL_INERTIA)

    # ── Link 2 + EE combined inertia via parallel axis theorem ──────────────
    print('\n[Link 2 + EE combined  (parallel axis theorem)]')

    # ---- Step 1: collect body inertias and COMs at θ=0 from MuJoCo --------
    # All bodies lumped into link2 in model.py: link2, ee, finger_left, finger_right
    lump_bodies = ['link2', 'ee', 'finger_left', 'finger_right']
    mj_data_z = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data_z)
    mujoco.mj_forward(mj_model, mj_data_z)

    body_masses = {}
    body_coms_world = {}   # world-frame COM position
    body_inertias_world = {}  # inertia tensor in world frame (at body COM)

    for bname in lump_bodies:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, bname)
        m = float(mj_model.body_mass[bid])
        # COM position in world frame = body xpos + R_body @ ipos
        R_body = mj_data_z.xmat[bid].reshape(3, 3)
        ipos_local = mj_model.body_ipos[bid]   # COM in body frame
        com_world = mj_data_z.xpos[bid] + R_body @ ipos_local
        # Inertia at COM in body principal frame, rotate to world frame
        I_diag = mj_model.body_inertia[bid]    # [Ixx, Iyy, Izz] in body frame
        I_body = np.diag(I_diag)
        I_world = R_body @ I_body @ R_body.T
        body_masses[bname] = m
        body_coms_world[bname] = com_world
        body_inertias_world[bname] = I_world

    # ---- Step 2: combined mass and COM in world frame ----------------------
    m_total = sum(body_masses.values())
    com_combined_world = sum(body_masses[b] * body_coms_world[b]
                             for b in lump_bodies) / m_total

    print(f'  {"Combined mass [kg]":35s}: {m_total:.5f}')
    print(f'  {"model.py link2 mass [kg]":35s}: {am.links[1].mass:.5f}  '
          f'{"OK" if abs(am.links[1].mass - m_total) < TOL_MASS else "*** MISMATCH ***"}')
    print(f'  {"Combined COM (world, θ=0) [m]":35s}: {np.round(com_combined_world, 5)}')

    # ---- Step 3: combined inertia at combined COM (world frame) via PAT ----
    I_combined_world = np.zeros((3, 3))
    for bname in lump_bodies:
        m = body_masses[bname]
        r = body_coms_world[bname] - com_combined_world  # vector COM_body → combined COM
        # PAT: I_total = I_body_at_its_com + m*(|r|^2 I - r r^T)
        I_combined_world += body_inertias_world[bname] + m * (
            np.dot(r, r) * np.eye(3) - np.outer(r, r))

    # ---- Step 4: rotate combined inertia to link2 DH frame -----------------
    # At θ=0: link2 XML body frame has x-axis = link direction (+x world at hover).
    # DH {2} frame and XML link2 body share the same x-axis (link direction).
    # DH z_2 = joint2 axis = y in XML link2 body = world y at θ=0.
    # DH y_2 = z_2 × x_2 = world_y × world_x = -world_z.
    # So DH {2} → world frame rotation (columns = DH axes in world):
    #   x_DH2 → world [1, 0, 0]
    #   y_DH2 → world [0, 0, -1]
    #   z_DH2 → world [0, 1, 0]
    bid2 = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'link2')
    R_link2_world = mj_data_z.xmat[bid2].reshape(3, 3)   # link2 body frame in world
    # DH rotation relative to link2 body (at θ=0, link2 body = world at origin):
    # DH x = link2 x, DH z = link2 y, DH y = -link2 z
    R_DH2_in_link2 = np.array([[1, 0,  0],   # DH x = link2 x
                                [0, 0, -1],   # DH y = -link2 z
                                [0, 1,  0]])  # DH z = link2 y
    R_DH2_world = R_link2_world @ R_DH2_in_link2
    # Rotate inertia to DH frame: I_DH = R^T @ I_world @ R
    I_combined_DH = R_DH2_world.T @ I_combined_world @ R_DH2_world

    print(f'\n  Combined inertia in DH frame (at combined COM):')
    print(f'  Full tensor:\n{np.round(I_combined_DH, 8)}')
    I_diag_DH = np.diag(I_combined_DH)
    print(f'\n  Diagonal (Ixx, Iyy, Izz): {np.round(I_diag_DH, 8)}')
    print(f'\n  >>> PASTE INTO model.py link2 inertia field:')
    print(f'  np.diag([{I_diag_DH[0]:.5e}, {I_diag_DH[1]:.5e}, {I_diag_DH[2]:.5e}])')

    # Also show COM offset in DH frame (distance along DH x from joint2 origin)
    j2_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'joint2')
    j2_pos_world = mj_data_z.xanchor[j2_id]  # joint2 anchor in world
    com_from_j2_world = com_combined_world - j2_pos_world
    com_from_j2_DH = R_DH2_world.T @ com_from_j2_world
    print(f'\n  Combined COM from joint2 in DH frame: {np.round(com_from_j2_DH, 5)}')
    print(f'  >>> PASTE INTO model.py link2 com_offset:')
    print(f'  np.array([{com_from_j2_DH[0]:.5f}, {com_from_j2_DH[1]:.5f}, {com_from_j2_DH[2]:.5f}])')

    compare('\nmodel.py link2 inertia diag', np.diag(am.links[1].inertia), I_diag_DH, TOL_INERTIA)

    # ── COM offsets ──────────────────────────────────────────────────────────
    print('\n[COM offsets  (model.py DH x-axis = link direction)]')
    print(f'  {"Link 1 COM along link [m]":30s}  model={am.links[0].com_offset[0]:.4f}  xml=0.0600')
    err1 = abs(am.links[0].com_offset[0] - 0.06)
    print(f'  {"":30s}  err={err1:.4f} {"  OK" if err1 < 0.001 else "  *** MISMATCH ***"}')

    # ── Geometry (link lengths) ──────────────────────────────────────────────
    print('\n[Link lengths (DH a parameter)]')
    print(f'  {"Link 1  a [m]":30s}  model={am.links[0].a:.4f}  xml=0.1200  '
          f'{"  OK" if abs(am.links[0].a - 0.12) < 0.001 else "  *** MISMATCH ***"}')
    print(f'  {"Link 2  a [m]":30s}  model={am.links[1].a:.4f}  xml_geom=0.1600'
          f'  (EE site at 0.2380 from joint2)')

    # ── EE position sanity check ─────────────────────────────────────────────
    print('\n[FK sanity check at θ=0, drone at origin, level]')
    from ams.kinematics import forward_kinematics
    pos  = np.array([0.0, 0.0, 0.0])
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    theta = np.array([0.0, 0.0])
    _, p, _ = forward_kinematics(am, quat, pos, theta)
    print(f'  FK p[0] (mount)  : {np.round(p[0], 4)}')
    print(f'  FK p[1] (j1 end) : {np.round(p[1], 4)}')
    print(f'  FK p[2] (j2 end) : {np.round(p[2], 4)}')
    print(f'  FK p[3] (EE)     : {np.round(p[3], 4)}')
    print(f'  (mount_offset = {am.mount_offset})')
    print(f'  Expected at θ=0: EE should be ~[0, 0, -(0.05+0.12+0.16)] = [0,0,-0.33] ...')
    print(f'  (exact depends on mount_rotation and DH frame alignment)')

    # Check with MuJoCo at same configuration
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    # Set drone at origin, level, joints at 0
    mj_data.qpos[0:3] = [0.0, 0.0, 0.0]
    mj_data.qpos[3]   = 1.0   # quaternion w=1
    mujoco.mj_forward(mj_model, mj_data)
    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
    ee_mj   = mj_data.site_xpos[site_id].copy()
    print(f'\n  MuJoCo end_effector site at same config: {np.round(ee_mj, 4)}')
    fk_err = np.linalg.norm(p[3] - ee_mj)
    print(f'  FK vs MuJoCo EE error: {fk_err*1000:.1f} mm  '
          f'{"  OK (<10mm)" if fk_err < 0.01 else "  *** LARGE ERROR ***"}')

    print('\n' + '=' * 70)
    print('To fix remaining mismatches:')
    print('  1. Update inertia tensors in ams/model.py to match XML diaginertia values.')
    print('  2. Consider setting link2 a=0.238 to align FK p[3] with the EE site.')
    print('=' * 70)


if __name__ == '__main__':
    main()
