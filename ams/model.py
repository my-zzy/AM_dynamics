import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class LinkParams:
    mass: float
    inertia: np.ndarray          # 3x3 inertia matrix about COM, in link frame
    com_offset: np.ndarray       # COM position in link frame (3,)
    alpha: float                 # DH twist angle
    a: float                     # DH link length
    d: float                     # DH link offset

    def dh_transform(self, theta: float) -> tuple:
        """Compute rotation matrix and translation from Craig DH parameters.

        Returns:
            R: (3,3) rotation matrix  {}^{i-1}R_i
            p: (3,)  translation      {}^{i-1}p_i
        """
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)

        R = np.array([
            [ct,       -st,       0   ],
            [st * ca,   ct * ca, -sa  ],
            [st * sa,   ct * sa,  ca  ],
        ])

        p = np.array([
            self.a,
            -self.d * sa,
             self.d * ca,
        ])

        return R, p


@dataclass
class AerialManipulatorModel:
    # --- Platform (quadrotor) ---
    platform_mass: float = 1.5   # kg
    platform_inertia: np.ndarray = field(
        default_factory=lambda: np.diag([0.008, 0.008, 0.015])  # kg·m²
    )

    # --- Gravity ---
    gravity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -9.81])     # world frame, z-up
    )

    # --- Arm mounting (fixed transform from platform frame to arm base frame) ---
    # Mounting point: bottom center of quadrotor
    mount_offset: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, -0.05])     # in platform frame
    )
    # Arm base frame {0} rotation (platform → arm base)
    # In platform coords: x_0 = [0,0,-1] (down), y_0 = [1,0,0] (fwd), z_0 = [0,-1,0] (right)
    # z_0 is the joint rotation axis; +θ swings arm forward
    mount_rotation: np.ndarray = field(
        default_factory=lambda: np.array([
            [0.0,  1.0,  0.0],
            [0.0,  0.0, -1.0],
            [-1.0, 0.0,  0.0],
        ])
    )

    # --- Manipulator links ---
    links: List[LinkParams] = field(default_factory=lambda: [
        # Link 1 (upper arm / boom): 0.25 m
        LinkParams(
            mass=0.15,
            inertia=np.diag([0.0001, 0.0008, 0.0008]),
            com_offset=np.array([0.125, 0.0, 0.0]),   # midpoint along link
            alpha=0.0,
            a=0.25,
            d=0.0,
        ),
        # Link 2 (forearm / stick): 0.20 m
        LinkParams(
            mass=0.12,
            inertia=np.diag([0.0001, 0.0004, 0.0004]),
            com_offset=np.array([0.10, 0.0, 0.0]),    # midpoint along link
            alpha=0.0,
            a=0.20,
            d=0.0,
        ),
    ])

    @property
    def n_joints(self) -> int:
        return len(self.links)

    @property
    def total_mass(self) -> float:
        return self.platform_mass + sum(link.mass for link in self.links)

    def compute_link_transforms(self, theta: np.ndarray):
        """Compute body-frame DH transforms for all frame transitions.

        Frame convention (body frames rotate with links):
          {0}: Fixed arm base at mount point (not rotating)
          {1}: Link 1 body frame at joint 1 (same origin as {0}, rotated by θ₁)
          {2}: Link 2 body frame at joint 2 (link 1 length from {1}, rotated by θ₂)
          {3}: End-effector (link 2 length from {2})

        Modified DH (Craig): transform {i}→{i+1} uses link i geometry (α_i, a_i)
        and joint i+1 angle (θ_{i+1}).  For {0}→{1}, no prior link → a=0.

        Args:
            theta: joint angles [θ₁, θ₂], shape (n,)

        Returns:
            R_local: list of (n+1) rotation matrices  [⁰R₁, ¹R₂, ²R₃]
            p_local: list of (n+1) translation vectors [⁰p₁, ¹p₂, ²p₃]
                     ⁱpᵢ₊₁ expressed in frame {i} coordinates
        """
        n = self.n_joints
        R_local = []
        p_local = []

        # {0} → {1}: pure rotation θ₁, no translation (origins coincide)
        ct, st = np.cos(theta[0]), np.sin(theta[0])
        R_local.append(np.array([[ct, -st, 0.0],
                                  [st,  ct, 0.0],
                                  [0.0, 0.0, 1.0]]))
        p_local.append(np.zeros(3))

        # {i} → {i+1} for i = 1, ..., n-1
        # Uses link[i-1] (NAME 'link i') geometry (α, a, d) with joint angle θ_{i}
        for i in range(1, n):
            R, p = self.links[i - 1].dh_transform(theta[i])
            R_local.append(R)
            p_local.append(p)

        # {n} → {n+1}: end-effector, last link geometry, no joint rotation
        R, p = self.links[-1].dh_transform(0.0)
        R_local.append(R)
        p_local.append(p)

        return R_local, p_local


if __name__ == '__main__':
    m = AerialManipulatorModel()
    theta = np.zeros(m.n_joints)

    print('=== DH Transforms at zero configuration ===')
    R_local, p_local = m.compute_link_transforms(theta)
    labels = [f'{{{i}}}→{{{i+1}}}' for i in range(m.n_joints + 1)]
    for label, R, p in zip(labels, R_local, p_local):
        print(f'\nTransform {label}:')
        print(f'  R = {np.round(R, 4).tolist()}')
        print(f'  p = {np.round(p, 4).tolist()}')

    print('\n=== DH Transforms at θ₁=45°, θ₂=30° ===')
    theta2 = np.radians([45, 30])
    R_local2, p_local2 = m.compute_link_transforms(theta2)
    for label, R, p in zip(labels, R_local2, p_local2):
        print(f'\nTransform {label}:')
        print(f'  R = {np.round(R, 4).tolist()}')
        print(f'  p = {np.round(p, 4).tolist()}')

    # Verify world-frame positions at zero config (identity platform)
    print('\n=== World positions at zero config (platform at [0,0,1]) ===')
    R_mount = m.mount_rotation
    p_0 = np.array([0.0, 0.0, 1.0]) + m.mount_offset  # frame {0}
    R_0 = R_mount  # identity platform rotation

    R_w = [R_0]     # world-frame rotations, starting with R_0
    p_w = [p_0]     # world-frame positions, starting with p_0
    for R, p in zip(R_local, p_local):
        R_w.append(R_w[-1] @ R)
        p_w.append(p_w[-1] + R_w[-2] @ p)

    frame_names = ['{0} (arm base)', '{1} (link 1)', '{2} (link 2)', '{3} (end-eff)']
    for name, p in zip(frame_names, p_w):
        print(f'  Frame {name}: {np.round(p, 4)}')

    # COM positions
    for i, link in enumerate(m.links):
        pc = p_w[i + 1] + R_w[i + 1] @ link.com_offset
        print(f'  Link {i+1} COM: {np.round(pc, 4)}')
