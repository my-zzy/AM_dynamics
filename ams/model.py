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
    # Arm base frame rotation (platform → arm base)
    # x_0 = down, y_0 = forward, z_0 = right
    # +θ swings arm forward (like raising an excavator boom)
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
