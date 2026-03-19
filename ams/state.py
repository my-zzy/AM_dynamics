import numpy as np
from dataclasses import dataclass, field

N_JOINTS = 2


@dataclass
class PlatformState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))       # p_A  (3,)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))       # ṗ_A  (3,)
    quaternion: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))  # q_A [x,y,z,w] (4,)
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # ω_A (3,)


@dataclass
class ManipulatorState:
    joint_angles: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))      # θ   (n,)
    joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))   # θ̇   (n,)


@dataclass
class SystemState:
    platform: PlatformState = field(default_factory=PlatformState)
    manipulator: ManipulatorState = field(default_factory=ManipulatorState)

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.platform.position,          # 3
            self.platform.velocity,          # 3
            self.platform.quaternion,        # 4
            self.platform.angular_velocity,  # 3
            self.manipulator.joint_angles,   # n
            self.manipulator.joint_velocities,  # n
        ])

    @staticmethod
    def from_vector(x: np.ndarray) -> "SystemState":
        n = N_JOINTS
        return SystemState(
            platform=PlatformState(
                position=x[0:3],
                velocity=x[3:6],
                quaternion=x[6:10],
                angular_velocity=x[10:13],
            ),
            manipulator=ManipulatorState(
                joint_angles=x[13:13 + n],
                joint_velocities=x[13 + n:13 + 2 * n],
            ),
        )

    @staticmethod
    def state_size() -> int:
        return 13 + 2 * N_JOINTS
