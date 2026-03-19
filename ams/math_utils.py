import numpy as np


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion [x,y,z,w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),        1 - 2*(x*x + y*y)],
    ])


def quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Quaternion derivative: q_dot = 0.5 * Omega(omega) @ q.

    q:     [x,y,z,w] quaternion
    omega: [wx,wy,wz] world-frame angular velocity
    """
    wx, wy, wz = omega
    Omega = np.array([
        [ 0,   -wz,  wy,  wx],
        [ wz,   0,  -wx,  wy],
        [-wy,  wx,   0,   wz],
        [-wx, -wy,  -wz,   0],
    ])
    return 0.5 * Omega @ q


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3-vector (for cross product: skew(a) @ b = a x b)."""
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0   ],
    ])


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product a x b."""
    return np.cross(a, b)
