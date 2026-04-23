"""Cascade PID controller for the aerial manipulator drone base.

Structure (outer → inner):
  Position loop  : (x, y, z) error → desired tilt angles + thrust T
  Attitude loop  : (roll, pitch, yaw) error → body torques [tau_x, tau_y, tau_z]

Control output: (T, tau_body) — matches apply_platform_control() in test_model.py.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _wrap(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def quat_to_euler_zyx(q_xyzw):
    """ZYX Euler angles (roll φ, pitch θ, yaw ψ) from quaternion [x,y,z,w]."""
    x, y, z, w = q_xyzw
    roll  = np.arctan2(2*(w*x + y*z),  1 - 2*(x*x + y*y))
    sinp  = np.clip(2*(w*y - z*x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw   = np.arctan2(2*(w*z + x*y),  1 - 2*(y*y + z*z))
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Single-axis PID
# ---------------------------------------------------------------------------

class PID:
    """PID controller with clamped integral and output."""

    def __init__(self, kp, ki, kd, ilim=np.inf, olim=np.inf):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.ilim, self.olim = ilim, olim
        self._integral   = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0

    def update(self, error, dt):
        self._integral   = np.clip(self._integral + error * dt, -self.ilim, self.ilim)
        deriv            = (error - self._prev_error) / dt if dt > 1e-9 else 0.0
        self._prev_error = error
        out = self.kp * error + self.ki * self._integral + self.kd * deriv
        return float(np.clip(out, -self.olim, self.olim))


# ---------------------------------------------------------------------------
# Cascade drone controller
# ---------------------------------------------------------------------------

class DroneController:
    """Cascade PID position + attitude controller for a quadrotor.

    Outer loop  (position):
        z error          → thrust T  (via altitude PID + gravity feedforward)
        x, y errors      → desired roll / pitch  (PD → desired accel → tilt angle)

    Inner loop  (attitude):
        roll, pitch, yaw errors → body torques [tau_x, tau_y, tau_z]

    Usage
    -----
        ctrl = DroneController(mass=total_mass)
        ctrl.reset()
        # in sim loop:
        T, tau = ctrl.compute(pos, vel, q_xyzw, omega, pos_des, yaw_des, dt)
        apply_platform_control(data, T, tau)
    """

    def __init__(self, mass, g=9.81,
                 # --- altitude ---
                 kp_z=10.0, ki_z=2.0, kd_z=6.0,
                 # --- horizontal position (PD, no integral) ---
                 kp_xy=2.5, kd_xy=3.0,
                 # --- attitude roll/pitch ---
                 kp_rp=9.0, ki_rp=0.5, kd_rp=2.5,
                 # --- attitude yaw ---
                 kp_yaw=5.0, ki_yaw=0.2, kd_yaw=1.5,
                 # --- limits ---
                 tilt_limit=np.deg2rad(30),
                 tau_limit=3.0):
        self.mass       = mass
        self.g          = g
        self.mg         = mass * g
        self.tilt_limit = tilt_limit

        # Altitude
        self.pid_z     = PID(kp_z, ki_z, kd_z, ilim=5.0, olim=2 * mass * g)
        # Horizontal (stored as gains; no integral to avoid windup during maneuvers)
        self.kp_xy     = kp_xy
        self.kd_xy     = kd_xy
        # Attitude
        self.pid_roll  = PID(kp_rp,  ki_rp,  kd_rp,  ilim=1.0, olim=tau_limit)
        self.pid_pitch = PID(kp_rp,  ki_rp,  kd_rp,  ilim=1.0, olim=tau_limit)
        self.pid_yaw   = PID(kp_yaw, ki_yaw, kd_yaw, ilim=1.0, olim=tau_limit)

    def reset(self):
        for pid in (self.pid_z, self.pid_roll, self.pid_pitch, self.pid_yaw):
            pid.reset()

    def compute(self, pos, vel, q_xyzw, omega, pos_des, yaw_des, dt):
        """Compute control output.

        Parameters
        ----------
        pos      : (3,) world position
        vel      : (3,) world velocity
        q_xyzw   : (4,) quaternion [x,y,z,w]
        omega    : (3,) world-frame angular velocity (unused here, kept for API)
        pos_des  : (3,) desired world position
        yaw_des  : float desired yaw angle [rad]
        dt       : float timestep [s]

        Returns
        -------
        T        : float  thrust [N]
        tau_body : (3,)   body-frame torques [N·m]
        """
        roll, pitch, yaw = quat_to_euler_zyx(q_xyzw)

        # ---- Altitude: PID → thrust ----
        e_z      = pos_des[2] - pos[2]
        T_level  = self.mg + self.pid_z.update(e_z, dt)   # gravity feedforward
        # Tilt compensation: actual vertical force = T·cos(roll)·cos(pitch)
        # Divide by cos product so vertical component stays correct when tilted
        cos_tilt = max(np.cos(roll) * np.cos(pitch), 0.5)   # clamp to avoid blow-up
        T        = T_level / cos_tilt

        # ---- Horizontal: PD → desired tilt ----
        e_x = pos_des[0] - pos[0]
        e_y = pos_des[1] - pos[1]
        # Desired world-frame horizontal acceleration
        a_x = self.kp_xy * e_x - self.kd_xy * vel[0]
        a_y = self.kp_xy * e_y - self.kd_xy * vel[1]
        # Rotate into body frame (yaw only — ignore roll/pitch for small angles)
        cy, sy   = np.cos(yaw), np.sin(yaw)
        a_bx     =  cy * a_x + sy * a_y   # body forward (+x)
        a_by     = -sy * a_x + cy * a_y   # body left (+y)
        # Small-angle: a = g * tilt_angle
        pitch_des = np.clip( a_bx / self.g, -self.tilt_limit, self.tilt_limit)
        roll_des  = np.clip(-a_by / self.g, -self.tilt_limit, self.tilt_limit)

        # ---- Attitude: PID → body torques ----
        tau_x = self.pid_roll.update( roll_des  - roll,               dt)
        tau_y = self.pid_pitch.update(pitch_des - pitch,              dt)
        tau_z = self.pid_yaw.update(  _wrap(yaw_des - yaw),          dt)

        return T, np.array([tau_x, tau_y, tau_z])
