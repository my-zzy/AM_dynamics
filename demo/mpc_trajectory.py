"""Minimum-jerk trajectory generator for EE position tracking.

The minimum-jerk polynomial satisfies zero velocity and acceleration at
both endpoints, producing the smoothest possible point-to-point motion.

The 5th-order polynomial for a scalar dimension:
    s(t) = 10*tau^3 - 15*tau^4 + 6*tau^5,   tau = t / T_f
    s(0) = 0,   s(T_f) = 1
    s'(0) = 0,  s'(T_f) = 0
    s''(0) = 0, s''(T_f) = 0

Usage
-----
    traj = EETrajectory(p_start, p_end, T_f=4.0, dt=0.05)
    p_ref, v_ref = traj.get_ref(t)          # single time query
    p_seq, v_seq = traj.get_sequence()      # full sampled arrays (N+1 points)
"""

import numpy as np


class EETrajectory:
    """Minimum-jerk end-effector trajectory from p_start to p_end.

    Parameters
    ----------
    p_start : array_like, shape (3,)
        Initial EE position (world frame).
    p_end : array_like, shape (3,)
        Final   EE position (world frame).
    T_f : float
        Total trajectory duration in seconds.
    dt : float
        MPC time-step used for pre-sampling the reference sequence.
    v_start : array_like, shape (3,), optional
        Initial EE velocity (default zero).
    v_end : array_like, shape (3,), optional
        Final   EE velocity (default zero, enforces exact stop).
    """

    def __init__(self, p_start, p_end, T_f=4.0, dt=0.05,
                 v_start=None, v_end=None):
        self.p_start = np.asarray(p_start, dtype=float)
        self.p_end   = np.asarray(p_end,   dtype=float)
        self.T_f     = float(T_f)
        self.dt      = float(dt)

        self.v_start = (np.zeros(3) if v_start is None
                        else np.asarray(v_start, dtype=float))
        self.v_end   = (np.zeros(3) if v_end   is None
                        else np.asarray(v_end,   dtype=float))

        # Pre-sample for MPC reference sequence
        self._N         = int(round(T_f / dt))
        self._t_samples = np.linspace(0.0, T_f, self._N + 1)
        self._p_samples = np.vstack([self._p(t) for t in self._t_samples])  # (N+1, 3)
        self._v_samples = np.vstack([self._v(t) for t in self._t_samples])  # (N+1, 3)

    # ------------------------------------------------------------------
    # Internal polynomial evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _poly(tau):
        """Normalised displacement s(tau) in [0,1]."""
        return 10*tau**3 - 15*tau**4 + 6*tau**5

    @staticmethod
    def _poly_dot(tau):
        """Normalised velocity ds/dtau."""
        return 30*tau**2 - 60*tau**3 + 30*tau**4

    @staticmethod
    def _poly_ddot(tau):
        """Normalised acceleration d²s/dtau²."""
        return 60*tau - 180*tau**2 + 120*tau**3

    def _tau(self, t):
        t = float(np.clip(t, 0.0, self.T_f))
        return t / self.T_f

    def _p(self, t):
        tau = self._tau(t)
        s   = self._poly(tau)
        return self.p_start + s * (self.p_end - self.p_start)

    def _v(self, t):
        tau = self._tau(t)
        ds  = self._poly_dot(tau) / self.T_f
        return ds * (self.p_end - self.p_start)

    def _a(self, t):
        tau = self._tau(t)
        dds = self._poly_ddot(tau) / self.T_f**2
        return dds * (self.p_end - self.p_start)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def N(self):
        """Number of MPC shooting intervals in the full trajectory."""
        return self._N

    def get_ref(self, t):
        """Query reference at a continuous time t (seconds from trajectory start).

        Returns
        -------
        p_ref : (3,) ndarray  EE position reference
        v_ref : (3,) ndarray  EE velocity reference
        """
        return self._p(t), self._v(t)

    def get_accel_ref(self, t):
        """EE acceleration reference at time t."""
        return self._a(t)

    def get_sequence(self, t_start=0.0, horizon=None):
        """Return a sampled reference sequence starting at t_start.

        Useful for initialising the MPC reference at each real-time step.

        Parameters
        ----------
        t_start : float
            Current time within the trajectory (clamped to [0, T_f]).
        horizon : int or None
            Number of points to return (inclusive of t_start).
            Defaults to N+1 (full trajectory from t_start=0).

        Returns
        -------
        p_seq : (H, 3) ndarray  position references
        v_seq : (H, 3) ndarray  velocity references
        t_seq : (H,)   ndarray  corresponding times
        """
        if horizon is None:
            horizon = self._N + 1

        t_seq = t_start + np.arange(horizon) * self.dt
        p_seq = np.vstack([self._p(t) for t in t_seq])
        v_seq = np.vstack([self._v(t) for t in t_seq])
        return p_seq, v_seq, t_seq

    def is_finished(self, t):
        """Return True once t >= T_f."""
        return float(t) >= self.T_f

    def remaining_time(self, t):
        """Seconds remaining in trajectory."""
        return max(0.0, self.T_f - float(t))


# ---------------------------------------------------------------------------
# Validation / quick plot
# ---------------------------------------------------------------------------

def _validate():
    import sys

    p0 = np.array([0.0,  0.0, 1.0])
    pf = np.array([0.25, 0.0, 0.85])
    T  = 4.0
    dt = 0.05

    traj = EETrajectory(p0, pf, T_f=T, dt=dt)

    # --- Boundary conditions ---
    p_start, v_start = traj.get_ref(0.0)
    p_end,   v_end   = traj.get_ref(T)

    assert np.allclose(p_start, p0,                 atol=1e-10), f'p(0) wrong: {p_start}'
    assert np.allclose(p_end,   pf,                 atol=1e-10), f'p(T) wrong: {p_end}'
    assert np.allclose(v_start, np.zeros(3),        atol=1e-10), f'v(0) wrong: {v_start}'
    assert np.allclose(v_end,   np.zeros(3),        atol=1e-10), f'v(T) wrong: {v_end}'
    print('Boundary conditions: PASSED')

    # --- Monotonicity along x (p0[0] < pf[0]) ---
    ts = np.linspace(0, T, 1000)
    xs = np.array([traj._p(t)[0] for t in ts])
    assert np.all(np.diff(xs) >= -1e-10), 'Position not monotone in x'
    print('Monotone position:  PASSED')

    # --- Numerical velocity check ---
    eps = 1e-5
    for t_check in [0.5, 1.0, 2.0, 3.5]:
        p_plus  = traj._p(t_check + eps)
        p_minus = traj._p(t_check - eps)
        v_num   = (p_plus - p_minus) / (2 * eps)
        v_ana   = traj._v(t_check)
        assert np.allclose(v_num, v_ana, atol=1e-5), \
            f't={t_check}: numerical v={v_num} vs analytic v={v_ana}'
    print('Velocity (finite-diff): PASSED')

    # --- Numerical acceleration check ---
    for t_check in [0.5, 1.0, 2.0, 3.5]:
        v_plus  = traj._v(t_check + eps)
        v_minus = traj._v(t_check - eps)
        a_num   = (v_plus - v_minus) / (2 * eps)
        a_ana   = traj.get_accel_ref(t_check)
        assert np.allclose(a_num, a_ana, atol=1e-5), \
            f't={t_check}: numerical a={a_num} vs analytic a={a_ana}'
    print('Acceleration (finite-diff): PASSED')

    # --- get_sequence shape ---
    p_seq, v_seq, t_seq = traj.get_sequence(t_start=0.0, horizon=21)
    assert p_seq.shape == (21, 3), f'p_seq shape wrong: {p_seq.shape}'
    assert v_seq.shape == (21, 3)
    assert t_seq.shape == (21,)
    assert np.allclose(p_seq[0],  p0,           atol=1e-10)
    assert np.allclose(v_seq[0],  np.zeros(3),  atol=1e-10)
    print('get_sequence shape & BCs: PASSED')

    # --- get_sequence at mid-trajectory ---
    t_mid = T / 2
    p_seq2, _, _ = traj.get_sequence(t_start=t_mid, horizon=5)
    p_ref_direct = traj._p(t_mid)
    assert np.allclose(p_seq2[0], p_ref_direct, atol=1e-10)
    print('get_sequence mid-traj: PASSED')

    # --- is_finished ---
    assert not traj.is_finished(T - 0.01)
    assert     traj.is_finished(T)
    assert     traj.is_finished(T + 1.0)
    print('is_finished: PASSED')

    print('\nAll EETrajectory validation checks PASSED.')

    # Optional plot (skip in non-interactive environments)
    if '--plot' in sys.argv:
        import matplotlib.pyplot as plt
        t_arr = np.linspace(0, T, 500)
        p_arr = np.array([traj._p(t) for t in t_arr])
        v_arr = np.array([traj._v(t) for t in t_arr])
        a_arr = np.array([traj.get_accel_ref(t) for t in t_arr])
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        labels = ['x', 'y', 'z']
        for i, lbl in enumerate(labels):
            axes[0].plot(t_arr, p_arr[:, i], label=lbl)
            axes[1].plot(t_arr, v_arr[:, i], label=lbl)
            axes[2].plot(t_arr, a_arr[:, i], label=lbl)
        axes[0].set_ylabel('Position (m)')
        axes[1].set_ylabel('Velocity (m/s)')
        axes[2].set_ylabel('Acceleration (m/s²)')
        axes[2].set_xlabel('Time (s)')
        for ax in axes:
            ax.legend(loc='upper right')
            ax.grid(True)
        fig.suptitle('Minimum-jerk EE trajectory')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    _validate()
