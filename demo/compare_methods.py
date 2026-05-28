"""Compare arm_only, drone_only and MPC on the grasp reach task.

Run from workspace root (no viewer, headless):
    conda activate main
    python demo/compare_methods.py

To include the MPC method (requires acados):
    python demo/compare_methods.py --mpc

Flags:
    --mpc              Also run the NMPC controller (slow, compiles acados).
    --rebuild          Force recompile of acados C code (only with --mpc).
    --dt-mpc=0.05      MPC shooting step in seconds.
    --horizon=20       MPC prediction horizon steps.
    --save-dir=demo    Directory for output figures.

Comparison metrics (computed over the *approach / reach* phase only)
----------------------------------------------------------------------
EE accuracy
    ee_final_error_mm   : ||p_EE - p_target|| at last timestep of phase [mm]
    ee_rmse_mm          : RMS EE error throughout the phase [mm]
    settling_time_s     : first time EE error stays < 15 mm for ≥ 0.5 s [s]

Platform disturbance
    drone_displacement_m  : ||p_drone_end - p_drone_start|| during phase [m]
    drone_path_length_m   : ∫||v_drone|| dt  (total path length) [m]
    drone_rms_vel_ms      : RMS drone speed during phase [m/s]
    max_tilt_deg          : max |roll| or |pitch| deviation from level [°]

Arm tracking
    joint1_rmse_deg       : RMSE(theta1_actual - theta1_des) [°]
    joint2_rmse_deg       : RMSE(theta2_actual - theta2_des) [°]

Grasp outcome
    grasp_success         : bool — box z ≥ 1.20 m at any time in LIFT phase
    box_max_lift_z_m      : max box z during LIFT phase [m]
"""

import os
import sys
import argparse
import numpy as np

_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _ROOT)

from demo.grasp_task import (
    run_grasp, BOX_TARGET,
    PHASE_TIMES,          # [(phase_id, t_end, label), ...]
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EE_TOL_M   = 0.015   # 15 mm settling tolerance
_SETTLE_WIN = 0.5     # seconds the error must stay below tolerance


def _quat_xyzw_to_rp(q):
    """Roll and pitch [rad] from quaternion [x, y, z, w]."""
    x, y, z, w = q
    roll  = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def compute_metrics(log, approach_phase_id, lift_phase_id, label=''):
    """Extract comparison metrics from a trajectory log.

    Parameters
    ----------
    log : dict
        Log produced by run_grasp() or run_mpc_grasp().
    approach_phase_id : int
        Phase number that corresponds to the reach / approach phase.
    lift_phase_id : int
        Phase number that corresponds to the LIFT phase.
    label : str
        Method name for printing.

    Returns
    -------
    dict  with all metric keys.  NaN if a metric could not be computed.
    """
    t       = np.array(log['t'])
    phases  = np.array(log['phase'])
    drone   = np.array(log['drone_pos'])    # (N, 3)
    vel     = np.array(log['vel'])          # (N, 3)
    quat    = np.array(log['quat'])         # (N, 4)  [x,y,z,w]
    ee      = np.array(log['ee_pos'])       # (N, 3)
    box     = np.array(log['box_pos'])      # (N, 3)
    theta   = np.array(log['theta'])        # (N, 2)  rad
    th_des  = np.array(log['theta_des'])    # (N, 2)  rad

    ap_mask = phases == approach_phase_id
    li_mask = phases == lift_phase_id

    nan = float('nan')
    metrics = dict(
        ee_final_error_mm   = nan,
        ee_min_error_mm     = nan,
        ee_rmse_mm          = nan,
        settling_time_s     = nan,
        drone_displacement_m= nan,
        drone_path_length_m = nan,
        drone_rms_vel_ms    = nan,
        max_tilt_deg        = nan,
        joint1_rmse_deg     = nan,
        joint2_rmse_deg     = nan,
        grasp_success       = False,
        box_max_lift_z_m    = nan,
    )

    # ── Approach-phase metrics ──────────────────────────────────────────────
    if not np.any(ap_mask):
        print(f'  [{label}] WARNING: no approach-phase data (phase={approach_phase_id})')
        return metrics

    t_ap    = t[ap_mask]
    ee_ap   = ee[ap_mask]       # (M, 3)
    dr_ap   = drone[ap_mask]    # (M, 3)
    vel_ap  = vel[ap_mask]      # (M, 3)
    quat_ap = quat[ap_mask]     # (M, 4)
    th_ap   = theta[ap_mask]    # (M, 2)
    thd_ap  = th_des[ap_mask]   # (M, 2)

    ee_err = np.linalg.norm(ee_ap - BOX_TARGET, axis=1)   # (M,)  metres

    # Final EE error
    metrics['ee_final_error_mm'] = float(ee_err[-1] * 1000.0)

    # Min EE error (best accuracy achieved, regardless of phase duration)
    metrics['ee_min_error_mm'] = float(np.min(ee_err) * 1000.0)

    # EE RMSE
    metrics['ee_rmse_mm'] = float(np.sqrt(np.mean(ee_err ** 2)) * 1000.0)

    # Settling time: first index where EE stays < tol for _SETTLE_WIN seconds
    dt_ap = float(np.median(np.diff(t_ap))) if len(t_ap) > 1 else 0.002
    win   = max(1, int(round(_SETTLE_WIN / dt_ap)))
    settle_t = nan
    for i in range(len(t_ap) - win + 1):
        if np.all(ee_err[i: i + win] < _EE_TOL_M):
            settle_t = float(t_ap[i] - t_ap[0])
            break
    metrics['settling_time_s'] = settle_t

    # Drone displacement (straight-line)
    metrics['drone_displacement_m'] = float(
        np.linalg.norm(dr_ap[-1] - dr_ap[0]))

    # Drone path length: integrate speed over time
    drone_speed  = np.linalg.norm(vel_ap, axis=1)   # (M,)
    dt_arr       = np.diff(t_ap)
    metrics['drone_path_length_m'] = float(
        np.sum(0.5 * (drone_speed[:-1] + drone_speed[1:]) * dt_arr))

    # Drone RMS speed
    metrics['drone_rms_vel_ms'] = float(np.sqrt(np.mean(drone_speed ** 2)))

    # Max tilt from level
    rp = np.array([_quat_xyzw_to_rp(q) for q in quat_ap])   # (M, 2)
    metrics['max_tilt_deg'] = float(np.degrees(np.max(np.abs(rp))))

    # Joint tracking RMSE
    joint_err = np.degrees(th_ap - thd_ap)   # (M, 2) degrees
    metrics['joint1_rmse_deg'] = float(np.sqrt(np.mean(joint_err[:, 0] ** 2)))
    metrics['joint2_rmse_deg'] = float(np.sqrt(np.mean(joint_err[:, 1] ** 2)))

    # ── Lift-phase metrics ──────────────────────────────────────────────────
    if np.any(li_mask):
        box_lift = box[li_mask, 2]
        metrics['box_max_lift_z_m'] = float(np.max(box_lift))
        metrics['grasp_success']    = bool(np.max(box_lift) >= 1.20)
    else:
        print(f'  [{label}] WARNING: no lift-phase data (phase={lift_phase_id})')

    return metrics


def print_metrics_table(results):
    """Pretty-print a comparison table to stdout."""
    methods = list(results.keys())
    metric_keys = [
        ('ee_final_error_mm',    'EE final error [mm]',      '{:.1f}'),
        ('ee_min_error_mm',      'EE min error [mm]',         '{:.1f}'),
        ('ee_rmse_mm',           'EE RMSE [mm]',              '{:.1f}'),
        ('settling_time_s',      'Settling time [s]',         '{:.2f}'),
        ('drone_displacement_m', 'Drone displacement [m]',    '{:.4f}'),
        ('drone_path_length_m',  'Drone path length [m]',     '{:.4f}'),
        ('drone_rms_vel_ms',     'Drone RMS speed [m/s]',     '{:.4f}'),
        ('max_tilt_deg',         'Max platform tilt [°]',     '{:.2f}'),
        ('joint1_rmse_deg',      'Joint-1 tracking RMSE [°]', '{:.2f}'),
        ('joint2_rmse_deg',      'Joint-2 tracking RMSE [°]', '{:.2f}'),
        ('grasp_success',        'Grasp success',              '{}'),
        ('box_max_lift_z_m',     'Max box lift z [m]',        '{:.3f}'),
    ]
    col_w = max(len(m) for m in methods) + 2
    row_w = 32

    header = ' ' * row_w + ''.join(f'{m:>{col_w}}' for m in methods)
    print('\n' + '=' * len(header))
    print('Comparison Metrics (approach / reach phase)')
    print('=' * len(header))
    print(header)
    print('-' * len(header))
    for key, label, fmt in metric_keys:
        row = f'{label:<{row_w}}'
        for m in methods:
            v = results[m].get(key, float('nan'))
            try:
                cell = fmt.format(v)
            except (ValueError, TypeError):
                cell = str(v)
            row += f'{cell:>{col_w}}'
        print(row)
    print('=' * len(header))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COLORS = {
    'arm_only':  'tab:blue',
    'drone_only': 'tab:green',
    'mpc':        'tab:red',
}
_LABELS = {
    'arm_only':  'Arm-only (PID)',
    'drone_only': 'Drone-only (PID)',
    'mpc':        'MPC (NMPC)',
}


def plot_ee_error_comparison(logs_ap, save_dir='.'):
    """Time-series EE error for each method during the approach phase."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    for method, (t_ap, ee_err_mm) in logs_ap.items():
        ax.plot(t_ap - t_ap[0], ee_err_mm,
                color=_COLORS.get(method, 'grey'),
                linewidth=1.8, label=_LABELS.get(method, method))
    ax.axhline(_EE_TOL_M * 1000, color='k', lw=1.0, ls='--',
               label=f'{_EE_TOL_M*1000:.0f} mm tolerance')
    ax.set_xlabel('Time within approach phase [s]')
    ax.set_ylabel('EE position error [mm]')
    ax.set_title('EE Error During Approach / Reach Phase')
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.4)
    plt.tight_layout()
    out = os.path.join(save_dir, 'compare_ee_error.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')
    try:
        plt.show()
    except Exception:
        pass


def plot_metric_bars(results, save_dir='.'):
    """Bar chart for each scalar metric across methods."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    scalar_metrics = [
        ('ee_final_error_mm',    'EE final error [mm]'),
        ('ee_min_error_mm',      'EE min error [mm]'),
        ('ee_rmse_mm',           'EE RMSE [mm]'),
        ('settling_time_s',      'Settling time [s]'),
        ('drone_displacement_m', 'Drone displacement [m]'),
        ('drone_path_length_m',  'Drone path length [m]'),
        ('drone_rms_vel_ms',     'Drone RMS speed [m/s]'),
        ('max_tilt_deg',         'Max tilt [°]'),
        ('joint1_rmse_deg',      'Joint-1 RMSE [°]'),
        ('joint2_rmse_deg',      'Joint-2 RMSE [°]'),
        ('box_max_lift_z_m',     'Box max lift z [m]'),
    ]

    methods  = list(results.keys())
    colors   = [_COLORS.get(m, 'grey') for m in methods]
    x        = np.arange(len(methods))
    n_plots  = len(scalar_metrics)
    ncols    = 2
    nrows    = (n_plots + 1) // ncols

    fig = plt.figure(figsize=(12, nrows * 3.0))
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.35)
    fig.suptitle('Method Comparison — Approach / Reach Phase', fontsize=13)

    for idx, (key, ylabel) in enumerate(scalar_metrics):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        vals = []
        for m in methods:
            v = results[m].get(key, float('nan'))
            vals.append(float(v) if not isinstance(v, bool) else float(v))
        bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor='k', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([_LABELS.get(m, m) for m in methods], fontsize=8, rotation=12)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis='y', lw=0.4)
        # Annotate bar values
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    out = os.path.join(save_dir, 'compare_metrics_bars.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')
    try:
        plt.show()
    except Exception:
        pass


def plot_ee_trajectory_overlay(logs_full, save_dir='.'):
    """xz EE trajectory overlay (all methods, full simulation)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for method, log in logs_full.items():
        ee = np.array(log['ee_pos'])
        ax.plot(ee[:, 0], ee[:, 2],
                color=_COLORS.get(method, 'grey'),
                lw=1.5, label=_LABELS.get(method, method), alpha=0.85)
        ax.scatter(ee[0, 0],  ee[0, 2],  color=_COLORS.get(method, 'grey'),
                   s=50, marker='o', zorder=5)
        ax.scatter(ee[-1, 0], ee[-1, 2], color=_COLORS.get(method, 'grey'),
                   s=50, marker='x', zorder=5)

    ax.scatter(BOX_TARGET[0], BOX_TARGET[2], color='limegreen',
               s=200, marker='*', zorder=7, label='Grasp target')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title('EE Trajectory Overlay  (o=start  x=end)')
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.4)
    plt.tight_layout()
    out = os.path.join(save_dir, 'compare_ee_trajectories.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')
    try:
        plt.show()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(run_mpc=False, mpc_kwargs=None, save_dir=None):
    if save_dir is None:
        save_dir = os.path.dirname(__file__)

    results  = {}   # method → metrics dict
    logs     = {}   # method → full log
    ap_plots = {}   # method → (t_ap, ee_err_mm) for time-series plot

    # ── PID methods ─────────────────────────────────────────────────────────
    for method in ('arm_only', 'drone_only'):
        print(f'\n{"="*60}')
        print(f' Running method: {method}')
        print('='*60)
        log = run_grasp(method=method, use_viewer=False, plot=False)
        if not log or not log['t']:
            print(f'  ERROR: no data returned for {method}')
            continue

        logs[method] = log

        # Approach phase = 3 (APPROACH), Lift phase = 5 (LIFT)
        m = compute_metrics(log,
                            approach_phase_id=3,
                            lift_phase_id=5,
                            label=method)
        results[method] = m

        # Build time-series for EE error plot (approach phase only)
        t       = np.array(log['t'])
        phases  = np.array(log['phase'])
        ee      = np.array(log['ee_pos'])
        ap_mask = phases == 3
        if np.any(ap_mask):
            ee_err_mm = np.linalg.norm(ee[ap_mask] - BOX_TARGET, axis=1) * 1000
            ap_plots[method] = (t[ap_mask], ee_err_mm)

    # ── MPC method ───────────────────────────────────────────────────────────
    if run_mpc:
        from demo.mpc_grasp_task import run_mpc_grasp

        print(f'\n{"="*60}')
        print(' Running method: mpc')
        print('='*60)
        kw = dict(use_viewer=False, plot=False, **mpc_kwargs) if mpc_kwargs else dict(use_viewer=False, plot=False)
        log = run_mpc_grasp(**kw)
        if log and log['t']:
            logs['mpc'] = log

            # Approach phase = 2 (MPC_REACH), Lift phase = 4 (LIFT)
            m = compute_metrics(log,
                                approach_phase_id=2,
                                lift_phase_id=4,
                                label='mpc')
            # Attach solve-time stats if available (stored separately in the log
            # under optional key 'solve_times')
            results['mpc'] = m

            t      = np.array(log['t'])
            phases = np.array(log['phase'])
            ee     = np.array(log['ee_pos'])
            ap_mask = phases == 2
            if np.any(ap_mask):
                ee_err_mm = np.linalg.norm(ee[ap_mask] - BOX_TARGET, axis=1) * 1000
                ap_plots['mpc'] = (t[ap_mask], ee_err_mm)

    # ── Report & plots ───────────────────────────────────────────────────────
    if not results:
        print('No results to compare.')
        return

    print_metrics_table(results)
    plot_ee_error_comparison(ap_plots, save_dir=save_dir)
    plot_metric_bars(results, save_dir=save_dir)
    plot_ee_trajectory_overlay(logs, save_dir=save_dir)

    return results, logs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='Compare arm_only / drone_only / MPC')
    p.add_argument('--mpc',      action='store_true',  help='Also run NMPC method')
    p.add_argument('--rebuild',  action='store_true',  help='Force recompile acados')
    p.add_argument('--dt-mpc',   type=float, default=0.05)
    p.add_argument('--horizon',  type=int,   default=20)
    p.add_argument('--save-dir', type=str,   default=None,
                   help='Directory for output figures (default: demo/)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    mpc_kw = dict(
        dt_mpc=args.dt_mpc,
        N=args.horizon,
        rebuild=args.rebuild,
        enable_terminal_constraint=True,
    )
    run_comparison(
        run_mpc=args.mpc,
        mpc_kwargs=mpc_kw,
        save_dir=args.save_dir,
    )
