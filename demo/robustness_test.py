"""Robustness sweep: wind disturbance, mass uncertainty, actuator delay, sensor noise.

Tests three methods (arm_only, drone_only, mpc) against:
  1. Lateral wind force  (0 → 1.0 N in X direction)
  2. Mass scale factor   (0.80 → 1.20, i.e. ±20%)
  3. Actuator delay      (0 → 200 ms)
  4. Sensor noise std    (0 → 50 mm position std)

Run from workspace root:
    conda activate main
    # Wind sweep (PID only, fast):
    python demo/robustness_test.py --test wind
    # Mass sweep:
    python demo/robustness_test.py --test mass
    # Actuator delay sweep:
    python demo/robustness_test.py --test delay
    # Sensor noise sweep:
    python demo/robustness_test.py --test noise
    # Include MPC (slow — acados required):
    python demo/robustness_test.py --test wind --mpc
    # All sweeps, all methods:
    python demo/robustness_test.py --test all --mpc

Output
------
  demo/robustness_wind.png    — success rate and min EE error vs wind magnitude
  demo/robustness_mass.png    — same vs mass scale factor
  demo/robustness_delay.png   — same vs actuator delay
  demo/robustness_noise.png   — same vs sensor noise std
  demo/robustness_results.npz — raw results for further analysis
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _ROOT)

from demo.grasp_task import run_grasp, BOX_TARGET
from demo.compare_methods import compute_metrics

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

# Wind sweep: lateral force in world X direction [N]
WIND_LEVELS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

# Mass scale sweep: fraction of nominal mass
MASS_LEVELS = np.array([0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20])

# Actuator delay sweep [seconds]
DELAY_LEVELS = np.array([0.0, 0.005, 0.01, 0.015, 0.020, 0.025])

# Sensor noise std sweep [metres] — scales all sensor noise proportionally
NOISE_LEVELS = np.array([0.0, 0.001, 0.002, 0.003, 0.004, 0.005])

# Phase IDs used by each method's log
_PID_APPROACH = 3
_PID_LIFT     = 5
_MPC_APPROACH = 2
_MPC_LIFT     = 4

# ---------------------------------------------------------------------------
# Single-trial runner
# ---------------------------------------------------------------------------

def run_pid_trial(method, wind_force=None, mass_scale=1.0,
                  actuator_delay=0.0, sensor_noise_std=0.0):
    """Run one PID trial and return (metrics_dict, grasp_success)."""
    log = run_grasp(
        method=method,
        use_viewer=False,
        plot=False,
        wind_force=wind_force,
        mass_scale=mass_scale,
        actuator_delay=actuator_delay,
        sensor_noise_std=sensor_noise_std,
    )
    if not log or not log['t']:
        return None, False
    m = compute_metrics(log,
                        approach_phase_id=_PID_APPROACH,
                        lift_phase_id=_PID_LIFT,
                        label=method)
    return m, m['grasp_success']


def run_mpc_trial(wind_force=None, mass_scale=1.0,
                  dt_mpc=0.05, N=20, rebuild=False,
                  actuator_delay=0.0, sensor_noise_std=0.0):
    """Run one MPC trial and return (metrics_dict, grasp_success)."""
    from demo.mpc_grasp_task import run_mpc_grasp
    log = run_mpc_grasp(
        dt_mpc=dt_mpc, N=N, rebuild=rebuild,
        enable_terminal_constraint=True,
        use_viewer=False, plot=False,
        wind_force=wind_force,
        mass_scale=mass_scale,
        actuator_delay=actuator_delay,
        sensor_noise_std=sensor_noise_std,
    )
    if not log or not log['t']:
        return None, False
    m = compute_metrics(log,
                        approach_phase_id=_MPC_APPROACH,
                        lift_phase_id=_MPC_LIFT,
                        label='mpc')
    return m, m['grasp_success']


# ---------------------------------------------------------------------------
# Wind sweep
# ---------------------------------------------------------------------------

def run_wind_sweep(run_mpc=False, mpc_kwargs=None, save_dir=None):
    """Sweep lateral wind force for all methods.

    Returns
    -------
    results : dict  keyed by method name, each value is a list of metric
              dicts (one per wind level in WIND_LEVELS).
    """
    mpc_kw = mpc_kwargs or {}
    methods_pid = ['arm_only', 'drone_only']
    all_methods = methods_pid + (['mpc'] if run_mpc else [])

    # results[method][wind_idx] = metrics dict or None
    results = {m: [] for m in all_methods}

    print('\n' + '=' * 60)
    print('WIND SWEEP')
    print('  Levels (N):', WIND_LEVELS)
    print('  Methods    :', all_methods)
    print('=' * 60)

    for fi, f_mag in enumerate(WIND_LEVELS):
        wind = np.array([f_mag, 0.0, 0.0])  # lateral in world X
        print(f'\n--- Wind {f_mag:.2f} N ---')

        for method in methods_pid:
            print(f'  [{method}] ', end='', flush=True)
            m, ok = run_pid_trial(method, wind_force=wind if f_mag > 0 else None)
            results[method].append(m)
            if m:
                print(f'success={ok}  ee_min={m["ee_min_error_mm"]:.1f} mm  '
                      f'tilt={m["max_tilt_deg"]:.1f}°')
            else:
                print('FAILED (no log)')

        if run_mpc:
            print('  [mpc] ', end='', flush=True)
            m, ok = run_mpc_trial(wind_force=wind if f_mag > 0 else None, **mpc_kw)
            results['mpc'].append(m)
            if m:
                print(f'success={ok}  ee_min={m["ee_min_error_mm"]:.1f} mm  '
                      f'tilt={m["max_tilt_deg"]:.1f}°')
            else:
                print('FAILED (no log)')

    _plot_robustness(results, WIND_LEVELS, x_label='Wind force [N]',
                     fname='robustness_wind.png', save_dir=save_dir)
    return results


# ---------------------------------------------------------------------------
# Mass sweep
# ---------------------------------------------------------------------------

def run_mass_sweep(run_mpc=False, mpc_kwargs=None, save_dir=None):
    """Sweep mass scale factor for all methods.

    Returns
    -------
    results : dict  keyed by method name.
    """
    mpc_kw = mpc_kwargs or {}
    methods_pid = ['arm_only', 'drone_only']
    all_methods = methods_pid + (['mpc'] if run_mpc else [])

    results = {m: [] for m in all_methods}

    print('\n' + '=' * 60)
    print('MASS SCALE SWEEP')
    print('  Levels:', MASS_LEVELS)
    print('  Methods:', all_methods)
    print('=' * 60)

    for si, scale in enumerate(MASS_LEVELS):
        print(f'\n--- Mass scale {scale:.2f} ---')

        for method in methods_pid:
            print(f'  [{method}] ', end='', flush=True)
            m, ok = run_pid_trial(method, mass_scale=scale)
            results[method].append(m)
            if m:
                print(f'success={ok}  ee_min={m["ee_min_error_mm"]:.1f} mm')
            else:
                print('FAILED (no log)')

        if run_mpc:
            print('  [mpc] ', end='', flush=True)
            m, ok = run_mpc_trial(mass_scale=scale, **mpc_kw)
            results['mpc'].append(m)
            if m:
                print(f'success={ok}  ee_min={m["ee_min_error_mm"]:.1f} mm')
            else:
                print('FAILED (no log)')

    _plot_robustness(results, MASS_LEVELS, x_label='Mass scale factor',
                     fname='robustness_mass.png', save_dir=save_dir)
    return results


# ---------------------------------------------------------------------------
# Actuator delay sweep
# ---------------------------------------------------------------------------

def run_delay_sweep(run_mpc=False, mpc_kwargs=None, save_dir=None):
    """Sweep actuator delay for all methods.

    Returns
    -------
    results : dict  keyed by method name, each value is a list of metric
              dicts (one per delay level in DELAY_LEVELS).
    """
    mpc_kw = mpc_kwargs or {}
    methods_pid = ['arm_only', 'drone_only']
    all_methods = methods_pid + (['mpc'] if run_mpc else [])

    results = {m: [] for m in all_methods}

    print('\n' + '=' * 60)
    print('ACTUATOR DELAY SWEEP')
    print('  Levels (s):', DELAY_LEVELS)
    print('  Methods   :', all_methods)
    print('=' * 60)

    for di, delay in enumerate(DELAY_LEVELS):
        print(f'\n--- Actuator delay {delay*1000:.0f} ms ---')

        for method in methods_pid:
            print(f'  [{method}] ', end='', flush=True)
            m, ok = run_pid_trial(method, actuator_delay=delay)
            results[method].append(m)
            if m:
                print(f'success={ok}  ee_min={m["ee_min_error_mm"]:.1f} mm  '
                      f'tilt={m["max_tilt_deg"]:.1f}°')
            else:
                print('FAILED (no log)')

        if run_mpc:
            print('  [mpc] ', end='', flush=True)
            m, ok = run_mpc_trial(actuator_delay=delay, **mpc_kw)
            results['mpc'].append(m)
            if m:
                print(f'success={ok}  ee_min={m["ee_min_error_mm"]:.1f} mm  '
                      f'tilt={m["max_tilt_deg"]:.1f}°')
            else:
                print('FAILED (no log)')

    _plot_robustness(results, DELAY_LEVELS * 1000, x_label='Actuator delay [ms]',
                     fname='robustness_delay.png', save_dir=save_dir)
    return results


# ---------------------------------------------------------------------------
# Sensor noise sweep
# ---------------------------------------------------------------------------

def run_noise_sweep(run_mpc=False, mpc_kwargs=None, save_dir=None):
    """Sweep sensor noise std for all methods.

    Returns
    -------
    results : dict  keyed by method name.
    """
    mpc_kw = mpc_kwargs or {}
    methods_pid = ['arm_only', 'drone_only']
    all_methods = methods_pid + (['mpc'] if run_mpc else [])

    results = {m: [] for m in all_methods}

    print('\n' + '=' * 60)
    print('SENSOR NOISE SWEEP')
    print('  Levels (m):', NOISE_LEVELS)
    print('  Methods   :', all_methods)
    print('=' * 60)

    for ni, noise in enumerate(NOISE_LEVELS):
        print(f'\n--- Sensor noise std {noise*1000:.1f} mm ---')

        for method in methods_pid:
            print(f'  [{method}] ', end='', flush=True)
            m, ok = run_pid_trial(method, sensor_noise_std=noise)
            results[method].append(m)
            if m:
                print(f'success={ok}  ee_min={m["ee_min_error_mm"]:.1f} mm  '
                      f'tilt={m["max_tilt_deg"]:.1f}°')
            else:
                print('FAILED (no log)')

        if run_mpc:
            print('  [mpc] ', end='', flush=True)
            m, ok = run_mpc_trial(sensor_noise_std=noise, **mpc_kw)
            results['mpc'].append(m)
            if m:
                print(f'success={ok}  ee_min={m["ee_min_error_mm"]:.1f} mm  '
                      f'tilt={m["max_tilt_deg"]:.1f}°')
            else:
                print('FAILED (no log)')

    _plot_robustness(results, NOISE_LEVELS * 1000, x_label='Sensor noise std [mm]',
                     fname='robustness_noise.png', save_dir=save_dir)
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COLORS = {
    'arm_only':  'tab:blue',
    'drone_only': 'tab:orange',
    'mpc':       'tab:green',
}
_MARKERS = {
    'arm_only':  'o',
    'drone_only': 's',
    'mpc':       '^',
}


def _plot_robustness(results, x_vals, x_label, fname, save_dir=None):
    """Two-panel plot: min EE error and grasp success rate vs perturbation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Robustness sweep — {x_label}', fontsize=13)

    ax_ee, ax_tilt, ax_ok = axes

    for method, trial_list in results.items():
        ee_min   = []
        tilt_max = []
        success  = []
        for m in trial_list:
            if m is None:
                ee_min.append(np.nan)
                tilt_max.append(np.nan)
                success.append(np.nan)
            else:
                ee_min.append(m['ee_min_error_mm'])
                tilt_max.append(m['max_tilt_deg'])
                success.append(1.0 if m['grasp_success'] else 0.0)

        kw = dict(color=_COLORS.get(method, 'gray'),
                  marker=_MARKERS.get(method, 'x'),
                  linewidth=1.8, markersize=6, label=method)
        ax_ee.plot(x_vals, ee_min,   **kw)
        ax_tilt.plot(x_vals, tilt_max, **kw)
        ax_ok.plot(x_vals, success,  **kw)

    ax_ee.set_xlabel(x_label)
    ax_ee.set_ylabel('Min EE error [mm]')
    ax_ee.set_title('EE accuracy (best in approach)')
    ax_ee.axhline(5.0, color='red', lw=0.8, ls='--', label='5 mm tol')
    ax_ee.axhline(15.0, color='gray', lw=0.8, ls=':', label='15 mm tol')
    ax_ee.legend(fontsize=8)
    ax_ee.grid(True, lw=0.4)

    ax_tilt.set_xlabel(x_label)
    ax_tilt.set_ylabel('Max tilt [°]')
    ax_tilt.set_title('Platform tilt (approach phase)')
    ax_tilt.legend(fontsize=8)
    ax_tilt.grid(True, lw=0.4)

    ax_ok.set_xlabel(x_label)
    ax_ok.set_ylabel('Grasp success')
    ax_ok.set_title('Grasp success (1 = lifted box ≥ 1.20 m)')
    ax_ok.set_ylim(-0.05, 1.05)
    ax_ok.set_yticks([0, 1])
    ax_ok.set_yticklabels(['fail', 'success'])
    ax_ok.legend(fontsize=8)
    ax_ok.grid(True, lw=0.4)

    plt.tight_layout()
    out_dir = save_dir or os.path.dirname(__file__)
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved → {out_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_robustness_table(results, x_vals, sweep_name):
    """Print ASCII table of min EE error and grasp success per level."""
    methods = list(results.keys())
    col_w   = 14

    print(f'\n{"=" * 60}')
    print(f'  {sweep_name} robustness — ee_min_error_mm  (grasp? Y/N)')
    print(f'{"=" * 60}')
    header = f'{"Level":>10}' + ''.join(f'{m:>{col_w}}' for m in methods)
    print(header)
    print('-' * len(header))
    for i, xv in enumerate(x_vals):
        row = f'{xv:>10.3f}'
        for m in methods:
            trial = results[m][i]
            if trial is None:
                row += f'{"ERR":>{col_w}}'
            else:
                val  = trial['ee_min_error_mm']
                flag = 'Y' if trial['grasp_success'] else 'N'
                row += f'{val:>8.1f} mm {flag:>3}{" " * (col_w - 14)}'
        print(row)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='Robustness sweep for grasp methods')
    p.add_argument('--test',
                   choices=['wind', 'mass', 'delay', 'noise', 'all'],
                   default='all',
                   help='Which sweep to run (default: all)')
    p.add_argument('--mpc',  action='store_true',
                   help='Also test the MPC method (requires acados)')
    p.add_argument('--rebuild', action='store_true',
                   help='Force recompile acados C code (MPC only)')
    p.add_argument('--dt-mpc', type=float, default=0.05,
                   dest='dt_mpc', metavar='DT')
    p.add_argument('--horizon', type=int, default=20,
                   dest='N', metavar='N')
    p.add_argument('--save-dir', default=None, dest='save_dir',
                   help='Directory for output figures (default: demo/)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    mpc_kw = dict(dt_mpc=args.dt_mpc, N=args.N, rebuild=args.rebuild)
    run_mpc = args.mpc

    wind_results  = None
    mass_results  = None
    delay_results = None
    noise_results = None

    if args.test in ('wind', 'all'):
        wind_results = run_wind_sweep(run_mpc=run_mpc,
                                      mpc_kwargs=mpc_kw,
                                      save_dir=args.save_dir)
        print_robustness_table(wind_results, WIND_LEVELS, 'Wind force [N]')

    if args.test in ('mass', 'all'):
        mass_results = run_mass_sweep(run_mpc=run_mpc,
                                      mpc_kwargs=mpc_kw,
                                      save_dir=args.save_dir)
        print_robustness_table(mass_results, MASS_LEVELS, 'Mass scale')

    if args.test in ('delay', 'all'):
        delay_results = run_delay_sweep(run_mpc=run_mpc,
                                        mpc_kwargs=mpc_kw,
                                        save_dir=args.save_dir)
        print_robustness_table(delay_results, DELAY_LEVELS * 1000, 'Actuator delay [ms]')

    if args.test in ('noise', 'all'):
        noise_results = run_noise_sweep(run_mpc=run_mpc,
                                        mpc_kwargs=mpc_kw,
                                        save_dir=args.save_dir)
        print_robustness_table(noise_results, NOISE_LEVELS * 1000, 'Sensor noise std [mm]')

    # Save raw data
    out_dir  = args.save_dir or os.path.dirname(__file__)
    npz_path = os.path.join(out_dir, 'robustness_results.npz')
    np.savez(npz_path,
             wind_levels=WIND_LEVELS,
             mass_levels=MASS_LEVELS,
             delay_levels=DELAY_LEVELS,
             noise_levels=NOISE_LEVELS,
             wind_results=wind_results,
             mass_results=mass_results,
             delay_results=delay_results,
             noise_results=noise_results)
    print(f'Raw results saved → {npz_path}')
