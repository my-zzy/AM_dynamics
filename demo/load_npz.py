import numpy as np

data = np.load('demo/robustness_results.npz', allow_pickle=True)

wind_levels  = data['wind_levels']
mass_levels  = data['mass_levels']
wind_results = data['wind_results'].item()   # dict or None
mass_results = data['mass_results'].item()   # dict or None

METRIC_KEYS = [
    'ee_final_error_mm', 'ee_min_error_mm', 'ee_rmse_mm', 'settling_time_s',
    'drone_displacement_m', 'drone_path_length_m', 'drone_rms_vel_ms',
    'max_tilt_deg', 'joint1_rmse_deg', 'joint2_rmse_deg',
    'grasp_success', 'box_max_lift_z_m',
]


def print_sweep(sweep_name, results, levels, level_unit):
    if results is None:
        print(f'\n{sweep_name}: not run (None)\n')
        return
    methods = list(results.keys())
    print(f'\n{"=" * 70}')
    print(f'  {sweep_name}  ({level_unit})')
    print(f'{"=" * 70}')
    for method in methods:
        print(f'\n  ── {method} ──')
        print(f'  {"Level":>8}', end='')
        for k in METRIC_KEYS:
            print(f'  {k[:16]:>16}', end='')
        print()
        print('  ' + '-' * (9 + 18 * len(METRIC_KEYS)))
        for level, trial in zip(levels, results[method]):
            print(f'  {level:>8.3f}', end='')
            if trial is None:
                print('  (no data)')
                continue
            for k in METRIC_KEYS:
                val = trial.get(k, float('nan'))
                if isinstance(val, bool):
                    print(f'  {"Y" if val else "N":>16}', end='')
                elif isinstance(val, float) and np.isnan(val):
                    print(f'  {"nan":>16}', end='')
                else:
                    print(f'  {val:>16.4f}', end='')
            print()


print_sweep('WIND SWEEP', wind_results, wind_levels, 'N lateral')
print_sweep('MASS SWEEP', mass_results, mass_levels, 'scale factor')
