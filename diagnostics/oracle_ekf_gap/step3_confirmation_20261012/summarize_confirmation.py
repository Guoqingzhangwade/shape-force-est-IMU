"""Evaluate the preregistered Step 3 recurrence rule from saved JSON; run no simulations.

Example:
    python -B summarize_confirmation.py --output-dir "$env:TEMP/step3-summary-new"

Reads the saved A/B/C/D and sensitivity results beside this file, plus the
preserved case-0 Step 1/Step 2 results. Existing outputs are never overwritten.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

STEP = Path(__file__).resolve().parent
BASE = STEP.parent

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def max_relative_difference(a, b):
    """Largest elementwise relative difference over every shared numeric entry."""
    if isinstance(a, dict):
        return max([max_relative_difference(a[k], b[k]) for k in a if k in b], default=0.0)
    if isinstance(a, bool) or isinstance(a, str) or a is None:
        return 0.0
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert x.shape == y.shape
    return float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-12))) if x.size else 0.0

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='Fresh directory; existing outputs are never replaced.')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        parser.error(f'Output directory must be new: {out}')
    protocol = load(STEP / 'protocol.json')
    threshold = protocol['descriptive_recurrence_threshold']
    inputs = {0: (BASE / 'step1_frechet_20261008/after_abcd/results.json', STEP / 'case_0/sensitivity_verified/results.json')}
    for case in protocol['selected_case_ids']:
        inputs[case] = (STEP / f'case_{case}/abcd/results.json', STEP / f'case_{case}/sensitivity_verified/results.json')
    completion = load(STEP / 'sensitivity_verified_completion.json')
    assert completion['case_ids'] == sorted(inputs) and not any(completion['returncodes'])

    summary, case_rows, perturbation_rows = {}, [], []
    for case, (fit_path, sensitivity_path) in inputs.items():
        fit, data = load(fit_path), load(sensitivity_path)
        settings = data['settings']
        assert settings['case'] == fit['case_id'] == case and settings['noise'] == protocol['noise_realization']
        assert data['head'] == protocol['source_commit'] and data['inputs_sources_and_status_unchanged']
        a = data['cases']['A']
        gains = a['directional_gains']
        V, singular = np.array(gains['V']), np.array(gains['singular_values'])
        per_noise, J = np.array(gains['per_noise_gain']), np.array(a['J_wrench_scaled_q'])
        most = int(np.argmax(per_noise))
        F0 = settings['F0_N']
        # The driver asserts these before saving; restate them from the saved numbers.
        checks = {
            'body_rank_six_all_states': all(c['body_retained_rank'] == 6 for c in data['cases'].values()),
            'measurement_rank_eight_all_states': all(c['directional_gains']['measurement_retained_rank'] == 8 for c in data['cases'].values()),
            'H_fd_max_relative_error': max(c['H_fd_relative_error'] for c in data['cases'].values()),
            'full_wrench_derivative_max_relative_error': max(c['J_fd_relative_error'] for c in data['cases'].values()),
            'A_half_step_relative_change': data['A_derivative_half_step_relative_change'],
        }
        checks['derivative_checks_pass'] = bool(checks['H_fd_max_relative_error'] < 1e-7
            and checks['full_wrench_derivative_max_relative_error'] < 3e-3 and checks['A_half_step_relative_change'] < 3e-3)
        ratios, targets_met, caps = {}, [], []
        for row in data['controlled_perturbations']:
            caps.append(row['curvature_cap_active'])
            targets_met.append(abs(row['actual_measurement_change_whitened_norm'] / row['target_stacked_whitened_norm'] - 1) < 1e-3)
            perturbation_rows.append({'case_id': case, 'direction_1_based': row['direction_index'] + 1,
                'target_stacked_whitened_norm': row['target_stacked_whitened_norm'], 'sign': row['sign'],
                'stacked_residual_change_deg': row['stacked_measurement_change_deg'], 'force_change_N': row['force_change_N'],
                'moment_change_common_base_Nm': row['moment_change_common_base_Nm'],
                'scaled_wrench_change_norm': float(np.linalg.norm(row['actual_delta_wrench_scaled'])),
                'shape_change_rms_mm': row['shape_change_rms_mm'],
                'relative_linear_prediction_error': row['relative_linear_prediction_error']})
        for sign in (-1, 1):
            chosen = {row['direction_index']: row for row in data['controlled_perturbations']
                      if row['sign'] == sign and row['target_stacked_whitened_norm'] == 0.1}
            ratios[str(sign)] = {
                'scaled_wrench_ratio': float(np.linalg.norm(chosen[most]['actual_delta_wrench_scaled']) / np.linalg.norm(chosen[0]['actual_delta_wrench_scaled'])),
                'force_ratio': chosen[most]['force_change_N'] / chosen[0]['force_change_N'],
                'largest_gain_force_change_mN': 1000 * chosen[most]['force_change_N'],
                'strongest_H_force_change_mN': 1000 * chosen[0]['force_change_N'],
                'largest_gain_shape_change_rms_mm': chosen[most]['shape_change_rms_mm'],
                'strongest_H_shape_change_rms_mm': chosen[0]['shape_change_rms_mm']}
        recurrence = bool(checks['body_rank_six_all_states'] and checks['measurement_rank_eight_all_states']
            and checks['derivative_checks_pass'] and all(targets_met) and not any(caps)
            and all(r['scaled_wrench_ratio'] >= threshold for r in ratios.values()))
        # Post hoc, not preregistered: express the saved A-to-C/D state change in the local H directions at A.
        saved_changes = {}
        for label in 'CD':
            change = data['finite_A_to_saved_states'][label]
            dq, actual = np.array(change['delta_q']), np.array(change['actual_delta_wrench_scaled'])
            coefficients = V.T @ dq
            contributions = J @ V * coefficients[None, :]
            linear = J @ dq
            single = contributions[:, most]
            saved_changes[label] = {
                'delta_q_norm': change['delta_q_norm'],
                'actual_scaled_wrench_change_norm': float(np.linalg.norm(actual)),
                'actual_force_change_mN': float(1000 * F0 * np.linalg.norm(actual[3:])),
                'linear_at_A_relative_error': change['linear_at_A_relative_error'],
                'coefficients_along_A_directions': coefficients,
                'linearized_whitened_residual_per_direction': singular * coefficients,
                'share_of_linear_prediction_per_direction': contributions.T @ linear / (linear @ linear),
                'largest_gain_direction_share_of_delta_q_squared_norm': float(coefficients[most]**2 / (dq @ dq)),
                'largest_gain_direction_share_of_linear_prediction': float(single @ linear / (linear @ linear)),
                'largest_gain_direction_only_relative_error_vs_actual': float(np.linalg.norm(single - actual) / np.linalg.norm(actual)),
                'largest_gain_direction_only_cosine_vs_actual': float(single @ actual / (np.linalg.norm(single) * np.linalg.norm(actual)))}
        metrics = {row['method'][0]: row for row in fit['metrics']}
        weakest = np.array(a['body_scaled_weakest_direction'])
        gt = np.array(settings['gt_world_base_moment_force'])
        summary[str(case)] = {
            'inputs': {'fit_results_sha256': digest(fit_path), 'sensitivity_results_sha256': digest(sensitivity_path),
                       'diagnostic_script_sha256': data['diagnostic_script_sha256']},
            'tau_N': settings['tau_N'], 'gt_force_norm_N': float(np.linalg.norm(gt[3:])),
            'actual_saved_noise_whitened_norm': settings['actual_saved_noise_whitened_norm'],
            'fit_metrics': {label: {k: row[k] for k in ('shape_rms_all100_mm', 'force_error_N', 'force_direction_deg', 'force_error_over_F0')}
                            for label, row in metrics.items()},
            'per_state': {label: {'H_raw_condition': c['H_raw_condition'], 'H_scaled_condition': c['H_scaled_condition'],
                    'wrench_scaled_condition_common_world_base': c['wrench_scaled_condition_common_world_base'],
                    'largest_gain_among_H_directions': float(max(c['directional_gains']['per_noise_gain'])),
                    'largest_gain_direction_1_based': int(np.argmax(c['directional_gains']['per_noise_gain'])) + 1,
                    'J_fd_relative_error': c['J_fd_relative_error'],
                    'drop_residual_term_relative_error': c['drop_residual_term_relative_error']}
                for label, c in data['cases'].items()},
            'A_per_noise_gains': per_noise, 'A_H_singular_values': singular,
            'A_weakest_body_direction_force_fraction': float(np.sum(weakest[3:]**2)),
            'A_weakest_body_direction_body_Fz_fraction': float(weakest[5]**2),
            'checks': checks, 'all_targets_achieved': bool(all(targets_met)), 'any_curvature_cap_active': bool(any(caps)),
            'largest_gain_direction_1_based_at_A': most + 1, 'ratios_by_sign': ratios,
            'preregistered_recurrence_rule_met': recurrence,
            'post_hoc_saved_state_decomposition': saved_changes}
        case_rows.append({'case_id': case, 'gt_force_norm_N': summary[str(case)]['gt_force_norm_N'],
            'C_force_error_N': metrics['C']['force_error_N'], 'C_force_direction_deg': metrics['C']['force_direction_deg'],
            'D_force_error_N': metrics['D']['force_error_N'], 'D_force_direction_deg': metrics['D']['force_direction_deg'],
            'D_shape_rms_all100_mm': metrics['D']['shape_rms_all100_mm'],
            'A_H_scaled_condition': a['H_scaled_condition'], 'A_largest_gain': float(per_noise[most]),
            'A_largest_gain_direction_1_based': most + 1,
            'scaled_wrench_ratio_min': min(r['scaled_wrench_ratio'] for r in ratios.values()),
            'force_ratio_min': min(r['force_ratio'] for r in ratios.values()),
            'full_derivative_max_relative_error': checks['full_wrench_derivative_max_relative_error'],
            'recurrence_rule_met': recurrence,
            'D_largest_gain_direction_share_of_linear_prediction': saved_changes['D']['largest_gain_direction_share_of_linear_prediction'],
            'D_largest_gain_direction_only_cosine_vs_actual': saved_changes['D']['largest_gain_direction_only_cosine_vs_actual']})

    # The case-selecting driver must reproduce the committed case-0 Step 2 numbers.
    step2, parity = load(BASE / 'step2_sensitivity_20261012/results/results.json'), load(inputs[0][1])
    parity_difference = max(max_relative_difference(parity['cases'], step2['cases']),
        max_relative_difference(parity['finite_A_to_saved_states'], step2['finite_A_to_saved_states']),
        max(max_relative_difference(a, b) for a, b in zip(parity['controlled_perturbations'], step2['controlled_perturbations'])))
    result = {'scope': 'Two preselected additional cases plus the case-0 baseline; saved results only; no population claim.',
        'protocol_sha256': digest(STEP / 'protocol.json'), 'descriptive_recurrence_threshold': threshold,
        'new_cases_meeting_recurrence_rule': [c for c in protocol['selected_case_ids'] if summary[str(c)]['preregistered_recurrence_rule_met']],
        'case_0_parity_with_committed_step2_max_relative_difference': parity_difference,
        'noise_dependence': protocol['noise_dependence'], 'cases': summary}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'confirmation_summary.json').write_text(json.dumps(result, indent=2, default=lambda v: v.tolist()) + '\n', encoding='utf-8', newline='\n')
    for name, rows in (('cross_case_summary.csv', case_rows), ('perturbations.csv', perturbation_rows)):
        with (out / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'output': str(out), 'new_cases_meeting_recurrence_rule': result['new_cases_meeting_recurrence_rule'],
        'case_0_parity_max_relative_difference': parity_difference}), flush=True)

if __name__ == '__main__':
    main()
