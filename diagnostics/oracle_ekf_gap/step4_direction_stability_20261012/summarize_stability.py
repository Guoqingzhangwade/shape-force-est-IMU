"""Summarize amplified-direction stability over saved case/noise runs; run no simulations.

Examples:
    python -B summarize_stability.py --output-dir "$env:TEMP/step4-summary-new"
    python -B summarize_stability.py --runs-dir <full run outputs> --output-dir <new dir>

Without --runs-dir the committed compact extraction results/runs_compact.json is
used. With --runs-dir the compact extraction is rebuilt from full runner outputs
(folders case_XX_noise_N/{abcd,sensitivity}/results.json). The GT dataset is read
only for the true tip rotation. Existing outputs are never overwritten.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

STEP = Path(__file__).resolve().parent
ROOT = STEP.parents[2]
AXES = ('bending_x', 'bending_y', 'torsion_z')
BLOCKS = (slice(0, 3), slice(3, 6), slice(6, 8))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def unit(vector):
    return vector / np.linalg.norm(vector)

def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def amplified_direction(H, T):
    """Maximizer of ||J v|| / ||H v||: pinv(H) applied to the top right singular vector of T = J pinv(H)."""
    _, singular, Vt = np.linalg.svd(T)
    v = unit(np.linalg.pinv(H, rcond=1e-15) @ Vt[0])
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return v, float(singular[0]), float(singular[1])

def extract(runs_dir, completion):
    with np.load(ROOT / 'src/shape_force_est_imu/crt/gt_data/kirchhoff_gt_dataset.npz', allow_pickle=False) as data:
        tip_rotation = {int(c): data['T_tip'][i][:3, :3].copy() for i, c in enumerate(data['case_id'])}
    rows = []
    for record in completion['runs']:
        case, noise = record['case_id'], record['noise_realization']
        folder = runs_dir / f'case_{case:02d}_noise_{noise}'
        row = dict(record)
        row['valid'] = record.get('abcd_returncode') == 0 and record.get('sensitivity_returncode') == 0
        if not row['valid']:
            log = folder / ('sensitivity.log' if record.get('abcd_returncode') == 0 else 'abcd.log')
            row['failure_log_tail'] = log.read_text(encoding='utf-8').strip().splitlines()[-3:]
            rows.append(row)
            continue
        fit, data = load(folder / 'abcd/results.json'), load(folder / 'sensitivity/results.json')
        settings, a = data['settings'], data['cases']['A']
        assert settings['case'] == fit['case_id'] == case and settings['noise'] == fit['noise_realization'] == noise
        assert data['inputs_sources_and_status_unchanged']
        gains = a['directional_gains']
        H, J, T = np.array(a['H_white']), np.array(a['J_wrench_scaled_q']), np.array(gains['total_response_operator'])
        V, singular, per_noise = np.array(gains['V']), np.array(gains['singular_values']), np.array(gains['per_noise_gain'])
        v, top_gain, second_gain = amplified_direction(H, T)
        np.testing.assert_allclose(np.linalg.norm(J @ v) / np.linalg.norm(H @ v), top_gain, rtol=1e-8)
        response = J @ v
        body_force = tip_rotation[case].T @ response[3:]
        most = int(np.argmax(per_noise))
        metrics = {m['method'][0]: m for m in fit['metrics']}
        row.update({
            'fit_results_sha256': digest(folder / 'abcd/results.json'),
            'sensitivity_results_sha256': digest(folder / 'sensitivity/results.json'),
            'head': data['head'], 'diagnostic_script_sha256': data['diagnostic_script_sha256'],
            'tau_N': settings['tau_N'], 'gt_world_base_moment_force': settings['gt_world_base_moment_force'],
            'actual_saved_noise_whitened_norm': settings['actual_saved_noise_whitened_norm'],
            'metrics': {label: {k: m[k] for k in ('shape_rms_all100_mm', 'force_error_N', 'force_direction_deg', 'force_error_over_F0')}
                        for label, m in metrics.items()},
            'H_scaled_condition_A': a['H_scaled_condition'], 'H_singular_values_A': singular,
            'wrench_scaled_condition_A': a['wrench_scaled_condition_common_world_base'],
            'largest_gain_among_H_directions_A': float(per_noise[most]), 'largest_gain_H_direction_1_based': most + 1,
            'H_fd_max_relative_error': max(c['H_fd_relative_error'] for c in data['cases'].values()),
            'J_fd_max_relative_error': max(c['J_fd_relative_error'] for c in data['cases'].values()),
            'state_A_q': np.linalg.solve(np.array(settings['modal_scale_m_equals_S_q']), np.array(a['state'])),
            'amplified_direction_q': v, 'amplified_direction_gain': top_gain, 'second_response_gain': second_gain,
            'amplified_direction_whitened_residual_per_unit_q': float(np.linalg.norm(H @ v)),
            'amplified_response_scaled_wrench_unit': unit(response),
            'amplified_response_force_fraction': float(np.sum(response[3:]**2) / np.sum(response**2)),
            'amplified_response_force_body_tip_unit': unit(body_force),
            'saved_changes': {}})
        for label in 'CD':
            change = data['finite_A_to_saved_states'][label]
            dq, actual = np.array(change['delta_q']), np.array(change['actual_delta_wrench_scaled'])
            # Component of the state change along v*, orthogonal in the whitened measurement metric.
            alpha = float((H @ v) @ (H @ dq) / ((H @ v) @ (H @ v)))
            coefficients = V.T @ dq
            linear, single = J @ dq, J @ V[:, most] * coefficients[most]
            row['saved_changes'][label] = {
                'delta_q': dq, 'delta_q_norm': change['delta_q_norm'], 'actual_delta_wrench_scaled': actual,
                'linear_at_A_relative_error': change['linear_at_A_relative_error'],
                'amplified_component_alpha': alpha,
                'amplified_component_whitened_residual': float(alpha * np.linalg.norm(H @ v)),
                'state_change_whitened_residual_norm': float(np.linalg.norm(H @ dq)),
                'cosine_actual_vs_amplified_response': cosine(actual, response),
                'amplified_component_only_relative_error': float(np.linalg.norm(alpha * response - actual) / np.linalg.norm(actual)),
                'largest_gain_H_direction_share_of_linear_prediction': float(single @ linear / (linear @ linear)),
                'largest_gain_H_direction_only_cosine_vs_actual': cosine(single, actual)}
        rows.append(row)
    return rows

def distribution(values):
    values = np.asarray(values, dtype=float)
    q = np.percentile(values, [0, 10, 25, 50, 75, 90, 100])
    return {'count': int(values.size), 'mean': float(values.mean()), 'min': q[0], 'p10': q[1], 'p25': q[2],
            'median': q[3], 'p75': q[4], 'p90': q[5], 'max': q[6]}

def summarize(rows, protocol):
    valid = [r for r in rows if r['valid']]
    for r in valid:
        r['amplified_direction_q'] = np.array(r['amplified_direction_q'], dtype=float)
    base = {r['case_id']: r for r in valid if r['noise_realization'] == 0}
    stack = np.array([r['amplified_direction_q'] for r in base.values()])
    eigenvalues, eigenvectors = np.linalg.eigh(stack.T @ stack / len(stack))
    reference = eigenvectors[:, -1]
    if reference[np.argmax(np.abs(reference))] < 0:
        reference = -reference
    for r in valid:
        r['abs_cos_to_reference'] = abs(float(r['amplified_direction_q'] @ reference))
        if r['case_id'] in base:
            r['abs_cos_to_own_noise0'] = abs(float(r['amplified_direction_q'] @ base[r['case_id']]['amplified_direction_q']))
    case_cos = np.array([r['abs_cos_to_reference'] for r in base.values()])
    noise_min = {c: min([r['abs_cos_to_own_noise0'] for r in valid if r['case_id'] == c and r['noise_realization'] != 0], default=None)
                 for c in sorted(base)}

    def block_shares(vectors):
        vectors = np.atleast_2d(vectors)
        return {axis: float(np.mean(np.sum(vectors[:, block]**2, axis=1))) for axis, block in zip(AXES, BLOCKS)}

    def group(selection):
        out = {'runs': len(selection)}
        for label in 'ABCD':
            out[label] = {k: distribution([r['metrics'][label][k] for r in selection]) for k in ('force_error_N', 'force_direction_deg', 'shape_rms_all100_mm')}
        for label in 'CD':
            cos = np.abs([r['saved_changes'][label]['cosine_actual_vs_amplified_response'] for r in selection])
            out[label]['abs_cosine_actual_vs_amplified_response'] = distribution(cos)
            out[label]['fraction_abs_cosine_at_least_0p9'] = float(np.mean(cos >= 0.9))
            for key in ('amplified_component_only_relative_error', 'linear_at_A_relative_error', 'largest_gain_H_direction_share_of_linear_prediction'):
                out[label][key] = distribution([r['saved_changes'][label][key] for r in selection])
            out[label]['abs_amplified_component_whitened_residual'] = distribution(np.abs([r['saved_changes'][label]['amplified_component_whitened_residual'] for r in selection]))
        return out

    body = np.array([r['amplified_response_force_body_tip_unit'] for r in valid])
    d_alpha = np.abs([r['saved_changes']['D']['amplified_component_whitened_residual'] for r in valid])
    d_error = np.array([r['metrics']['D']['force_error_over_F0'] for r in valid])
    d_gain = np.array([r['amplified_direction_gain'] for r in valid])
    case_fraction = float(np.mean(case_cos >= 0.9))
    # Post hoc, not preregistered: modal content of v*, its bending plane, and gain versus the rod's own bending.
    plane, bending = [], []
    for r in base.values():
        q, v = np.array(r['state_A_q']), r['amplified_direction_q']
        plane.append(abs(cosine(v[[2, 5]], q[[0, 3]])))
        bending.append(float(np.linalg.norm(q[[0, 3]])))
    base_gain = np.array([r['amplified_direction_gain'] for r in base.values()])
    slope, intercept = np.polyfit(np.log(bending), np.log(base_gain), 1)
    aligned = np.abs([r['saved_changes']['D']['cosine_actual_vs_amplified_response'] for r in valid]) >= 0.9
    d_newton = np.array([r['metrics']['D']['force_error_N'] for r in valid])
    gt_force = np.array([np.linalg.norm(r['gt_world_base_moment_force'][3:]) for r in valid])
    post_hoc = {
        'note': 'Added after seeing the results; not covered by protocol.json.',
        'quadratic_bending_mode_share_of_direction': distribution(np.sum(stack[:, [2, 5]]**2, axis=1)),
        'linear_bending_mode_share_of_direction': distribution(np.sum(stack[:, [1, 4]]**2, axis=1)),
        'share_in_top_two_eigenvector_subspace': distribution(np.sum((stack @ eigenvectors[:, -2:])**2, axis=1)),
        'abs_cos_direction_plane_vs_mean_bending_plane': distribution(plane),
        'mean_bending_q_norm': distribution(bending),
        'log_gain_vs_log_mean_bending': {'slope': float(slope), 'gain_at_unit_bending': float(np.exp(intercept)),
            'pearson_r': float(np.corrcoef(np.log(bending), np.log(base_gain))[0, 1])},
        'gain_times_mean_bending': distribution(base_gain * np.array(bending)),
        'D_force_error_N_median_when_aligned': float(np.median(d_newton[aligned])),
        'D_force_error_N_median_when_not_aligned': float(np.median(d_newton[~aligned])),
        'runs_with_D_force_error_above_0p2_N': int(np.sum(d_newton > 0.2)),
        'aligned_fraction_among_those': float(np.mean(aligned[d_newton > 0.2])),
        'gt_force_norm_N': distribution(gt_force),
        'D_force_error_over_gt_force_norm': distribution(d_newton / gt_force)}

    return {
        'post_hoc': post_hoc,
        'scope': 'All saved cases and noise realizations; local analysis at A; five shared noise patterns, not independent trials.',
        'run_count': len(rows), 'valid_run_count': len(valid),
        'failed_runs': [{k: r[k] for k in r if k != 'valid'} for r in rows if not r['valid']],
        'reference_direction_q': reference, 'reference_axis_shares': block_shares(reference),
        'mean_projector_eigenvalues_descending': eigenvalues[::-1],
        'per_case_direction_axis_shares_mean': block_shares(stack),
        'abs_cos_to_reference_noise0': distribution(case_cos),
        'fraction_cases_abs_cos_at_least_0p9': case_fraction,
        'case_stability_rule_met': bool(case_fraction >= 0.9),
        'abs_cos_to_reference_by_case_noise0': {str(c): r['abs_cos_to_reference'] for c, r in sorted(base.items())},
        'min_abs_cos_to_own_noise0_over_noise_1_to_4': distribution([v for v in noise_min.values() if v is not None]),
        'amplified_direction_gain': distribution(d_gain),
        'second_response_gain': distribution([r['second_response_gain'] for r in valid]),
        'largest_gain_among_H_directions_A': distribution([r['largest_gain_among_H_directions_A'] for r in valid]),
        'H_scaled_condition_A': distribution([r['H_scaled_condition_A'] for r in valid]),
        'wrench_scaled_condition_A': distribution([r['wrench_scaled_condition_A'] for r in valid]),
        'amplified_response_force_fraction': distribution([r['amplified_response_force_fraction'] for r in valid]),
        'amplified_response_force_body_tip_squared_components_mean': {axis: float(np.mean(body[:, i]**2)) for i, axis in enumerate('xyz')},
        'amplified_response_force_body_tip_abs_z': distribution(np.abs(body[:, 2])),
        'H_fd_max_relative_error': max(r['H_fd_max_relative_error'] for r in valid),
        'J_fd_max_relative_error': max(r['J_fd_max_relative_error'] for r in valid),
        'D_force_error_over_F0_vs_gain_times_abs_component': {
            'pearson_r': float(np.corrcoef(d_error, d_gain * d_alpha)[0, 1]),
            'median_ratio': float(np.median(d_error / (d_gain * d_alpha)))},
        'all_runs': group(valid),
        'by_noise_realization': {str(n): group([r for r in valid if r['noise_realization'] == n]) for n in protocol['noise_realizations']},
        'thresholds': 'Descriptive 0.9 cosine and 90% case fraction fixed in protocol.json before the runs.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir', type=Path, help='Full runner outputs; default uses the committed compact extraction.')
    parser.add_argument('--completion', type=Path, help='completion.json of the launch (default: beside this script).')
    parser.add_argument('--output-dir', type=Path, required=True, help='Fresh directory; existing outputs are never replaced.')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        parser.error(f'Output directory must be new: {out}')
    protocol = load(STEP / 'protocol.json')
    encode = lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item()
    if args.runs_dir:
        rows = extract(args.runs_dir.resolve(), load(args.completion or STEP / 'completion.json'))
        rows = json.loads(json.dumps(rows, default=encode))
    else:
        rows = load(STEP / 'results/runs_compact.json')
    assert len(rows) == protocol['run_count']
    summary = summarize([dict(r) for r in rows], protocol)
    summary['protocol_sha256'] = digest(STEP / 'protocol.json')
    out.mkdir(parents=True, exist_ok=False)
    if args.runs_dir:
        (out / 'runs_compact.json').write_text(json.dumps(rows, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=encode) + '\n', encoding='utf-8', newline='\n')
    reference = np.array(summary['reference_direction_q'])
    with (out / 'per_run.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['case_id', 'noise_realization', 'valid', 'gt_force_norm_N', 'C_force_error_N', 'C_force_direction_deg',
                  'D_force_error_N', 'D_force_direction_deg', 'D_shape_rms_all100_mm', 'H_scaled_condition_A',
                  'amplified_direction_gain', 'abs_cos_to_reference', 'D_abs_cosine_actual_vs_amplified_response',
                  'D_amplified_component_whitened_residual', 'D_amplified_component_only_relative_error', 'D_linear_at_A_relative_error']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for r in rows:
            line = {'case_id': r['case_id'], 'noise_realization': r['noise_realization'], 'valid': r['valid']}
            if r['valid']:
                d = r['saved_changes']['D']
                line.update({'gt_force_norm_N': float(np.linalg.norm(r['gt_world_base_moment_force'][3:])),
                    'C_force_error_N': r['metrics']['C']['force_error_N'], 'C_force_direction_deg': r['metrics']['C']['force_direction_deg'],
                    'D_force_error_N': r['metrics']['D']['force_error_N'], 'D_force_direction_deg': r['metrics']['D']['force_direction_deg'],
                    'D_shape_rms_all100_mm': r['metrics']['D']['shape_rms_all100_mm'], 'H_scaled_condition_A': r['H_scaled_condition_A'],
                    'amplified_direction_gain': r['amplified_direction_gain'],
                    'abs_cos_to_reference': abs(float(np.array(r['amplified_direction_q']) @ reference)),
                    'D_abs_cosine_actual_vs_amplified_response': abs(d['cosine_actual_vs_amplified_response']),
                    'D_amplified_component_whitened_residual': d['amplified_component_whitened_residual'],
                    'D_amplified_component_only_relative_error': d['amplified_component_only_relative_error'],
                    'D_linear_at_A_relative_error': d['linear_at_A_relative_error']})
            writer.writerow(line)
    print(json.dumps({'output': str(out), 'valid_runs': summary['valid_run_count'],
        'fraction_cases_abs_cos_at_least_0p9': summary['fraction_cases_abs_cos_at_least_0p9'],
        'D_fraction_error_along_direction': summary['all_runs']['D']['fraction_abs_cosine_at_least_0p9']}), flush=True)

if __name__ == '__main__':
    main()
