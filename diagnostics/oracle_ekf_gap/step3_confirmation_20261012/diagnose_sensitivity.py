"""Same-case local orientation-to-wrench sensitivity; no estimator or data edits."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import scipy
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation
from sensitivity_math import modal_scale, least_squares_derivative, directional_gains

ORDER, GAMMA = (2, 2, 1), 10
SOURCE_COMMIT = '9a994c4dde921c07150072fa55c8bf3f19d0364f'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def json_default(value):
    return value.tolist() if isinstance(value, np.ndarray) else value.item()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output-dir', type=Path, required=True, help='Fresh directory; existing outputs are never replaced.')
    parser.add_argument('--state-results', type=Path, help='Saved A/B/C/D results.json for a selected case; default is the preserved case-0 Step 1 result.')
    args = parser.parse_args()
    if not sys.dont_write_bytecode:
        parser.error('Run with python -B to preserve tracked bytecode.')
    root, out = args.repo_root.resolve(), args.output_dir.resolve()
    if out.exists():
        parser.error(f'Output directory must be new: {out}')
    crt = root / 'src/shape_force_est_imu/crt'
    sys.path.insert(0, str(crt))
    sys.path.insert(0, str(root / 'diagnostics/oracle_ekf_gap/recovered_run'))
    import compare_oracle_vs_ekf_wrench_estimation as pipeline
    import virtual_work as vw
    from check_jacobian import capture_closures
    assert Path(pipeline.__file__).resolve() == crt / 'compare_oracle_vs_ekf_wrench_estimation.py'
    saved_path = (args.state_results.resolve() if args.state_results else root / 'diagnostics/oracle_ekf_gap/step1_frechet_20261008/after_abcd/results.json')
    states_path = saved_path.parent / 'original_states.npz'
    saved = json.loads(saved_path.read_text(encoding='utf-8'))
    assert saved['order'] == list(ORDER)
    selected_case_id, selected_noise = int(saved['case_id']), int(saved['noise_realization'])
    assert saved['gamma'] == GAMMA and saved['steps'] == 30 and saved['production_derivative_implementation'] == 'scipy.linalg.expm_frechet'
    source_checks = {}
    for name, expected in saved['hashes'].items():
        path = crt / name
        if digest(path) == expected:
            source_checks[name] = 'raw SHA256 match'
        elif path.suffix == '.py':
            reference = subprocess.check_output(['git', '-C', str(root), 'show', SOURCE_COMMIT + ':' + path.relative_to(root).as_posix()])
            normalized = reference.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')
            assert path.read_text(encoding='utf-8') == normalized
            candidates = [normalized.encode(), normalized.replace('\n', '\r\n').encode()]
            assert expected in [hashlib.sha256(data).hexdigest() for data in candidates]
            source_checks[name] = 'identical source; only Windows line endings differ'
        else:
            raise AssertionError(f'Saved input changed: {path}')
    states = {row['method'][0]: np.array(row['state']) for row in saved['metrics']}
    assert set(states) == set('ABCD') and all(m.shape == (8,) and np.isfinite(m).all() for m in states.values())
    with np.load(states_path) as cache:
        np.testing.assert_array_equal(cache['oracle'], states['A'])
        np.testing.assert_array_equal(cache['ekf'], states['D'])
    gt_path, imu_path = crt / 'gt_data/kirchhoff_gt_dataset.npz', crt / 'gt_data/kirchhoff_imu_3imu.npz'
    with np.load(gt_path, allow_pickle=False) as data:
        gt_matches = np.flatnonzero(data['case_id'] == selected_case_id)
        assert len(gt_matches) == 1, 'Saved case ID must occur exactly once in GT'
        gt_index = int(gt_matches[0])
        assert gt_index == saved.get('gt_case_index', saved['case_index'])
        positions, tau, force, moment = [data[name][gt_index].copy() for name in ('positions', 'tau', 'f_ext', 'l_ext')]
        gt_meta = json.loads(data['meta'].item())
    with np.load(imu_path, allow_pickle=False) as data:
        imu_matches = np.flatnonzero(data['case_id'] == selected_case_id)
        assert len(imu_matches) == 1, 'Saved case ID must occur exactly once in IMU data'
        imu_index = int(imu_matches[0])
        assert imu_index == saved.get('imu_case_index', gt_index)
        assert 0 <= selected_noise < data['R_meas'].shape[1]
        noisy, true = data['R_meas'][imu_index, selected_noise].copy(), data['R_true'][imu_index].copy()
        sensor_s = data['imu_arc_indices'].copy() / (len(positions) - 1)
        imu_meta = json.loads(data['meta'].item())
    np.testing.assert_allclose(sensor_s, saved['actual_s_used'], rtol=0, atol=1e-15)
    sigma = np.deg2rad(imu_meta['meas_std_deg'])
    assert imu_meta['meas_std_deg'] == saved['sigma_deg'] == 0.5
    L, EI = pipeline._L, pipeline._EIX
    assert L == gt_meta['length_m'] == 0.1
    M0, F0 = EI / L, EI / L**2
    scales = np.array([M0] * 3 + [F0] * 3)
    Q, S = modal_scale(vw, ORDER)
    np.testing.assert_allclose(S.T @ Q @ S, np.eye(8), atol=1e-12)
    protected = [crt / name for name in saved['hashes']] + [saved_path, states_path]
    def protected_key(path):
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()
    protected_paths = {protected_key(p): p for p in protected}
    hashes_before = {name: digest(path) for name, path in protected_paths.items()}
    git_args = ['git', '-C', str(root), '-c', 'core.fsmonitor=false']
    status_args = git_args + ['status', '--porcelain=v1', '--', '.']
    try:
        status_args.append(':(exclude,literal)' + out.relative_to(root).as_posix())
    except ValueError:
        pass
    status_before = subprocess.check_output(status_args, text=True)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    closure = capture_closures(pipeline, noisy, sensor_s)['H_and_r_general']

    def observation_residual(m):
        return np.concatenate([Rotation.from_matrix(R @ pipeline.fwd_transform_general(
            m, float(s), GAMMA, L, *ORDER)[:3, :3].T).as_rotvec() for R, s in zip(noisy, sensor_s)])

    def terms(m):
        grad = vw.elastic_energy_gradient(m, EI, pipeline._EIY, pipeline._GJ, L, *ORDER)
        pull = vw.pull_jacobian(m, pipeline._R_LIST, L, *ORDER)
        J, T = vw.body_jacobian_at_s(m, 1., GAMMA, L, *ORDER)
        b = vw.generalized_modal_load(grad, pull, tau)
        Ab = J.T
        body_singular = np.linalg.svd(Ab, compute_uv=False)
        assert np.count_nonzero(body_singular > body_singular[0] * 1e-8) == 6
        wb = vw.solve_wrench(J, pull, grad, tau, rcond=1e-8)
        R, p = T[:3, :3], T[:3, 3]
        D = np.block([[R, vw.hat(p) @ R], [np.zeros((3, 3)), R]])
        C = np.block([[R.T, -R.T @ vw.hat(p)], [np.zeros((3, 3)), R.T]])
        np.testing.assert_allclose(C @ D, np.eye(6), atol=1e-12)
        A, w = Ab @ C, D @ wb
        B, y, x = A * scales[None, :] / M0, b / M0, w / scales
        np.testing.assert_allclose(A @ w, Ab @ wb, atol=1e-12)
        # Descriptive column scaling and uniform row scaling preserve the LS objective.
        np.testing.assert_allclose(np.linalg.lstsq(B, y, rcond=1e-8)[0], x, rtol=1e-8, atol=1e-8)
        body_scaled = Ab * scales[None, :] / M0
        _, bs, bvt = np.linalg.svd(body_scaled, full_matrices=False)
        weakest = bvt[-1].copy()
        if weakest[np.argmax(np.abs(weakest))] < 0:
            weakest *= -1
        return {'A': A, 'b': b, 'w': w, 'B': B, 'y': y, 'x': x, 'T': T,
                'r': b - A @ w, 'body_singular': body_singular,
                'body_scaled_singular': bs, 'body_scaled_weakest': weakest}

    def derivatives(m, epsilon):
        plus, minus = [terms(m + epsilon * S[:, k]) for k in range(8)], [terms(m - epsilon * S[:, k]) for k in range(8)]
        differences = {key: np.stack([(a[key] - b[key]) / (2 * epsilon) for a, b in zip(plus, minus)], axis=-1)
                       for key in ('B', 'y', 'x')}
        return differences

    def relative(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-15))

    cases, nominal = {}, {}
    for label, m in states.items():
        current = terms(m)
        nominal[label] = current
        f, l = pipeline.wrench_from_modal(m, tau, *ORDER, gamma=GAMMA)
        independent = np.r_[l + np.cross(current['T'][:3, 3], f), f]
        np.testing.assert_allclose(current['w'], independent, rtol=1e-10, atol=1e-10)
        saved_row = next(row for row in saved['metrics'] if row['method'][0] == label)
        np.testing.assert_allclose(current['w'], saved_row['world_base_wrench'], rtol=1e-10, atol=1e-10)
        H, residual = closure(m)
        Hq = H @ S / sigma
        h_fd = np.column_stack([(observation_residual(m + 1e-6 * S[:, k]) - observation_residual(m - 1e-6 * S[:, k])) / (2e-6 * sigma) for k in range(8)])
        assert relative(Hq, h_fd) < 1e-7
        d = derivatives(m, 1e-5)
        Jx, components = least_squares_derivative(current['B'], current['y'], current['x'], d['B'], d['y'])
        derivative_error = relative(Jx, d['x'])
        assert derivative_error < 3e-3, (label, derivative_error)
        gains = directional_gains(Hq, Jx)
        cases[label] = {'state': m, 'H_white': Hq, 'H_raw_condition': float(np.linalg.cond(H)),
            'H_scaled_condition': float(np.linalg.cond(Hq)), 'H_fd_relative_error': relative(Hq, h_fd),
            'wrench_scaled_condition_common_world_base': float(np.linalg.cond(current['B'])),
            'body_original_singular_values': current['body_singular'], 'body_original_pinv_rcond': 1e-8,
            'body_original_pinv_cutoff': float(current['body_singular'][0] * 1e-8), 'body_retained_rank': 6,
            'body_scaled_singular_values': current['body_scaled_singular'],
            'body_scaled_weakest_direction': current['body_scaled_weakest'],
            'world_base_wrench': current['w'], 'generalized_residual_Nm': current['r'],
            'measurement_residual_whitened_norm': float(np.linalg.norm(residual) / sigma),
            'J_wrench_scaled_q': Jx, 'J_actual_solver_fd': d['x'], 'J_fd_relative_error': derivative_error,
            'least_squares_derivative_components': components, 'directional_gains': gains,
            'drop_residual_term_relative_error': relative(components['db_term'] + components['dA_term'], d['x']),
            'db_only_relative_error': relative(components['db_term'], d['x'])}
        print(f'{label}: H scaled condition={cases[label]["H_scaled_condition"]:.3g}; wrench condition={np.linalg.cond(current["B"]):.3g}; full derivative FD error={derivative_error:.3g}', flush=True)
    # One smaller finite-difference step at A checks numerical convergence without a sweep.
    half = derivatives(states['A'], 5e-6)
    convergence = relative(half['x'], cases['A']['J_actual_solver_fd'])
    assert convergence < 3e-3, convergence

    finite_changes = {}
    for label in 'BCD':
        a, b = nominal['A'], nominal[label]
        db, dAw, dr = b['b'] - a['b'], (b['A'] - a['A']) @ a['w'], b['r'] - a['r']
        left = b['A'] @ (b['w'] - a['w'])
        np.testing.assert_allclose(left, db - dAw - dr, rtol=1e-10, atol=1e-12)
        dq = np.linalg.solve(S, states[label] - states['A'])
        finite_changes[label] = {'delta_q': dq, 'delta_q_norm': float(np.linalg.norm(dq)),
            'delta_b_q_scaled': S.T @ db / M0, 'delta_A_times_wA_q_scaled': S.T @ dAw / M0,
            'delta_residual_q_scaled': S.T @ dr / M0, 'lhs_q_scaled': S.T @ left / M0,
            'identity_error': float(np.linalg.norm(left - db + dAw + dr)),
            'actual_delta_wrench_scaled': (b['w'] - a['w']) / scales,
            'linear_at_A_delta_wrench_scaled': cases['A']['J_wrench_scaled_q'] @ dq,
            'linear_at_A_relative_error': relative(cases['A']['J_wrench_scaled_q'] @ dq, (b['w'] - a['w']) / scales),
            'row_scaling_note': 'S.T used for work-covector display only; estimator weighting is unchanged.'}

    gains = cases['A']['directional_gains']
    most = int(np.argmax(gains['per_noise_gain']))
    selected = sorted(set([0, 7, most]))
    perturbations = []
    checks = [(index, 0.1) for index in selected] + [(most, 0.01)]
    base = nominal['A']
    grid = np.linspace(0, 1, len(positions))
    base_curve = np.array([pipeline.fwd_transform_general(states['A'], float(s), GAMMA, L, *ORDER)[:3, 3] for s in grid])
    for index, target in checks:
        vector = gains['V'][:, index]
        h = min(target / gains['singular_values'][index], 0.05)
        for sign in (-1, 1):
            dq = sign * h * vector
            m = states['A'] + S @ dq
            perturbed = terms(m)
            predicted = cases['A']['J_wrench_scaled_q'] @ dq
            actual = perturbed['x'] - base['x']
            curves = np.array([pipeline.fwd_transform_general(m, float(s), GAMMA, L, *ORDER)[:3, 3] for s in grid])
            perturbations.append({'direction_index': index, 'target_stacked_whitened_norm': target,
                'sign': sign, 'delta_q_norm': h, 'curvature_cap_active': bool(h == 0.05), 'delta_state': S @ dq,
                'actual_measurement_change_whitened_norm': float(np.linalg.norm(observation_residual(m) - observation_residual(states['A'])) / sigma),
                'stacked_measurement_change_deg': float(np.degrees(np.linalg.norm(observation_residual(m) - observation_residual(states['A'])))),
                'actual_delta_wrench_scaled': actual, 'predicted_delta_wrench_scaled': predicted,
                'relative_linear_prediction_error': relative(predicted, actual),
                'force_change_N': float(np.linalg.norm(perturbed['w'][3:] - base['w'][3:])),
                'moment_change_common_base_Nm': float(np.linalg.norm(perturbed['w'][:3] - base['w'][:3])),
                'shape_change_rms_mm': float(1000 * np.sqrt(np.mean(np.sum((curves - base_curve)**2, axis=1))))})

    actual_noise = np.concatenate([Rotation.from_matrix(a @ b.T).as_rotvec() for a, b in zip(noisy, true)])
    assert all(digest(protected_paths[name]) == value for name, value in hashes_before.items())
    assert subprocess.check_output(status_args, text=True) == status_before
    result = {'scope': 'Local sensitivity on one saved physical case; fixed tensions; no refitting or new estimator.',
        'head': subprocess.check_output(git_args + ['rev-parse', 'HEAD'], text=True).strip(),
        'branch': subprocess.check_output(git_args + ['branch', '--show-current'], text=True).strip(),
        'command_argv': [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:]],
        'python': sys.executable, 'numpy': np.__version__, 'scipy': scipy.__version__,
        'saved_state_results_path': str(saved_path), 'diagnostic_script_sha256': digest(Path(__file__).resolve()),
        'source_checks': source_checks, 'protected_hashes': hashes_before, 'inputs_sources_and_status_unchanged': True,
        'settings': {'case': selected_case_id, 'gt_case_index': gt_index, 'imu_case_index': imu_index, 'noise': selected_noise, 'order': ORDER, 'gamma': GAMMA, 'sensor_s': sensor_s,
            'sigma_rad': sigma, 'actual_saved_noise_whitened_norm': float(np.linalg.norm(actual_noise) / sigma),
            'L_m': L, 'EI_Nm2': EI, 'M0_Nm': M0, 'F0_N': F0, 'tau_N': tau,
            'gt_world_base_moment_force': np.r_[moment + np.cross(positions[-1], force), force],
            'modal_Gram': Q, 'modal_scale_m_equals_S_q': S,
            'modal_metric': 'Integral of squared normalized curvature/torsion change; not prior/energy weighting.',
            'wrench_frame_origin': 'World axes and fixed world/base origin; [moment, force].',
            'weighting_preservation': 'Original body pinv rcond1e-8 throughout. Column scaling and uniform row scale only for derivative; no modal-row reweighting.',
            'observation_budget': 'One nine-component triplet. Saved D repeats same frame30 times; no independent averaging assumed.'},
        'cases': cases, 'A_derivative_half_step_relative_change': convergence,
        'finite_A_to_saved_states': finite_changes, 'controlled_perturbations': perturbations,
        'elapsed_seconds': time.perf_counter() - start,
        'limitations': 'Local constant-rank sensitivity; not global/dynamic observability, EKF posterior covariance, or manuscript aggregate causality.'}
    (out / 'results.json').write_text(json.dumps(result, indent=2, default=json_default) + '\n', encoding='utf-8', newline='\n')
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['method', 'H_scaled_condition', 'wrench_scaled_condition_common_world_base', 'J_fd_relative_error', 'drop_residual_term_relative_error', 'max_wrench_gain_per_whitened_noise']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for label, data in cases.items():
            writer.writerow({'method': label, **{k: data[k] for k in fields[1:-1]}, 'max_wrench_gain_per_whitened_noise': float(np.max(data['directional_gains']['per_noise_gain']))})
    print(json.dumps({'output': str(out), 'H_condition_A': cases['A']['H_scaled_condition'],
        'maximum_per_noise_gain_A': float(np.max(gains['per_noise_gain'])), 'perturbation_count': len(perturbations),
        'inputs_sources_unchanged': True, 'elapsed_seconds': result['elapsed_seconds']}), flush=True)

if __name__ == '__main__':
    main()
