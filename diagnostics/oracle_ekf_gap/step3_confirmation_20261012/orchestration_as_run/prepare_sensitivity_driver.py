from pathlib import Path
import shutil

TASK = Path(__file__).resolve().parent
ROOT = Path(r'C:\Users\guoqi\OneDrive - stevens.edu\Documents\GitHub\shape-force-est-IMU')
REL = 'diagnostics/oracle_ekf_gap/step2_sensitivity_20261012'
candidate = TASK / 'candidate' / REL
candidate.mkdir(parents=True, exist_ok=True)
for name in ('diagnose_sensitivity.py', 'sensitivity_math.py', 'render_figures.py'):
    shutil.copy2(ROOT / REL / name, candidate / name)
path = candidate / 'diagnose_sensitivity.py'
source = path.read_text(encoding='utf-8')
replacements = [
    ("    args = parser.parse_args()", "    parser.add_argument('--state-results', type=Path, help='Saved A/B/C/D results.json for a selected case; default is the preserved case-0 Step 1 result.')\n    args = parser.parse_args()"),
    ("    saved_path = root / 'diagnostics/oracle_ekf_gap/step1_frechet_20261008/after_abcd/results.json'", "    saved_path = (args.state_results.resolve() if args.state_results else root / 'diagnostics/oracle_ekf_gap/step1_frechet_20261008/after_abcd/results.json')"),
    ("    assert saved['order'] == list(ORDER) and saved['case_index'] == saved['noise_realization'] == 0", "    assert saved['order'] == list(ORDER)\n    selected_case_id, selected_noise = int(saved['case_id']), int(saved['noise_realization'])"),
    ("        positions, tau, force, moment = [data[name][0].copy() for name in ('positions', 'tau', 'f_ext', 'l_ext')]", "        gt_matches = np.flatnonzero(data['case_id'] == selected_case_id)\n        assert len(gt_matches) == 1, 'Saved case ID must occur exactly once in GT'\n        gt_index = int(gt_matches[0])\n        assert gt_index == saved.get('gt_case_index', saved['case_index'])\n        positions, tau, force, moment = [data[name][gt_index].copy() for name in ('positions', 'tau', 'f_ext', 'l_ext')]"),
    ("        noisy, true = data['R_meas'][0, 0].copy(), data['R_true'][0].copy()", "        imu_matches = np.flatnonzero(data['case_id'] == selected_case_id)\n        assert len(imu_matches) == 1, 'Saved case ID must occur exactly once in IMU data'\n        imu_index = int(imu_matches[0])\n        assert imu_index == saved.get('imu_case_index', gt_index)\n        assert 0 <= selected_noise < data['R_meas'].shape[1]\n        noisy, true = data['R_meas'][imu_index, selected_noise].copy(), data['R_true'][imu_index].copy()"),
    ("        'settings': {'case': 0, 'noise': 0, 'order': ORDER", "        'settings': {'case': selected_case_id, 'gt_case_index': gt_index, 'imu_case_index': imu_index, 'noise': selected_noise, 'order': ORDER"),
    ("        'source_checks': source_checks, 'protected_hashes': hashes_before", "        'saved_state_results_path': str(saved_path), 'diagnostic_script_sha256': digest(Path(__file__).resolve()),\n        'source_checks': source_checks, 'protected_hashes': hashes_before"),
]
for old, new in replacements:
    assert source.count(old) == 1, old
    source = source.replace(old, new)
path.write_text(source, encoding='utf-8', newline='\n')
compile(source, str(path), 'exec')
figure = candidate / 'render_figures.py'
source = figure.read_text(encoding='utf-8')
for old, new in [
    ('"Case 0: local measurement and wrench sensitivity at A / B / C / D"', 'f"Case {data[\'settings\'][\'case\']}: local measurement and wrench sensitivity at A / B / C / D"'),
    ('"Case 0 at A: bounded positive and negative perturbations"', 'f"Case {data[\'settings\'][\'case\']} at A: bounded positive and negative perturbations"'),
]:
    assert source.count(old) == 1, old
    source = source.replace(old, new)
figure.write_text(source, encoding='utf-8', newline='\n')
compile(source, str(figure), 'exec')
print('Sensitivity case selection and figure labels prepared; numerical kernel unchanged.')
