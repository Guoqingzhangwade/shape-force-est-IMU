from pathlib import Path
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

TASK = Path(__file__).resolve().parent
ROOT = Path(r'C:\Users\guoqi\OneDrive - stevens.edu\Documents\GitHub\shape-force-est-IMU')
driver = TASK / 'candidate/diagnostics/oracle_ekf_gap/recovered_run/diagnose_recovered.py'
protocol_path = TASK / 'protocol.json'
protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
assert protocol['selected_case_ids'] == [1, 2] and protocol['noise_realization'] == 0
manifest = json.loads((TASK / 'baseline_manifest.json').read_text(encoding='utf-8'))
assert subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip() == protocol['source_commit']
proof = {'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
         'driver_sha256': hashlib.sha256(driver.read_bytes()).hexdigest(),
         'selected_case_ids_before_runs': protocol['selected_case_ids']}
(TASK / 'run_registration.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8', newline='\n')

def run(case_id):
    case_dir = TASK / f'case_{case_id}'
    case_dir.mkdir(exist_ok=False)
    argv = [sys.executable, '-B', str(driver), '--repo-root', str(ROOT), '--case-id', str(case_id),
            '--noise-realization', '0', '--output-dir', str(case_dir / 'abcd')]
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    (case_dir / 'abcd.log').write_text(result.stdout + result.stderr, encoding='utf-8', newline='\n')
    (case_dir / 'abcd_command.json').write_text(json.dumps({'argv': argv, 'returncode': result.returncode}, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'case_id': case_id, 'returncode': result.returncode,
                      'log': str(case_dir / 'abcd.log')}), flush=True)
    return result.returncode

with ThreadPoolExecutor(max_workers=2) as pool:
    codes = list(pool.map(run, protocol['selected_case_ids']))
assert all(code == 0 for code in codes), 'Case failure preserved in logs; do not replace the selected case'
print('Both preselected fixed-case A/B/C/D runs completed.', flush=True)
