from pathlib import Path
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

TASK = Path(__file__).resolve().parent
ROOT = Path(r'C:\Users\guoqi\OneDrive - stevens.edu\Documents\GitHub\shape-force-est-IMU')
driver = TASK / 'candidate/diagnostics/oracle_ekf_gap/step2_sensitivity_20261012/diagnose_sensitivity.py'
protocol = json.loads((TASK / 'protocol.json').read_text(encoding='utf-8'))
assert protocol['selected_case_ids'] == [1, 2]

def run(case_id):
    if case_id == 0:
        saved = ROOT / 'diagnostics/oracle_ekf_gap/step1_frechet_20261008/after_abcd/results.json'
        output = TASK / 'case_0_final_parity'
    else:
        saved = TASK / f'case_{case_id}/abcd/results.json'
        output = TASK / f'case_{case_id}/sensitivity_verified'
    argv = [sys.executable, '-B', str(driver), '--repo-root', str(ROOT), '--state-results', str(saved), '--output-dir', str(output)]
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    log_dir = TASK if case_id == 0 else TASK / f'case_{case_id}'
    (log_dir / f'sensitivity_case_{case_id}_verified.log').write_text(result.stdout + result.stderr, encoding='utf-8', newline='\n')
    (log_dir / f'sensitivity_case_{case_id}_verified_command.json').write_text(json.dumps({'argv': argv, 'returncode': result.returncode}, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'case_id': case_id, 'returncode': result.returncode, 'output': str(output)}), flush=True)
    return result.returncode

with ThreadPoolExecutor(max_workers=3) as pool:
    codes = list(pool.map(run, [0] + protocol['selected_case_ids']))
(TASK / 'sensitivity_verified_completion.json').write_text(json.dumps({'case_ids': [0,1,2], 'returncodes': codes}, indent=2) + '\n', encoding='utf-8', newline='\n')
print('Selected-case outcomes and any failures preserved; no replacement or parameter changes.', flush=True)
