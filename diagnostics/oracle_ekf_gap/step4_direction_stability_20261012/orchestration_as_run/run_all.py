"""Launch the committed Step 3 runners on every saved case/noise pair; outputs stay outside the repo."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

TASK = Path(__file__).resolve().parent
ROOT = Path(r'C:\Users\guoqi\OneDrive - stevens.edu\Documents\GitHub\shape-force-est-IMU')
STEP3 = ROOT / 'diagnostics/oracle_ekf_gap/step3_confirmation_20261012'
RUNS = TASK / 'runs'
WORKERS = 6

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), '-c', 'core.fsmonitor=false', *args], text=True).strip()

protocol = json.loads((TASK / 'protocol.json').read_text(encoding='utf-8'))
head = git('rev-parse', 'HEAD')
assert head.startswith(protocol['source_commit']) and not git('status', '--porcelain=v1')
with np.load(ROOT / 'src/shape_force_est_imu/crt/gt_data/kirchhoff_gt_dataset.npz', allow_pickle=False) as data:
    case_ids = [int(c) for c in data['case_id']]
assert len(case_ids) == 50 and len(set(case_ids)) == 50
pairs = [(c, n) for c in case_ids for n in protocol['noise_realizations']]
registration = {'protocol_sha256': sha(TASK / 'protocol.json'), 'head': head,
    'runner_sha256': {name: sha(STEP3 / name) for name in ('diagnose_recovered.py', 'diagnose_sensitivity.py', 'sensitivity_math.py')},
    'pair_count': len(pairs), 'workers': WORKERS}
(TASK / 'run_registration.json').write_text(json.dumps(registration, indent=2) + '\n', encoding='utf-8', newline='\n')
RUNS.mkdir(exist_ok=False)

def run(pair):
    case, noise = pair
    folder = RUNS / f'case_{case:02d}_noise_{noise}'
    folder.mkdir()
    record = {'case_id': case, 'noise_realization': noise}
    stages = [('abcd', [sys.executable, '-B', str(STEP3 / 'diagnose_recovered.py'), '--repo-root', str(ROOT), '--case-id', str(case),
                        '--noise-realization', str(noise), '--output-dir', str(folder / 'abcd')]),
              ('sensitivity', [sys.executable, '-B', str(STEP3 / 'diagnose_sensitivity.py'), '--repo-root', str(ROOT),
                               '--state-results', str(folder / 'abcd/results.json'), '--output-dir', str(folder / 'sensitivity')])]
    for name, argv in stages:
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        (folder / f'{name}.log').write_text(result.stdout + result.stderr, encoding='utf-8', newline='\n')
        record[f'{name}_returncode'] = result.returncode
        if result.returncode:
            break
    return record

start = time.perf_counter()
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    records = list(pool.map(run, pairs))
completion = {'elapsed_seconds': time.perf_counter() - start, 'head_after': git('rev-parse', 'HEAD'),
    'status_after': git('status', '--porcelain=v1'), 'runs': records}
(TASK / 'completion.json').write_text(json.dumps(completion, indent=2) + '\n', encoding='utf-8', newline='\n')
failed = [r for r in records if any(v for k, v in r.items() if k.endswith('returncode'))]
print(json.dumps({'runs': len(records), 'failed': len(failed), 'elapsed_seconds': completion['elapsed_seconds']}), flush=True)
