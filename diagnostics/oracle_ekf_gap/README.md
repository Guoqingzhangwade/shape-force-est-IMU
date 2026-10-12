# Oracle–EKF audit evidence and diagnostic runners

Start with [Step 2: local orientation-to-wrench sensitivity](step2_sensitivity_20261012/README.md) for the latest fixed-case diagnosis. [Step 1: production Frechet fix](step1_frechet_20261008/README.md) records the preceding production correction and its validation. [recovered_diagnosis.md](recovered_diagnosis.md) preserves the earlier scientific diagnosis, and [recovered_manuscript_crosscheck.md](recovered_manuscript_crosscheck.md) records manuscript-to-result provenance. The existing reports preserve the original audit language and chronology. This English guide describes the portable runners and the limits of the saved evidence.

## Running a new diagnostic

Both runners default to the repository containing the script, inferred as `Path(__file__).resolve().parents[3]`. Use `--repo-root` to select another checkout; `--recovered-root` is an equivalent alias. The selected checkout must contain the recovered CRT implementation and these original inputs:

- `src/shape_force_est_imu/crt/gt_data/kirchhoff_gt_dataset.npz`
- `src/shape_force_est_imu/crt/gt_data/kirchhoff_imu_3imu.npz`

Use a Python environment with NumPy, SciPy, and the repository dependencies, and invoke both scripts with `-B`. Each invocation requires an explicit `--output-dir` that does **not** already exist. Use a separate new directory for each run; the scripts create it and refuse existing directories, including empty ones. Source files and input datasets are not modified. The A/B/C/D runner checks repository status while excluding only its own new output directory if that directory is inside the selected checkout.

PowerShell examples, from the repository root after migration:

```powershell
& '.\.venv\Scripts\python.exe' -B diagnostics/oracle_ekf_gap/recovered_run/diagnose_recovered.py --repo-root . --output-dir "$env:TEMP/oracle-ekf-abcd-new"
& '.\.venv\Scripts\python.exe' -B diagnostics/oracle_ekf_gap/recovered_run/check_jacobian.py --repo-root . --baseline-ref 5312a0fa9c2c5e98399de8a6251fe28df1f616e8 --output-dir "$env:TEMP/oracle-ekf-jacobian-new"
```

Choose unused output names on subsequent runs. Before migration, supply the recovered checkout's path to `--repo-root`; the original, older checkout lacks the required implementation and data. Omitting `--repo-root` is appropriate once these scripts reside in the intended recovered repository.

`diagnose_recovered.py` writes `results.json`, `results.csv`, and `original_states.npz` (the retained filename). It first preserves the dense-oracle and selected production EKF states, then performs the bounded offline comparisons. New output labels D as `D_production_ekf` and records the derivative implementation and source hash; old evidence retains its original labels.

`check_jacobian.py` writes `jacobian_correction.json` and a reviewable `jacobian_correction.patch`. After the production fix, it requires an explicit `--baseline-ref`, loads that pre-fix source with `git show`, and executes the baseline **in memory only** while importing the actual fixed production module. It requires production to match the baseline plus exactly the intended import and derivative-expression edits after LF normalization, and records the resolved commit/blob and source/input hashes. The patch describes baseline to fixed code and is never applied by the runner. Without `--baseline-ref`, the earlier in-memory-correction comparison remains available only when production still contains the expected pre-fix expression. A fixed production source without a baseline, or unrelated source differences, stops the comparison rather than silently selecting a different baseline.

## What the historical recovered A/B/C/D run established

The saved run uses one physical case, one saved three-IMU noise realization, modal order `(2,2,1)`, the actual stored sensor positions, and the original mechanics. A/B/C/D differ in available observations and inference:

| Method | Information and inference | True 100-point shape RMS (mm) | Force error (N) | Force direction error (degrees) |
|---|---|---:|---:|---:|
| A | Dense noiseless GT orientations; curvature least squares | 0.0143924 | 0.000989835 | 0.200049 |
| B | One noiseless orientation triplet; offline least squares | 0.0202995 | 0.0322491 | 5.38576 |
| C | One noisy orientation triplet; offline least squares | 0.661900 | 0.778697 | 94.0307 |
| D | The same noisy triplet reused for 30 original EKF updates | 0.554543 | 0.649148 | 89.6631 |

These are the existing [recovered_run/results.csv](recovered_run/results.csv) values, with full configuration, trial outcomes, and hashes in [results.json](recovered_run/results.json). They are a one-case diagnosis, not a reproduction of manuscript aggregate means. Offline optimization convergence does not establish global uniqueness or observability.

The historical derivative check compared the original analytic residual Jacobian with central finite differences, then replaced one exponential derivative expression with `scipy.linalg.expm_frechet` in a separate in-memory module. Relative Jacobian error fell from `0.103602` to approximately `1.53e-10`. Its effect on this case was modest: force error changed from `0.649148` to `0.646810 N`, direction error from `89.6631` to `89.2880 degrees`, and true shape RMS from `0.554543` to `0.552549 mm`. The derivative error was real, but this check did not explain away the large remaining wrench error. Migration itself left production unchanged. See the immutable [jacobian_correction.json](recovered_run/jacobian_correction.json); the separate Step 1 report records the subsequent production change and fresh checks.

## Chronology and immutable historical evidence

1. The initial [diagnosis.md](diagnosis.md), [manuscript_crosscheck.md](manuscript_crosscheck.md), and [history_evidence.md](history_evidence.md) examined the older workspace before original inputs were recovered. Their missing-input statements describe that earlier scope.
2. Another local checkout contained the ignored original datasets and saved results. [recovered_pipeline_notes.md](recovered_pipeline_notes.md), [recovered_manuscript_crosscheck.md](recovered_manuscript_crosscheck.md), and [recovered_diagnosis.md](recovered_diagnosis.md) document that recovery and supersede the earlier availability conclusions.
3. `recovered_run/` stores the original A/B/C/D and in-memory derivative-check evidence. The `results_before_metric_index_fix.*` files preserve the first diagnostic output; the later `results.*` only correct the historical 20-point field's sample indexing to match the table generator. Fitted states, force errors, and true 100-point RMS were unchanged.
4. [repo_reconciliation_20261008/](repo_reconciliation_20261008/) records the checkout reconciliation and preserves preexisting source changes. Portability edits to the two runners do not alter any historical JSON, CSV, NPZ, patch, verification record, or source snapshot. Historical verification hashes identify the earlier script bytes, not these updated runner files.
5. [step1_frechet_20261008/README.md](step1_frechet_20261008/README.md) records the later authorized production Fréchet change and fresh same-case before/after validation. Its explicit pre-fix baseline is `5312a0fa9c2c5e98399de8a6251fe28df1f616e8`. Those new results are kept separate from `recovered_run/` and the manuscript aggregate tables; consult the Step 1 report for check status and measured outcomes.

The manuscript summaries come from different runs: the matched-model study uses 10 shapes × 3 noise realizations; the recovered oracle-vs-EKF table uses 20 Kirchhoff cases × 3 static-noise realizations; the separate oracle order sweep uses an older 10-case TDCR dataset; and the cross-model geometry study uses 50 cases × 5 time-varying noise sequences. The table's `0.07` and `44.2 degrees` are confirmed rounded saved aggregate values, not targets for the one-case diagnostic. Its `rms_mm` field is actually mean centerline distance over 20 points, and its `NRMSE` fields average relative vector errors rather than implementing the manuscript's stated fixed-scale aggregate RMSE. Keep those historical fields distinct from the diagnostic's explicitly named true RMS and fixed-scale errors.

The extracted manuscript remains an external reference and was not part of this directory's migration. The migration preserved evidence and bounded diagnostic utilities without applying the saved derivative patch. The subsequent Step 1 production change is a separate, explicitly authorized action.

## Implementation and reporting discrepancies

The original variable-order 8D CRT EKF and manuscript inputs were absent from the older checkout, but they are present in the recovered checkout used for this branch. The separate `ekf/kinematics.py` demo still supports only 5D/6D and is not the manuscript's variable-order implementation.

The manuscript's rounded oracle and EKF values match saved committee results, but the code's `rms_mm` and `NRMSE` definitions differ from the manuscript definitions as described above. Saved aggregate tables were traced, not regenerated. Their exact generation source state cannot be established from a commit hash alone: the body-frame constrained datasets were generated with local changes that were not committed at that time. This branch now preserves both those source changes and the dataset bytes.

The historical `transverse_force` generator constrains world-frame Fz, whereas the corresponding inversion constrains body-frame transverse force. The newer `local_transverse_force` implementation consistently uses the tip body frame and is preserved here; the legacy case remains available and retains this limitation.

The original EKF exponential derivative had a verified analytic-Jacobian error. Its correction was first checked in memory and preserved as a patch; the later Step 1 production change replaces that expression with `scipy.linalg.expm_frechet`. The historical one-case comparison showed only a small wrench improvement. Fresh production validation belongs to the Step 1 report. Sparse noisy orientation data and wrench sensitivity remain relevant to the observed gap; one case cannot establish the aggregate cause.

For the completed migration and everyday Git commands, see [SYNC_WORKFLOW.md](SYNC_WORKFLOW.md). New validation records are separate from historical evidence.
