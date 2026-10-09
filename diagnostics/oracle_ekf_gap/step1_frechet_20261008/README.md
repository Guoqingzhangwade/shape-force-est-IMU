# Step 1: production EKF Frechet derivative correction

The saved two-line correction is now applied to the production variable-order EKF in `compare_oracle_vs_ekf_wrench_estimation.run_ekf_for_order`. The remaining residual, priors, covariance update, initialization, settings and mechanics are unchanged.

Pre-fix branch: `diagnostic/oracle-ekf-gap`; commit `5312a0fa9c2c5e98399de8a6251fe28df1f616e8`. Source and input hashes, exact commands and all fresh outputs are preserved here. The before and after runs were executed before the fix commit, so the recorded HEAD remains the pre-fix commit; `verification.json` identifies the changed working source by its SHA256.

## Fixed-case result

Case 0, noise realization 0, order `(2,2,1)`, actual sensor positions `[25/99, 50/99, 1]`, gamma 10, sigma 0.5 degrees, alpha 1, and 30 updates of the same saved noisy triplet. This is a static one-case comparison, not independent new noise or dynamic tracking.

| Quantity | Before | Production fix |
|---|---:|---:|
| Jacobian relative Frobenius error at the common Oracle state | 0.103601817 | 1.52991444e-10 |
| True 100-point shape RMS (mm) | 0.554543226 | 0.552548895 |
| Force vector error (N) | 0.649148481 | 0.646809856 |
| Force direction error (degrees) | 89.6631379 | 89.2879506 |
| Common world-base moment error (N m) | 0.00362691933 | 0.0036648255 |

A (dense Oracle), B (noiseless sparse offline fit) and C (noisy sparse offline fit) are unchanged within `1e-10` for the reported metrics. The production D result matches the independently reconstructed pre-fix-versus-fixed comparison. The Jacobian inconsistency is corrected, but force and shape improvements are small, and the common-origin moment error increases slightly. This does not resolve the large noisy wrench gap or establish its aggregate cause.

## Checks and preservation

The regression uses independent SO(3) residual finite differences in four deterministic nonzero 5D/8D states. All four fail with the old derivative and pass after the fix. The saved-case Jacobian passes the same finite-difference check at the common dense-Oracle state. All historical diagnostic evidence, GT/IMU inputs, and unrelated protected production sources remain byte-identical.

The old baseline is loaded from the explicit Git commit into a separate in-memory module; it is never restored over production. Both implementations use the same current dependencies. This isolates the derivative change.

## Scope limitation

The final artifact generator calls the corrected general-order function for every configured order, including 5D. The standalone comparison CLI still routes `(1,1,0)` to the separate `run_ekf_on_frame` implementation. That legacy path and the older sensor-placement/measurement-model/demo implementations still contain the approximation. They were deliberately left outside this bounded correction; this is not a fix of every EKF in the repository.

## Reproduce

From the repository root, choose fresh output directories:

```powershell
& '.\.venv\Scripts\python.exe' -B tests/check_variable_order_ekf_jacobian.py
& '.\.venv\Scripts\python.exe' -B diagnostics/oracle_ekf_gap/recovered_run/diagnose_recovered.py --output-dir "$env:TEMP/ekf-frechet-abcd-new"
& '.\.venv\Scripts\python.exe' -B diagnostics/oracle_ekf_gap/recovered_run/check_jacobian.py --baseline-ref 5312a0fa9c2c5e98399de8a6251fe28df1f616e8 --output-dir "$env:TEMP/ekf-frechet-jacobian-new"
```

Use `before_after.csv` for the complete A/B/C/D before/after values and `verification.json` for checks and hashes.
