## Table 3 — Known-Direction Tip-Force Estimation (Force-Only Constraint)

Dataset: `kirchhoff_gt_force_only.npz` (N=50 cases, zero-moment wrenches),
Modal order: (1,1,0),  IMU layouts: 2-IMU = {0.50, 1.00}, 3-IMU = {0.25, 0.50, 1.00}

| Method | Layout | Dir error F (°) | NRMSE-F | Force err (mN) |
|---|---|---|---|---|
| oracle_constrained | 2-IMU {0.50,1.00} | 5.6° | 10.9% | 43.7 |
| ekf_unconstrained | 2-IMU {0.50,1.00} | 34.5° | 53.3% | 205.5 |
| ekf_constrained | 2-IMU {0.50,1.00} | 19.5° | 55.2% | 186.1 |
| oracle_constrained | 3-IMU {0.25,0.50,1.00} | 5.6° | 10.9% | 43.7 |
| ekf_unconstrained | 3-IMU {0.25,0.50,1.00} | 34.3° | 52.9% | 203.9 |
| ekf_constrained | 3-IMU {0.25,0.50,1.00} | 15.9° | 49.7% | 171.1 |