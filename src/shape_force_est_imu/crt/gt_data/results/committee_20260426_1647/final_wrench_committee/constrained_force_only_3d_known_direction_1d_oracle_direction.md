## Table 3 — Force-Only 3D and Known-Direction 1D Oracle-Direction Tip-Force Estimation

Dataset: `kirchhoff_gt_force_only.npz` (N=50 cases, zero-moment wrenches),
Modal order: (1,1,0),  IMU layouts: 2-IMU = {0.50, 1.00}, 3-IMU = {0.25, 0.50, 1.00}

| Method | Layout | Dir error F (°) | NRMSE-F | Force err (mN) |
|---|---|---|---|---|
| oracle_force_only_3d | 2-IMU {0.50,1.00} | 1.9° | 3.7% | 15.1 |
| oracle_known_direction_1d_oracle_direction | 2-IMU {0.50,1.00} | 0.0° | 1.4% | 6.0 |
| ekf_direct_6d_baseline | 2-IMU {0.50,1.00} | 34.5° | 53.7% | 206.4 |
| ekf_force_only_3d | 2-IMU {0.50,1.00} | 17.1° | 52.7% | 175.2 |
| ekf_known_direction_1d_oracle_direction | 2-IMU {0.50,1.00} | 0.0° | 6.0% | 22.3 |
| oracle_force_only_3d | 3-IMU {0.25,0.50,1.00} | 1.9° | 3.7% | 15.1 |
| oracle_known_direction_1d_oracle_direction | 3-IMU {0.25,0.50,1.00} | 0.0° | 1.4% | 6.0 |
| ekf_direct_6d_baseline | 3-IMU {0.25,0.50,1.00} | 34.3° | 53.0% | 204.7 |
| ekf_force_only_3d | 3-IMU {0.25,0.50,1.00} | 13.5° | 41.9% | 145.9 |
| ekf_known_direction_1d_oracle_direction | 3-IMU {0.25,0.50,1.00} | 0.0° | 6.6% | 25.4 |