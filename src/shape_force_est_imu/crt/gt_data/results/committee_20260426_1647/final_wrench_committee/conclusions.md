# External Wrench Estimation — Key Findings

## Oracle ceiling (GT-curvature, no IMU noise)

- Order (1,1,0), 5 params: NRMSE-F = 0.563,  force dir = 36.1°.  The 5-param polynomial basis cannot represent general 3D bending;
  force estimation is limited even with perfect shape.

- Order (3,3,2), 11 params: NRMSE-F = 0.002,  force dir = 0.1°.  With a rich modal basis and GT shape, wrench estimation is near-perfect.

## EKF gap: sparse IMU vs. oracle ceiling

- (1,1,0): oracle 36.1°  → EKF 2-IMU 34.8°.  EKF nearly matches oracle — the gap is the model mismatch floor, not sensor limitation.

- (3,3,2): oracle 0.1°  → EKF 2-IMU 47.3°.  Higher modal orders improve the oracle ceiling but remain underdetermined
  under sparse IMU sensing. 2-IMU outperforms 3-IMU for high orders due to
  better conditioning of the {0.50, 1.00} sensor placement.

## Constrained estimator (force-only, 2-IMU layout)

- Oracle force_only_3d: dir-F = 1.9°,  NRMSE-F = 3.7%.
- Oracle known_direction_1d_oracle_direction: dir-F = 0.0°,  NRMSE-F = 1.4%.
- EKF direct_6d_baseline: dir-F = 34.5°,  NRMSE-F = 53.7%.
- EKF force_only_3d:   dir-F = 17.1°,  NRMSE-F = 52.7%.

## Files

| File | Content |
|---|---|
| `oracle_order_summary.csv / .md` | Table 1: oracle accuracy vs modal order (TDCR dataset) |
| `oracle_vs_ekf_table.csv / .md`  | Table 2: oracle vs EKF shape + wrench (Kirchhoff GT) |
| `constrained_force_only_3d_known_direction_1d_oracle_direction.csv / .md` | Table 3: force-only and oracle-direction diagnostics |
| `trend_force_direction.pdf / .png` | Figure: force direction error vs modal order |