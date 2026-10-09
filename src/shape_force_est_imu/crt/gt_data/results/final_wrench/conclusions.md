# External Wrench Estimation - Key Findings

## Oracle ceiling (GT-curvature, no IMU noise)

- Order (1,1,0), 5 params: NRMSE-F = 0.568, force dir = 36.4 deg.
  The 5-param polynomial basis cannot represent general 3D bending;
  force estimation is limited even with perfect shape.

- Order (3,3,2), 11 params: NRMSE-F = 0.083, force dir = 4.4 deg.
  With a rich modal basis and GT shape, wrench estimation is near-perfect.

## EKF gap: sparse IMU vs oracle ceiling

- (1,1,0): oracle 36.4 deg -> EKF 2-IMU 33.5 deg.
  EKF nearly matches oracle -- the gap is the model mismatch floor, not sensor limitation.

- (3,3,2): oracle 4.4 deg -> EKF 2-IMU 26.9 deg.
  Higher modal orders improve the oracle ceiling but remain underdetermined
  under sparse IMU sensing. 2-IMU outperforms 3-IMU for high orders due to
  better conditioning of the {0.50, 1.00} sensor placement.

## Constrained estimator (force-only, 2-IMU layout)

- Oracle constrained: dir-F = 5.6 deg, NRMSE-F = 10.9%.
- EKF unconstrained:  dir-F = 34.5 deg, NRMSE-F = 53.3%.
- EKF constrained:    dir-F = 19.5 deg, NRMSE-F = 55.2%.
  Constraining to the force-only subspace reduces direction error by
  15.0 deg vs unconstrained EKF.

## Files

| File | Content |
|---|---|
| oracle_order_summary.csv / .md   | Table 1: oracle accuracy vs modal order (TDCR dataset) |
| oracle_vs_ekf_table.csv / .md    | Table 2: oracle vs EKF shape + wrench (Kirchhoff GT) |
| constrained_force_only.csv / .md | Table 3: constrained force-only estimator |
| trend_force_direction.pdf / .png | Figure: force direction error vs modal order |