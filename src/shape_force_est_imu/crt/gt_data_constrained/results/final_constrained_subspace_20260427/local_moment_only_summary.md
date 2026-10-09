# Local Moment-Only Summary

Source: `gt_data_constrained/results/local_moment_only_20260427_20steps_5noise/constrained_summary.csv`.

The load is pure moment in the local tip/body frame; force NRMSE and force direction are N/A because the GT force is zero. Absolute force error is retained as force leakage.

| Case | Method | Layout | Force err [mN] | NRMSE-F | Force dir [deg] | Moment err [mN*m] | NRMSE-M | Moment dir [deg] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Local moment only | Direct 6D baseline | 2-IMU | 32.216 +/- 16.236 | N/A | N/A | 1.789 +/- 0.820 | 10.87% | 4.767 |
| Local moment only | Direct 6D baseline | 3-IMU | 25.822 +/- 14.680 | N/A | N/A | 1.531 +/- 0.778 | 9.27% | 4.276 |
| Local moment only | Local moment 3D | GT | 0.000 +/- 0.000 | N/A | N/A | 0.477 +/- 0.542 | 2.02% | 1.159 |
| Local moment only | Local moment 3D | 2-IMU | 0.000 +/- 0.000 | N/A | N/A | 0.862 +/- 0.580 | 4.69% | 2.077 |
| Local moment only | Local moment 3D | 3-IMU | 0.000 +/- 0.000 | N/A | N/A | 0.876 +/- 0.619 | 4.79% | 2.146 |
