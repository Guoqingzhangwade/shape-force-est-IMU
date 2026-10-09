# Local Transverse-Force Summary

Source: `gt_data_constrained/results/local_transverse_force_20260427_20steps_5noise/constrained_summary.csv`.

The load is force-only in the local tip/body x-y plane; moment metrics are N/A because the GT moment is zero.

| Case | Method | Layout | Force err [mN] | NRMSE-F | Force dir [deg] | Moment err [mN*m] | NRMSE-M | Moment dir [deg] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Local transverse force | Direct 6D baseline | 2-IMU | 87.658 +/- 50.950 | 36.29% | 20.301 | N/A | N/A | N/A |
| Local transverse force | Direct 6D baseline | 3-IMU | 85.161 +/- 51.365 | 34.94% | 19.432 | N/A | N/A | N/A |
| Local transverse force | Local transverse 2D | GT | 5.841 +/- 5.464 | 2.08% | 1.194 | N/A | N/A | N/A |
| Local transverse force | Local transverse 2D | 2-IMU | 16.937 +/- 11.094 | 7.07% | 2.104 | N/A | N/A | N/A |
| Local transverse force | Local transverse 2D | 3-IMU | 17.410 +/- 10.880 | 7.33% | 2.163 | N/A | N/A | N/A |
