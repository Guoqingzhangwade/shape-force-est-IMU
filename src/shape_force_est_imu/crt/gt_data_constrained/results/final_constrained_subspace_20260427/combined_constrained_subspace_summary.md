# Frame-Consistent Constrained-Subspace Summary

Both datasets use body-frame load subspaces matching the constrained estimator, while `f_ext` and `l_ext` remain stored in world frame for compatibility.

| Case | Method | Layout | Force err [mN] | NRMSE-F | Force dir [deg] | Moment err [mN*m] | NRMSE-M | Moment dir [deg] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Local transverse force | Direct 6D baseline | 2-IMU | 87.658 +/- 50.950 | 36.29% | 20.301 | N/A | N/A | N/A |
| Local transverse force | Direct 6D baseline | 3-IMU | 85.161 +/- 51.365 | 34.94% | 19.432 | N/A | N/A | N/A |
| Local transverse force | Local transverse 2D | GT | 5.841 +/- 5.464 | 2.08% | 1.194 | N/A | N/A | N/A |
| Local transverse force | Local transverse 2D | 2-IMU | 16.937 +/- 11.094 | 7.07% | 2.104 | N/A | N/A | N/A |
| Local transverse force | Local transverse 2D | 3-IMU | 17.410 +/- 10.880 | 7.33% | 2.163 | N/A | N/A | N/A |
| Local moment only | Direct 6D baseline | 2-IMU | 32.216 +/- 16.236 | N/A | N/A | 1.789 +/- 0.820 | 10.87% | 4.767 |
| Local moment only | Direct 6D baseline | 3-IMU | 25.822 +/- 14.680 | N/A | N/A | 1.531 +/- 0.778 | 9.27% | 4.276 |
| Local moment only | Local moment 3D | GT | 0.000 +/- 0.000 | N/A | N/A | 0.477 +/- 0.542 | 2.02% | 1.159 |
| Local moment only | Local moment 3D | 2-IMU | 0.000 +/- 0.000 | N/A | N/A | 0.862 +/- 0.580 | 4.69% | 2.077 |
| Local moment only | Local moment 3D | 3-IMU | 0.000 +/- 0.000 | N/A | N/A | 0.876 +/- 0.619 | 4.79% | 2.146 |

Notes: direct 6D pseudoinverse is a minimum-norm baseline. NRMSE and direction entries are N/A when the corresponding GT force or moment norm is zero.
