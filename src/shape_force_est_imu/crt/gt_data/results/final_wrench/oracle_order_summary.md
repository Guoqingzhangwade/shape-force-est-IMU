## Table 1 — Oracle Wrench Estimation: Modal Order vs. Accuracy

Dataset: `tdcr_gt_samples_10.npz` (N=10), GT-curvature oracle (no IMU noise)

| Order | n params | NRMSE-F | Rel. mag error F | Dir error F (°) | Dir error M (°) |
|---|---|---|---|---|---|
| (1,1,0) | 5 | 0.568 | 25.0% | 36.4° | 2.6° |
| (1,2,0) | 6 | 0.427 | 22.7% | 21.4° | 2.7° |
| (2,2,0) | 7 | 0.154 | 4.9% | 8.0° | 2.3° |
| (2,2,1) | 8 | 0.092 | 2.6% | 5.0° | 1.6° |
| (3,3,2) | 11 | 0.083 | 2.5% | 4.4° | 1.6° |