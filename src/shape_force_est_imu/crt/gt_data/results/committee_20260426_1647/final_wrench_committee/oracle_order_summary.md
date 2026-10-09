## Table 1 — Oracle Wrench Estimation: Modal Order vs. Accuracy

Dataset: `tdcr_gt_samples_10.npz` (N=10), GT-curvature oracle (no IMU noise)

| Order | n params | NRMSE-F | Rel. mag error F | Dir error F (°) | Dir error M (°) |
|---|---|---|---|---|---|
| (1,1,0) | 5 | 0.563 | 24.9% | 36.1° | 2.8° |
| (1,2,0) | 6 | 0.350 | 20.2% | 17.5° | 2.1° |
| (2,2,0) | 7 | 0.082 | 3.7% | 3.9° | 1.5° |
| (2,2,1) | 8 | 0.012 | 0.5% | 0.6° | 0.3° |
| (3,3,2) | 11 | 0.002 | 0.1% | 0.1° | 0.0° |