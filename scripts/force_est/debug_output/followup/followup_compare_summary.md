
## Follow-up Study: Oracle vs EKF Wrench Estimation

Dataset: kirchhoff_gt_dataset.npz (N=3 cases, 5 noise realizations)


### Order (1,1,0)

| Method | F-dir (°) | NRMSE-F | M-dir (°) | NRMSE-M |
|---|---|---|---|---|
| oracle | 32.7° | 0.522 | 1.6° | 0.064 |
| 2-IMU [0.5,1.0] | 89.5° | 1.479 | 62.6° | 1.544 |
| 3-IMU [0.25,0.5,1.0] | 89.4° | 1.484 | 62.5° | 1.543 |

### Order (3,3,2)

| Method | F-dir (°) | NRMSE-F | M-dir (°) | NRMSE-M |
|---|---|---|---|---|
| oracle | 2.6° | 0.072 | 1.5° | 0.048 |
| 2-IMU [0.5,1.0] | 49.0° | 8.516 | 102.3° | 3.174 |
| 3-IMU [0.25,0.5,1.0] | 49.1° | 8.381 | 102.3° | 3.169 |