# Follow-up Study: Modal Order & Sparse-IMU Wrench Estimation

**Datasets used**
- Task 1: `artifacts/data/tdcr_gt_samples_10.npz` — 10 samples, random 3D wrenches
- Tasks 2–3: `src/.../crt/gt_data/kirchhoff_gt_dataset.npz` — 50 cases (3 used), 5 noise realizations
- IMU files: `kirchhoff_imu_2imu.npz` [s=0.505, 1.0], `kirchhoff_imu_3imu.npz` [s=0.253, 0.505, 1.0]

---

## Task 1 — Oracle: Fitted vs GT-curvature at order (3,3,2)

Dataset: `tdcr_gt_samples_10.npz` (10 samples, random 3D force+moment)

| Method | NRMSE-F | Rel-mag F | Dir F (°) ± std | Dir M (°) ± std |
|---|---|---|---|---|
| GT-curvature (1,2,0) — 6 params | 0.778 | 43% | 21.4 ± 18.8 | 2.8 ± 1.6 |
| Fitted L-BFGS (1,2,0) — 6 params | 1.549 | 101% | 47.2 ± 34.7 | 7.0 ± 8.5 |
| **GT-curvature (3,3,2) — 11 params** | **0.203** | **8.3%** | **4.4 ± 5.2** | **1.7 ± 1.6** |
| Fitted L-BFGS (3,3,2) — 11 params | 0.980 | 39% | 12.6 ± 25.7 | 2.0 ± 1.7 |

**Key finding**: The 11-param GT-curvature oracle (4.4°) is the true ceiling. Nonlinear
fitting (L-BFGS-B) with 11 params degrades to 12.6° due to local-minima risk and
high variance (std = 25.7° vs 5.2° for GT-curvature). It still improves over 6-param
fitting (12.6° vs 47.2°) but remains ~3× worse than the ideal ceiling.

---

## Tasks 2–3 — Oracle vs EKF for 2-IMU / 3-IMU layouts

Dataset: `kirchhoff_gt_dataset.npz` (3 representative cases)
IMU layouts: 2-IMU [0.5, 1.0] and 3-IMU [0.25, 0.5, 1.0]

### Order (1,1,0) — 5 params, EKF baseline

| Method | F-dir (°) | NRMSE-F | M-dir (°) | NRMSE-M |
|---|---|---|---|---|
| Oracle (GT-curvature) | 32.7 | 0.522 | 1.6 | 0.064 |
| EKF 2-IMU [0.5, 1.0] | 89.5 | 1.479 | 62.6 | 1.544 |
| EKF 3-IMU [0.25, 0.5, 1.0] | 89.4 | 1.484 | 62.5 | 1.543 |

### Order (3,3,2) — 11 params

| Method | F-dir (°) | NRMSE-F | M-dir (°) | NRMSE-M |
|---|---|---|---|---|
| **Oracle (GT-curvature)** | **2.6** | **0.072** | **1.5** | **0.048** |
| EKF 2-IMU [0.5, 1.0] | 49.0 | 8.516 | 102.3 | 3.174 |
| EKF 3-IMU [0.25, 0.5, 1.0] | 49.1 | 8.381 | 102.3 | 3.169 |

---

## Answers to the Three Questions

### Q1: How good is the realistic oracle ceiling?

With **GT-curvature + 11 modal params**, the oracle achieves:
- Force direction error: **2.6–4.4°** (cross-dataset, 3–10 samples)
- Force magnitude error: **7–8%**
- Moment direction error: **1.5–1.7°**, magnitude error **4–5%**

This is an excellent ceiling. The virtual-work wrench principle is sound when shape
is accurately known with a rich enough modal basis.

### Q2: Does 11-param survive fitted modal states?

Partially. Fitted 11-param achieves **12.6°** direction (vs 4.4° for GT-curvature).
The degradation from fitting introduces ~3× worse direction accuracy with very high
variance (25.7° std), indicating local-minima sensitivity. The fitted result is still
better than 6-param fitting (47°) but cannot reach the GT-curvature ceiling.

### Q3: How much degradation from sparse IMU?

The gap from oracle ceiling to EKF is enormous:

| Transition | Force direction gap |
|---|---|
| GT-curvature (11p) → fitted (11p) | +8.2° (fitting itself) |
| Fitted (11p) oracle → EKF 2-IMU (11p) | +36° (IMU observability) |
| GT-curvature (5p) → EKF 2-IMU (5p) | +57° (combined model+observability) |

**Sparse IMU observability is the dominant bottleneck.** 2–3 IMU orientation
measurements cannot constrain 5–11 modal parameters precisely enough for
virtual-work wrench estimation. Adding more modes makes the oracle better but
makes the EKF worse (underdetermined state space).

---

## Files saved

- `followup_compare_results.json` — raw per-case metrics
- `followup_compare_summary.md` — this table (Tasks 2-3)
- `s10_fitted_o11_results.txt` — fitted 11-param run log (Task 1)
- `fitted_o11_fig_00.png` — shape figure from fitted run
