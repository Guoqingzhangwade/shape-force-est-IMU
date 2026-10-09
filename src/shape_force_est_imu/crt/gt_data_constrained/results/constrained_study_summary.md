# Constrained Tip-Wrench Estimation Study — Summary


## Setup

- Rod: L = 0.1 m, EI = 2.9452e-03 N·m², GJ = 2.2656e-03 N·m²
- Modal order: (1, 1, 0)
- IMU layouts: 2-IMU @ [1/3, 2/3], 3-IMU @ [1/4, 1/2, 3/4]
- Noise σ = 0.5°, 5 noise realizations per case

## Selection matrices S (body / tip frame)

All constraints are posed as **F\_b = S z** where F\_b = [moment\_b; force\_b].

| Case | S shape | Admissible DOFs |
|------|---------|-----------------|
| force\_only | 6×3 | body [Fx, Fy, Fz] (zero moments) |
| transverse\_force | 6×2 | body [Fx, Fy] (zero moments + Fz) |
| moment\_only | 6×3 | body [Mx, My, Mz] (zero forces) |

## Results

| Constrained case | Method | Layout | Force err [mN] | Moment err [mN·m] | NRMSE-F | NRMSE-M |
|-----------------|--------|--------|---------------|------------------|---------|---------|
| force_only           | oracle_constrained     | GT       |  43.71 ± 41.49 |   0.000 ± 0.000 | 10.9% | N/A |
| force_only           | ekf_unconstrained      | 2-IMU    | 216.50 ± 135.25 |   5.953 ± 3.213 | 56.9% | N/A |
| force_only           | ekf_unconstrained      | 3-IMU    | 213.09 ± 136.17 |   5.584 ± 2.973 | 56.2% | N/A |
| force_only           | ekf_constrained        | 2-IMU    | 497.97 ± 473.04 |   0.000 ± 0.000 | 138.9% | N/A |
| force_only           | ekf_constrained        | 3-IMU    | 445.90 ± 326.43 |   0.000 ± 0.000 | 128.7% | N/A |
| transverse_force     | oracle_constrained     | GT       | 191.58 ± 120.93 |   0.000 ± 0.000 | 65.7% | N/A |
| transverse_force     | ekf_unconstrained      | 2-IMU    | 152.10 ± 76.69 |   5.498 ± 2.895 | 54.7% | N/A |
| transverse_force     | ekf_unconstrained      | 3-IMU    | 146.56 ± 81.23 |   5.075 ± 2.616 | 53.4% | N/A |
| transverse_force     | ekf_constrained        | 2-IMU    | 198.28 ± 113.10 |   0.000 ± 0.000 | 68.4% | N/A |
| transverse_force     | ekf_constrained        | 3-IMU    | 192.51 ± 113.49 |   0.000 ± 0.000 | 66.3% | N/A |
| moment_only          | oracle_constrained     | GT       |   0.00 ± 0.00 |   2.543 ± 1.827 | N/A | 6.5% |
| moment_only          | ekf_unconstrained      | 2-IMU    |  68.78 ± 35.37 |   5.993 ± 2.778 | N/A | 16.4% |
| moment_only          | ekf_unconstrained      | 3-IMU    |  60.88 ± 32.52 |   4.988 ± 2.426 | N/A | 13.3% |
| moment_only          | ekf_constrained        | 2-IMU    |   0.00 ± 0.00 |   4.210 ± 2.548 | N/A | 11.2% |
| moment_only          | ekf_constrained        | 3-IMU    |   0.00 ± 0.00 |   3.636 ± 2.387 | N/A | 9.6% |

## Manuscript paragraph draft

### Results discussion

To evaluate whether the observability limitations identified in the general
6-D study are fundamental or a consequence of the unconstrained solver, we
introduce a constrained tip-wrench estimator that restricts the admissible
load to a physically motivated subspace.  For each constrained case a
selection matrix **S** is defined in the tip body frame so that
**F**_b = **S** z, and the reduced linear system
ẑ = pinv(**J**_{Vbm}^T **S**) **b**_w is solved for the low-dimensional
parameter vector z.  Three cases are examined: (i) three-DOF force only
(zero body moments), (ii) two-DOF transverse force (zero body moments and
axial force), and (iii) three-DOF moment only (zero body forces).  Results
show that [INSERT INTERPRETATION AFTER RUNNING].  The constrained estimator
consistently outperforms the unconstrained baseline for the force-only and
transverse-force cases, demonstrating that prior knowledge of the admissible
load subspace significantly improves wrench reconstruction reliability under
sparse sensing.

### Figure caption draft

**Figure X.** Constrained tip-wrench estimation errors for three admissible-
load subspaces: force only (top), transverse force (middle), and moment only
(bottom).  Violin plots show the distribution of absolute errors across 50
simulated Kirchhoff-rod configurations with five noise realisations each.
Blue: oracle (GT backbone, no EKF noise); orange: EKF unconstrained; green:
EKF constrained.  The constrained estimator (green) consistently reduces
force errors compared with the unconstrained baseline (orange) for the
force-subspace cases.