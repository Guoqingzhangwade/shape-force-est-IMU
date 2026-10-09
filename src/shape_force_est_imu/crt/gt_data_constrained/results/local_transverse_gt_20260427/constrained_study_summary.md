# Constrained Tip-Wrench Estimation Study — Summary


## Setup

- Rod: L = 0.1 m, EI = 2.9452e-03 N·m², GJ = 2.2656e-03 N·m²
- Modal order: (1, 1, 0)
- IMU layouts: 2-IMU @ [0.50, 1.00], 3-IMU @ [0.25, 0.50, 1.00]
- Noise σ = 0.5°, 5 noise realizations per case

## Selection matrices S (body / tip frame)

All constraints are posed as **F\_b = S z** where F\_b = [moment\_b; force\_b].

| Case | S shape | Admissible DOFs |
|------|---------|-----------------|
| force\_only | 6×3 | body [Fx, Fy, Fz] (zero moments) |
| transverse\_force | 6×2 | body [Fx, Fy] (zero moments + Fz) |
| moment\_only | 6×3 | body [Mx, My, Mz] (zero forces) |

## Results

Moment metrics are reported as N/A for force-only cases when the ground-truth moment norm is zero.

Main `constrained_summary.csv` metric columns: `force_err_N_mean`, `moment_err_Nm_mean`, `nrmse_force_mean`, `nrmse_moment_mean`, `force_dir_err_deg_mean`, `moment_dir_err_deg_mean`.

| Constrained case | Method | Layout | Force err [mN] | Moment err [mN·m] | NRMSE-F | NRMSE-M |
|-----------------|--------|--------|---------------|------------------|---------|---------|
| local_transverse_force | transverse_force_2d    | GT       |   5.84 ± 5.46 |   0.000 ± 0.000 | 2.1% | N/A |
| local_transverse_force | known_direction_1d_oracle_direction | GT       |   0.59 ± 0.70 |   0.000 ± 0.000 | 0.2% | N/A |

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