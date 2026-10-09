# Wrench Observability Diagnostics

## Definitions

- Numerical rank is the number of singular values satisfying `sigma_i / sigma_1 >= 1e-06`.
- Effective rank is the practical rank under the larger threshold `sigma_i / sigma_1 >= 0.01`.
- Wrench components are ordered `[Mx, My, Mz, Fx, Fy, Fz] = [moment; force]`.
- Component observability is computed in dimensionless wrench coordinates.

## Scaling

The component diagnostics use `A_scaled = A @ D_phys`, where `D_phys = diag([moment_scale]*3 + [force_scale]*3)`.
For this run, `moment_scale = 1` and `force_scale = 10`.
Observable fractions depend on this scaling because the moment and force coordinates have different physical units.

The original rank summary is computed from the raw map `A = J_Vbm.T`. Component observable fractions are computed from the scaled map `A_scaled = A @ D_phys`, so the scaled effective rank can differ from the raw effective rank. The sum of the component observable fractions equals the trace of the observable-subspace projection and should approximately equal the scaled effective rank. This scaling is used because moment and force coordinates have different units.

## Interpretation

- `observable_fraction_*` near 1 means that component lies mostly in the strong observable subspace.
- `observable_fraction_*` near 0 means that component lies mostly in the weak or null subspace.
- `weakest_v_*_abs` reports absolute component weights of the weakest right singular vector of `A_scaled`.
- `null_v_*_abs` reports an exact right-null direction when `A_scaled` has fewer than six singular values, such as the common 5x6 case.
- Load-subspace diagnostics report rank and conditioning of `A @ S` for force-only, transverse-force, and moment-only subspaces.

Known-direction gains were computed from GT force direction rotated into the tip body frame.
