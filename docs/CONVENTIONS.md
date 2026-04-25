# Conventions for shape-force-est-IMU

## Arclength and curvature

The normalized arclength is

s = l / L,  s ∈ [0, 1],

where l is the physical arclength and L is the segment length.

The modal curvature κ(s) is the normalized curvature,

κ(s) = L κ_phys(l),  l = Ls.

The SE(3) backbone kinematics are written with respect to normalized arclength as

dT/ds = T [ hat(κ(s))   L e3
             0          0  ].

Equivalently, the implementation may integrate with translational component e3 and multiply the final position by L, but κ(s) remains normalized curvature.

## Tendon coordinate

The tendon coordinate used in the virtual-work equation is cable shortening / pull displacement,

q_i = ℓ_i,0 - ℓ_i.

Therefore,

J_qm = ∂q/∂m = - ∂ℓ/∂m.

Positive tension τ_i > 0 does positive virtual work when q_i increases.

## Virtual work

The modal static equilibrium is

J_qm^T τ + J_Vbm^T F_b = ∇_m U.

Therefore the generalized modal load attributed to the external wrench is

b_w = ∇_m U - J_qm^T τ,

and

J_Vbm^T F_b = b_w.

## Wrench and twist ordering

The body twist ordering is

V_b = [ω_b; v_b].

The dual body wrench ordering is therefore

F_b = [μ_b; f_b],

where μ_b is moment and f_b is force.

All code should use

F_b = [Mx, My, Mz, Fx, Fy, Fz]^T.

## Wrench reference point

Unless otherwise stated, all estimated wrenches are expressed about the tip-frame origin.

The same-origin body-to-world wrench conversion is rotation only:

[μ_w^tip; f_w] = blockdiag(R_tip, R_tip) [μ_b; f_b].

The full inverse-transpose adjoint should only be used when shifting the moment reference point to the world/base origin.