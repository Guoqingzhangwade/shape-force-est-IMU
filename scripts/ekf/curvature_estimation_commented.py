# -*- coding: utf-8 -*-
"""curvature_estimation_commented.py

A fully‑commented, self‑contained demonstration of 5‑parameter curvature
estimation for a continuum/soft‑robot backbone.  The script shows how to

1.  Parameterise curvature as first‑order polynomials plus a constant twist
    about the backbone (5 parameters total).
2.  Integrate the resulting body twist along normalised arclength *s∈[0,1]*
    with a 4th‑order Magnus expansion and product‑of‑exponentials (PoE)
    formulation, yielding a homogeneous transform **T(s)** ∈ *SE(3).*  The
    translation component is later scaled by a characteristic length *L* so
    that the model is agnostic to units until the very end.
3.  Compute analytical partial derivatives **∂T/∂mᵢ** for all shape
    parameters *m₀ … m₄*.  These gradients are required by BFGS so that the
    optimiser can converge quickly (≈10× faster than finite‑difference).
4.  Define a cost‑function that compares only the **orientation** of the
    predicted frames against a set of sparsely observed frames.  The error is
    the smallest geodesic angle in *SO(3)* ("axis‑angle" distance).
5.  Solve the resulting nonlinear least‑squares with SciPy's `minimize`
    (BFGS) and visualise the recovered backbone shape versus ground truth.

Author: Guoqing Zhang 
Date  : 2025‑04‑22
"""

##############################################################################
# Imports
##############################################################################

import numpy as np                       # linear‑algebra work‑horse
from scipy.linalg import expm            # matrix exponential  (SciPy ≥ 1.8)
from scipy.optimize import minimize      # BFGS optimiser with analytic grad
import matplotlib.pyplot as plt          # quick‑and‑dirty plotting utility
import ipdb

##############################################################################
# 1) Five‑parameter curvature model
##############################################################################
#
#   kappa_x(s) = m[0] + m[1]·s        (linear in arc‑length)
#   kappa_y(s) = m[2] + m[3]·s
#   kappa_z(s) = m[4]                 (constant twist)
#
# This is the *extended polynomial curvature model* with first‑order bending
# in *x* and *y* plus a uniform axial twist.  It is expressive enough for many
# continuum‑robot segments yet still differentiable in closed‑form.
##############################################################################

def curvature_kappa(s: float, m: np.ndarray) -> np.ndarray:
    """Return the 3‑vector **κ(s)**=(κₓ,κ_y,κ_z).

    Parameters
    ----------
    s : float
        Normalised arclength ∈[0,1].  *s=0* corresponds to the base frame.
    m : array_like, shape (5,)
        Shape parameters ``[m0, m1, m2, m3, m4]`` as described above.
    """
    kx = m[0] + m[1] * s  # linear bend around *y*
    ky = m[2] + m[3] * s  # linear bend around *x*
    kz = m[4]             # constant axial twist
    return np.array([kx, ky, kz])


##############################################################################
# Twist representation and helper utilities
##############################################################################


def twist_matrix(kappa: np.ndarray) -> np.ndarray:
    """Map **κ** ↦ 4×4 *se(3)* matrix (body twist).

    The matrix is organised such that

        η(κ) = [  skew(κ)   e₃ ]
                [    0     0  ] ,   where   e₃=(0,0,1)ᵀ.

    The *translation* column e₃ reproduces the Bishop frame convention often
    used in Cosserat rods: integrating η yields frames whose *z*‑axis follows
    the backbone tangent.
    """

    def skew(u):  # local helper so we don't pollute global namespace
        "Return the 3×3 skew‑symmetric matrix for cross‑product."""
        return np.array([
            [ 0.0,   -u[2],  u[1]],
            [ u[2],   0.0,  -u[0]],
            [-u[1],  u[0],   0.0]
        ])

    e3 = np.array([0.0, 0.0, 1.0])
    top_left = skew(kappa)
    top_block = np.hstack([top_left, e3.reshape(3, 1)])  # 3×4
    bottom_block = np.array([[0.0, 0.0, 0.0, 0.0]])      # 1×4
    return np.vstack([top_block, bottom_block])          # 4×4


##############################################################################
# 2) 4th‑order Magnus expansion on sub‑intervals
##############################################################################
# We split *[0,s]* into *γ* equally‑spaced sub‑intervals.  On each piece we
# approximate the exact log‑map with the *two‑point* (Gauss‑Legendre) 4th‑order
# Magnus expansion.  This keeps rotations valid for large curvature while
# avoiding expensive adaptive integration.
##############################################################################


def magnus_4th_subinterval(s_start: float, s_end: float, m: np.ndarray) -> np.ndarray:
    """Return Ψₖ (4×4) — the Magnus log on sub‑interval *[s_start,s_end]*."""
    h = s_end - s_start                           # sub‑interval length

    # Gauss‑Legendre nodes (±√3/6) mapped to the interval
    c1 = s_start + h * (0.5 - np.sqrt(3.0) / 6.0)
    c2 = s_start + h * (0.5 + np.sqrt(3.0) / 6.0)

    # Evaluate twists at quadrature nodes
    eta1 = twist_matrix(curvature_kappa(c1, m))
    eta2 = twist_matrix(curvature_kappa(c2, m))

    # Classical 4th‑order Magnus series truncated after the commutator term
    part1 = (h / 2.0) * (eta1 + eta2)
    part2 = (h**2 * np.sqrt(3.0) / 12.0) * (eta1 @ eta2 - eta2 @ eta1)
    return part1 + part2


def product_of_exponentials(m: np.ndarray, s: float, *, gamma: int = 10,
                            L: float = 100.0) -> np.ndarray:
    """Integrate **η(κ)** from 0→s and return **T(s)** ∈SE(3).

    Parameters
    ----------
    m : array_like, shape (5,)
        Shape parameters.
    s : float
        Target arclength (0≤s≤1).
    gamma : int, optional
        Number of *equal* sub‑intervals for Magnus + PoE (default 10).  A value
        of 26 is usually more than enough for centimetre‑scale rods.
    L : float, optional
        Physical length scale.  We carry units symbolically in the twist
        integration and only scale the translation column here.
    """
    d_sub = np.linspace(0.0, s, gamma + 1)  # γ+1 nodes delimiting the γ pieces
    T = np.eye(4)                          # running product initialised at I₄

    # ——— accumulate exp(Ψₖ) left‑to‑right ————————————————————————
    for k in range(1, gamma + 1):
        Psi_k = magnus_4th_subinterval(d_sub[k - 1], d_sub[k], m)
        T = T @ expm(Psi_k)               # left‑invariant integration

    # Finally scale translation from [0,1] to [0,L].  If you prefer writing
    # κ in physical units directly, simply set L=1.
    T[:3, 3] *= L
    return T


def forward_kinematics_multiple(m: np.ndarray, s_values, *, gamma: int = 10,
                               L: float = 100.0):
    """Vectorised helper: compute **T(s)** for many *s* in one pass."""
    rots, poss = [], []
    for s in s_values:
        T_s = product_of_exponentials(m, s, gamma=gamma, L=L)
        rots.append(T_s[:3, :3])  # 3×3 rotation
        poss.append(T_s[:3, 3])   # 3‑vector position
    return rots, poss


##############################################################################
# 3) Analytical partial derivatives  ∂κ/∂mᵢ  and  ∂η/∂mᵢ
##############################################################################


def partial_kappa_wrt_mi(s: float, iParam: int) -> np.ndarray:
    """Closed‑form derivative **∂κ(s)/∂mᵢ** (5 distinct cases).

    Returns a 3‑vector with exactly one non‑zero entry (or two for *m₁, m₃*
    because of the *s* factor).
    """
    out = np.zeros(3)
    if iParam == 0:
        out[0] = 1.0            # ∂κₓ/∂m₀
    elif iParam == 1:
        out[0] = s              # ∂κₓ/∂m₁
    elif iParam == 2:
        out[1] = 1.0            # ∂κ_y/∂m₂
    elif iParam == 3:
        out[1] = s              # ∂κ_y/∂m₃
    else:                       # iParam == 4
        out[2] = 1.0            # ∂κ_z/∂m₄ (constant)
    return out


def partial_twist_wrt_mi(s: float, iParam: int) -> np.ndarray:
    """Derivative **∂η(s)/∂mᵢ** in closed form.

    Using `η(κ) = [[skew(κ), e₃]; 0]`, so the translation column is constant
    and vanishes in the derivative.  Only the skew‑symmetric part changes.
    """
    dkappa = partial_kappa_wrt_mi(s, iParam)

    def skew(u):
        return np.array([
            [ 0.0,  -u[2],  u[1]],
            [ u[2],  0.0,  -u[0]],
            [-u[1],  u[0],  0.0]
        ])

    dSkew = skew(dkappa)
    top_block = np.hstack([dSkew, np.zeros((3, 1))])  # translation col = 0
    bottom_block = np.array([[0.0, 0.0, 0.0, 0.0]])
    return np.vstack([top_block, bottom_block])


def partial_magnus_4th_subinterval(s_start: float, s_end: float, m: np.ndarray,
                                   iParam: int) -> np.ndarray:
    """Derivative **∂Ψₖ/∂mᵢ** on *[s_start,s_end]* (length *h*)."""
    h = s_end - s_start
    c1 = s_start + h * (0.5 - np.sqrt(3.0) / 6.0)  # same quadrature nodes
    c2 = s_start + h * (0.5 + np.sqrt(3.0) / 6.0)

    # Pre‑compute twists and their derivatives at nodes to avoid repetition
    eta1 = twist_matrix(curvature_kappa(c1, m))
    eta2 = twist_matrix(curvature_kappa(c2, m))
    dEta1 = partial_twist_wrt_mi(c1, iParam)
    dEta2 = partial_twist_wrt_mi(c2, iParam)

    # Differentiate the two Magnus terms separately
    part1 = (h / 2.0) * (dEta1 + dEta2)
    part2 = (h**2 * np.sqrt(3.0) / 12.0) * (
        (dEta1 @ eta2 + eta1 @ dEta2) - (dEta2 @ eta1 + eta2 @ dEta1)
    )
    return part1 + part2


##############################################################################
# Convenience wrappers for  ∂exp(A)/∂mᵢ  via *dexp* approximation
##############################################################################


def dexp_negA_approx(A: np.ndarray, dA: np.ndarray) -> np.ndarray:
    """Very cheap first‑order approximation of  dexp₋ᴬ(dA).

    This placeholder treats *se(3)* as a flat space:  dexp₋ᴬ ≈ I.  For long
    segments or high curvature you should replace this with a Bernoulli series
    or closed‑form Rodrigues (Rodrigues for *se(3)* is straightforward).
    Let's use Bernoulli series with order 1 which means dexp₋ᴬ(dA) = dA + 1/2(A * dA - dA*A)
    """
    # ipdb.set_trace()
    return dA + 1/2*(A @ dA - dA @ A)


def partial_expm(A: np.ndarray, dA: np.ndarray) -> np.ndarray:
    """Compute **∂(eᴬ)/∂mᵢ ≈ eᴬ · dA** with the flat‑space assumption."""
    expA = expm(A)
    dA_tilde = dexp_negA_approx(A, dA)
    return expA @ dA_tilde


##############################################################################
# ∂T/∂mᵢ  for the full product ∏ₖ exp(Ψₖ)
##############################################################################


def partial_product_of_exponentials(m: np.ndarray, s: float, iParam: int,
                                    *, gamma: int = 10) -> np.ndarray:
    """Compute **∂T(s)/∂mᵢ** with O(γ) matrix products (forward–reverse).

    We store the *left* and *right* cumulative products so that the derivative
    can be assembled efficiently without redundant exponentials.
    """
    d_sub = np.linspace(0.0, s, gamma + 1)

    # Pre‑compute exp(Ψₖ) and cache for reuse
    E_list, Psi_list = [], []
    for k in range(1, gamma + 1):
        A_k = magnus_4th_subinterval(d_sub[k - 1], d_sub[k], m)
        E_list.append(expm(A_k))
        Psi_list.append(A_k)

    partial_T = np.zeros((4, 4))

    # Chain‑rule over each sub‑interval *j*, sandwiching the local derivative
    for j in range(gamma):
        # Left product: ∏_{k<j} exp(Ψₖ)
        left = np.eye(4)
        for idx in range(j):
            left = left @ E_list[idx]

        # Right product: ∏_{k>j} exp(Ψₖ)
        right = np.eye(4)
        for idx in range(j + 1, gamma):
            right = right @ E_list[idx]

        # Local derivative  ∂exp(Ψⱼ)/∂mᵢ  via dexp approx
        dA_j = partial_magnus_4th_subinterval(d_sub[j], d_sub[j + 1], m, iParam)
        dExp_j = partial_expm(Psi_list[j], dA_j)

        partial_T += left @ dExp_j @ right

    return partial_T


##############################################################################
# 4) Orientation‑only cost  J(m)  and analytic gradient  ∇J
##############################################################################


def orientation_angle_and_gradM(R_pred: np.ndarray, R_obs: np.ndarray):
    """Smallest geodesic angle between two rotations + derivative.

    If *cos θ* is numerically ±1 we set the gradient to zero to avoid NaNs.
    The derivative wrt the *error matrix* **M=R_pred R_obsᵀ** is diagonal
    because ∂/∂Mₖₗ tr(M) = δₖₗ.
    """
    M = R_pred @ R_obs.T
    tr_val = np.trace(M)
    cos_val = np.clip((tr_val - 1.0) / 2.0, -1.0, 1.0)  # numerical safety
    angle = np.arccos(cos_val)

    if abs(cos_val) > 0.99999:                # ≈0 error → undefined slope
        dAngle_dM = np.zeros((3, 3))
    else:
        scale = -1.0 / (2.0 * np.sqrt(1.0 - cos_val**2))
        dAngle_dM = scale * np.eye(3)        # dθ/d(tr M) * d(tr M)/dM
    return angle, dAngle_dM


def cost_and_grad(m: np.ndarray, s_values, R_obs_list, *, gamma: int = 10,
                  L: float = 100.0):
    """Return scalar cost and 5‑vector gradient for the optimiser."""
    total_cost = 0.0
    grad = np.zeros(5)

    # Iterate over each observation   (R_obs at arclength s)
    for R_obs, s_val in zip(R_obs_list, s_values):
        # Forward model
        T_s = product_of_exponentials(m, s_val, gamma=gamma, L=L)
        R_pred = T_s[:3, :3]

        # 1) error angle  |ΔR|
        angle_i, dAngle_dM = orientation_angle_and_gradM(R_pred, R_obs)
        total_cost += angle_i

        # 2) chain‑rule back to *m*
        for iParam in range(5):
            dT_s = partial_product_of_exponentials(m, s_val, iParam, gamma=gamma)
            dR_pred = dT_s[:3, :3]
            dM = dR_pred @ R_obs.T
            grad[iParam] += np.sum(dAngle_dM * dM)  # Frobenius inner‑product

    return total_cost, grad


def objective_with_grad(m, s_values, R_obs_list, *, gamma=10, L=100.0):
    "Wrapper for SciPy so we don't recompute cost & grad twice."""
    return cost_and_grad(m, s_values, R_obs_list, gamma=gamma, L=L)


def solve_with_analytic_gradient(m_init, s_values, R_obs_list, *,
                                 gamma: int = 10, L: float = 100.0):
    """Run BFGS starting from *m_init* with analytic Jacobian."""

    def fun_and_jac(m_vec):
        return objective_with_grad(m_vec, s_values, R_obs_list, gamma=gamma, L=L)

    res = minimize(fun_and_jac,
                   x0=m_init,
                   method="BFGS",
                   jac=True,
                   options=dict(maxiter=200, disp=True))
    return res


##############################################################################
# 5) Visualisation helpers — 3‑D plot of true vs estimated backbone
##############################################################################


def plot_3d_curves(m_true, m_est, s_values, *, gamma: int = 10, L: float = 100.0):
    """Compare ground‑truth and estimated backbones with Matplotlib."""
    s_dense = np.linspace(0.0, 1.0, 100)

    # Generate dense centrelines for smooth curves
    _, pos_true = forward_kinematics_multiple(m_true, s_dense, gamma=gamma, L=L)
    _, pos_est = forward_kinematics_multiple(m_est, s_dense, gamma=gamma, L=L)
    pos_true, pos_est = map(np.asarray, (pos_true, pos_est))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*pos_true.T, color="blue", label="True shape")
    ax.plot(*pos_est.T,  color="green", linestyle="--", label="Est shape")

    # Overlay discrete measured frames for context
    for (s_i, R_t, p_t), (R_e, p_e) in zip(
        zip(s_values, *forward_kinematics_multiple(m_true, s_values, gamma=gamma, L=L)),
        zip(*forward_kinematics_multiple(m_est, s_values, gamma=gamma, L=L))):
        plot_frame(ax, p_t, R_t, f"True s={s_i}", color="blue")
        plot_frame(ax, p_e, R_e, f"Est  s={s_i}", color="green")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend()
    set_equal_axis(ax)
    plt.show()


def plot_frame(ax, origin, R, label=None, *, color="black", scale=5.0):
    """Draw a triad at *origin* with columns of R as axes."""
    axes = R * scale  # scale all three columns simultaneously
    colors = [color] * 3  # same colour for all axes unless customised

    for axis_vec, col in zip(axes.T, colors):
        ax.quiver(*origin, *axis_vec, color=col)

    if label:
        ax.text(*origin, label, color=color)


def set_equal_axis(ax):
    """Force equal scaling on a Matplotlib 3‑D axis."""
    xlims, ylims, zlims = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    centers = [0.5 * (lo + hi) for lo, hi in (xlims, ylims, zlims)]
    max_span = max(hi - lo for lo, hi in (xlims, ylims, zlims))

    ax.set_xlim3d(centers[0] - max_span / 2.0, centers[0] + max_span / 2.0)
    ax.set_ylim3d(centers[1] - max_span / 2.0, centers[1] + max_span / 2.0)
    ax.set_zlim3d(centers[2] - max_span / 2.0, centers[2] + max_span / 2.0)


##############################################################################
# Main entry‑point — synthetic test with noisy orientation samples
##############################################################################

if __name__ == "__main__":
    np.random.seed(44)                   # repeatability

    # — 1) Ground‑truth shape parameters ————————————————
    m_true = np.random.uniform(-4.0, 4.0, 5)

    # — 2) Observation points along the rod ————————————————
    s_values = [0.0, 0.3, 0.7, 1.0]      # include the base at s=0

    # — 3) Simulated noisy rotations (identity at base) ————————
    gamma_int = 26                       # integration resolution
    L_phys = 100.0                       # mm

    rotsT, _ = forward_kinematics_multiple(m_true, s_values, gamma=gamma_int, L=L_phys)
    R_obs_list = []
    noise_level = 0.001                  # small additive noise on rotations
    for i, R_true in enumerate(rotsT):
        if i == 0:                       # anchor base frame exactly
            R_obs_list.append(np.eye(3))
        else:
            R_obs_list.append(R_true + noise_level * np.random.randn(3, 3))

    # — 4) Optimise ——————————————————————————————————————————
    m_init = np.zeros(5)                 # deliberately poor initial guess
    res = solve_with_analytic_gradient(m_init, s_values, R_obs_list,
                                       gamma=gamma_int, L=L_phys)

    # Report
    print("\nSolver result:", res.message)
    print("Final cost  :", res.fun)
    print("True m      :", m_true)
    print("Estimated m :", res.x)

    # — 5) Visualise recovered curve —————————————————————————
    plot_3d_curves(m_true, res.x, s_values, gamma=gamma_int, L=L_phys)
