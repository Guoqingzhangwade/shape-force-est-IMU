import numpy as np
from scipy.linalg import expm, expm_frechet


E3 = np.array([0.0, 0.0, 1.0])


def _trapezoid(y, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def hat(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def vee(se3_mat):
    omega_hat = se3_mat[:3, :3]
    v = se3_mat[:3, 3]
    omega = np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])
    return np.hstack([omega, v])


def monomial_basis(s, order):
    return np.array([s ** i for i in range(order + 1)])


def gram_matrix(order):
    idx = np.arange(order + 1, dtype=float)
    return 1.0 / (idx[:, None] + idx[None, :] + 1.0)


def total_params(order_x, order_y, order_z):
    return (order_x + 1) + (order_y + 1) + (order_z + 1)


def split_modal_vector(m, order_x, order_y, order_z):
    ox = order_x + 1
    oy = order_y + 1
    mx = m[:ox]
    my = m[ox:ox + oy]
    mz = m[ox + oy:ox + oy + (order_z + 1)]
    return mx, my, mz


def param_axis_and_power(i_param, order_x, order_y, order_z):
    ox = order_x + 1
    oy = order_y + 1
    if i_param < ox:
        return 0, i_param
    if i_param < ox + oy:
        return 1, i_param - ox
    return 2, i_param - ox - oy


def curvature_from_modal(s, m, order_x, order_y, order_z):
    mx, my, mz = split_modal_vector(m, order_x, order_y, order_z)
    kx = monomial_basis(s, order_x) @ mx
    ky = monomial_basis(s, order_y) @ my
    kz = monomial_basis(s, order_z) @ mz
    return np.array([kx, ky, kz])


def partial_kappa_wrt_param(s, i_param, order_x, order_y, order_z):
    axis, power = param_axis_and_power(i_param, order_x, order_y, order_z)
    dkappa = np.zeros(3)
    dkappa[axis] = s ** power
    return dkappa


def twist_matrix(kappa):
    e3 = np.array([0.0, 0.0, 1.0])
    top_left = hat(kappa)
    top_block = np.hstack([top_left, e3.reshape(3, 1)])
    bottom_block = np.array([[0.0, 0.0, 0.0, 0.0]])
    return np.vstack([top_block, bottom_block])


def magnus_4th_subinterval(s_start, s_end, m, order_x, order_y, order_z):
    h = s_end - s_start
    c1 = s_start + h * (0.5 - np.sqrt(3) / 6)
    c2 = s_start + h * (0.5 + np.sqrt(3) / 6)

    kappa1 = curvature_from_modal(c1, m, order_x, order_y, order_z)
    kappa2 = curvature_from_modal(c2, m, order_x, order_y, order_z)
    eta1 = twist_matrix(kappa1)
    eta2 = twist_matrix(kappa2)

    part1 = (h / 2.0) * (eta1 + eta2)
    part2 = (h**2 * np.sqrt(3) / 12.0) * (eta1 @ eta2 - eta2 @ eta1)
    return part1 + part2


def product_of_exponentials(m, s, gamma, L, order_x, order_y, order_z):
    d_sub = np.linspace(0.0, s, gamma + 1)
    T = np.eye(4)
    for k in range(1, gamma + 1):
        Psi_k = magnus_4th_subinterval(
            d_sub[k - 1], d_sub[k], m, order_x, order_y, order_z
        )
        T = T @ expm(Psi_k)
    T[:3, 3] *= L
    return T


def partial_twist_wrt_param(s, i_param, order_x, order_y, order_z):
    dkappa = partial_kappa_wrt_param(s, i_param, order_x, order_y, order_z)
    dskew = hat(dkappa)
    top_block = np.hstack([dskew, np.zeros((3, 1))])
    bottom_block = np.array([[0.0, 0.0, 0.0, 0.0]])
    return np.vstack([top_block, bottom_block])


def partial_magnus_4th_subinterval(s_start, s_end, m, i_param,
                                   order_x, order_y, order_z):
    h = s_end - s_start
    c1 = s_start + h * (0.5 - np.sqrt(3) / 6)
    c2 = s_start + h * (0.5 + np.sqrt(3) / 6)

    eta1 = twist_matrix(curvature_from_modal(c1, m, order_x, order_y, order_z))
    eta2 = twist_matrix(curvature_from_modal(c2, m, order_x, order_y, order_z))

    d_eta1 = partial_twist_wrt_param(c1, i_param, order_x, order_y, order_z)
    d_eta2 = partial_twist_wrt_param(c2, i_param, order_x, order_y, order_z)

    part1 = (h / 2.0) * (d_eta1 + d_eta2)
    part2 = (h**2 * np.sqrt(3) / 12.0) * (
        (d_eta1 @ eta2 + eta1 @ d_eta2) - (d_eta2 @ eta1 + eta2 @ d_eta1)
    )
    return part1 + part2


def partial_expm(A, dA):
    _, dExp = expm_frechet(A, dA, compute_expm=True)
    return dExp


def partial_product_of_exponentials(m, s, i_param, gamma, L,
                                    order_x, order_y, order_z):
    d_sub = np.linspace(0.0, s, gamma + 1)
    E_list = []
    Psi_list = []
    for k in range(1, gamma + 1):
        A_k = magnus_4th_subinterval(
            d_sub[k - 1], d_sub[k], m, order_x, order_y, order_z
        )
        E_list.append(expm(A_k))
        Psi_list.append(A_k)

    partial_T = np.zeros((4, 4))
    for j in range(gamma):
        left = np.eye(4)
        for idx in range(j):
            left = left @ E_list[idx]
        dA_j = partial_magnus_4th_subinterval(
            d_sub[j], d_sub[j + 1], m, i_param, order_x, order_y, order_z
        )
        dExp_j = partial_expm(Psi_list[j], dA_j)
        right = np.eye(4)
        for idx in range(j + 1, gamma):
            right = right @ E_list[idx]
        partial_T += left @ dExp_j @ right
    partial_T[:3, 3] *= L
    return partial_T


def body_jacobian_at_s(m, s, gamma, L, order_x, order_y, order_z):
    n_params = total_params(order_x, order_y, order_z)
    T = product_of_exponentials(m, s, gamma, L, order_x, order_y, order_z)
    T_inv = np.linalg.inv(T)
    J = np.zeros((6, n_params))
    for i_param in range(n_params):
        dT = partial_product_of_exponentials(
            m, s, i_param, gamma, L, order_x, order_y, order_z
        )
        Vb_hat = T_inv @ dT
        J[:, i_param] = vee(Vb_hat)
    return J, T


def cable_tangent_body(kappa, r_i, L):
    """
    Physical derivative of the tendon/cable point with respect to normalized
    arclength s, expressed in the local body frame.

    With s = l/L and normalized curvature kappa(s) = L*kappa_phys(l),

        R.T d w_i / ds = L e3 + kappa(s) x r_i

    for constant cross-section routing r_i'(s) = 0.
    """
    return L * E3 + hat(kappa) @ r_i


def cable_dir_body(kappa, r_i, L=1.0):
    """
    Legacy helper name for cable_tangent_body().

    New code should call cable_tangent_body(kappa, r_i, L) so the normalized
    arclength scaling is explicit.
    """
    return cable_tangent_body(kappa, r_i, L)


def cable_length_physical(
    m,
    r_i,
    L,
    order_x,
    order_y,
    order_z,
    n_int=400,
):
    """
    Physical tendon/cable length:

        ell_i = integral_0^1 ||L e3 + kappa(s) x r_i|| ds

    This is mainly for finite-difference convention checks.
    """
    s_vals = np.linspace(0.0, 1.0, n_int + 1)
    integrand = np.zeros(len(s_vals))
    for idx, s in enumerate(s_vals):
        kappa = curvature_from_modal(s, m, order_x, order_y, order_z)
        tangent = cable_tangent_body(kappa, r_i, L)
        integrand[idx] = np.linalg.norm(tangent)
    return float(_trapezoid(integrand, s_vals, axis=0))


def pull_jacobian(m, r_list, L, order_x, order_y, order_z, n_int=200):
    """
    Modal tendon-pull/shortening Jacobian:

        J_qm = d q/dm = -d ell/dm.

    This is the object used in:

        b_w = gradU - J_qm.T @ tau.

    Wrench/tension convention:
        positive tau_i performs positive virtual work when q_i increases.
    """
    n_params = total_params(order_x, order_y, order_z)
    s_vals = np.linspace(0.0, 1.0, n_int + 1)
    J_qm = np.zeros((len(r_list), n_params))
    for i_r, r_i in enumerate(r_list):
        integrand = np.zeros((len(s_vals), n_params))
        for idx, s in enumerate(s_vals):
            kappa = curvature_from_modal(s, m, order_x, order_y, order_z)
            tangent = cable_tangent_body(kappa, r_i, L)
            t_hat = tangent / np.linalg.norm(tangent)
            moment_arm = np.cross(r_i, t_hat)
            for i_param in range(n_params):
                axis, power = param_axis_and_power(
                    i_param, order_x, order_y, order_z
                )
                integrand[idx, i_param] = moment_arm[axis] * (s ** power)
        J_qm[i_r, :] = -_trapezoid(integrand, s_vals, axis=0)
    return J_qm


def cable_jacobian(m, r_list, L, order_x, order_y, order_z, n_int=200):
    """
    Legacy alias.

    Historically this function name was used in the repo, but it returns the
    tendon-pull/shortening Jacobian J_qm, not the true cable-length Jacobian
    J_lm. New code should call pull_jacobian().
    """
    return pull_jacobian(m, r_list, L, order_x, order_y, order_z, n_int)


def cable_length_jacobian(m, r_list, L, order_x, order_y, order_z, n_int=200):
    """
    True tendon/cable-length Jacobian:

        J_lm = d ell/dm = -J_qm.

    This is provided only for clarity or diagnostics.
    """
    return -pull_jacobian(m, r_list, L, order_x, order_y, order_z, n_int)


def elastic_energy_gradient(m, EIx, EIy, GJ, L, order_x, order_y, order_z):
    mx, my, mz = split_modal_vector(m, order_x, order_y, order_z)
    Mx = gram_matrix(order_x)
    My = gram_matrix(order_y)
    Mz = gram_matrix(order_z)
    grad_x = (EIx / L) * (Mx @ mx)
    grad_y = (EIy / L) * (My @ my)
    grad_z = (GJ / L) * (Mz @ mz)
    return np.hstack([grad_x, grad_y, grad_z])


def adjoint(T):
    R = T[:3, :3]
    p = T[:3, 3]
    p_hat = hat(p)
    upper = np.hstack([R, np.zeros((3, 3))])
    lower = np.hstack([p_hat @ R, R])
    return np.vstack([upper, lower])


def generalized_modal_load(gradU, J_qm, tau):
    """
    Generalized modal load attributed to the external wrench:

        b_w = gradU - J_qm.T @ tau.

    J_qm is the tendon pull/shortening Jacobian d q/dm.
    """
    return gradU - J_qm.T @ tau


def solve_wrench(J_vb_m, J_qm, gradU, tau, rcond=1e-8):
    """
    Nominal minimum-norm full-6D wrench baseline.

    This is not generally a unique physical wrench estimate when J_vb_m.T is
    rank deficient. Use load-subspace solvers for the main constrained
    estimator.
    """
    A = J_vb_m.T
    b = generalized_modal_load(gradU, J_qm, tau)
    return np.linalg.pinv(A, rcond=rcond) @ b
