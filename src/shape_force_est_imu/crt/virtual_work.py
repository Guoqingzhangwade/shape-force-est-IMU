import numpy as np
from scipy.linalg import expm


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
    expA = expm(A)
    return expA @ dA


def partial_product_of_exponentials(m, s, i_param, gamma,
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
    return partial_T


def body_jacobian_at_s(m, s, gamma, L, order_x, order_y, order_z):
    n_params = total_params(order_x, order_y, order_z)
    T = product_of_exponentials(m, s, gamma, L, order_x, order_y, order_z)
    T_inv = np.linalg.inv(T)
    J = np.zeros((6, n_params))
    for i_param in range(n_params):
        dT = partial_product_of_exponentials(
            m, s, i_param, gamma, order_x, order_y, order_z
        )
        Vb_hat = T_inv @ dT
        J[:, i_param] = vee(Vb_hat)
    return J, T


def cable_dir_body(kappa, r_i):
    return np.array([0.0, 0.0, 1.0]) + hat(kappa) @ r_i


def cable_jacobian(m, r_list, L, order_x, order_y, order_z, n_int=200):
    n_params = total_params(order_x, order_y, order_z)
    s_vals = np.linspace(0.0, 1.0, n_int + 1)
    J = np.zeros((len(r_list), n_params))
    for i_r, r_i in enumerate(r_list):
        integrand = np.zeros((len(s_vals), n_params))
        for idx, s in enumerate(s_vals):
            kappa = curvature_from_modal(s, m, order_x, order_y, order_z)
            t_dir = cable_dir_body(kappa, r_i)
            t_hat = t_dir / np.linalg.norm(t_dir)
            moment_arm = np.cross(r_i, t_hat)
            for i_param in range(n_params):
                axis, power = param_axis_and_power(
                    i_param, order_x, order_y, order_z
                )
                integrand[idx, i_param] = moment_arm[axis] * (s ** power)
        J[i_r, :] = L * np.trapz(integrand, s_vals, axis=0)
    return J


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


def solve_wrench(J_vb_m, J_lm, gradU, tau, rcond=1e-8):
    A = J_vb_m.T
    b = gradU - J_lm.T @ tau
    return np.linalg.pinv(A, rcond=rcond) @ b
