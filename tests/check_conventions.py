from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CRT_DIR = ROOT / "src" / "shape_force_est_imu" / "crt"
sys.path.insert(0, str(CRT_DIR))

import virtual_work as vw  # noqa: E402


def trapezoid(y, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def independent_cable_length(m, r_i, L, order_x, order_y, order_z, n_int=800):
    e3 = np.array([0.0, 0.0, 1.0])
    s_vals = np.linspace(0.0, 1.0, n_int + 1)
    integrand = np.zeros(len(s_vals))
    for idx, s in enumerate(s_vals):
        kappa = vw.curvature_from_modal(s, m, order_x, order_y, order_z)
        tangent = L * e3 + np.cross(kappa, r_i)
        integrand[idx] = np.linalg.norm(tangent)
    return float(trapezoid(integrand, s_vals, axis=0))


def check_pull_jacobian_fd():
    L = 0.10
    Rp = 0.008
    order_x, order_y, order_z = 1, 1, 0
    m = np.array([0.4, -0.2, 0.3, 0.1, 0.05])
    r_list = [
        np.array([Rp, 0.0, 0.0]),
        np.array([0.0, Rp, 0.0]),
        np.array([-Rp, 0.0, 0.0]),
        np.array([0.0, -Rp, 0.0]),
    ]

    eps = 1e-6
    n_int = 800
    J_qm = vw.pull_jacobian(
        m, r_list, L, order_x, order_y, order_z, n_int=n_int
    )
    J_fd = np.zeros_like(J_qm)
    for i_r, r_i in enumerate(r_list):
        for j in range(len(m)):
            dm = np.zeros_like(m)
            dm[j] = eps
            ell_p = independent_cable_length(
                m + dm, r_i, L, order_x, order_y, order_z, n_int=n_int
            )
            ell_m = independent_cable_length(
                m - dm, r_i, L, order_x, order_y, order_z, n_int=n_int
            )
            J_fd[i_r, j] = -(ell_p - ell_m) / (2.0 * eps)

    diff = J_qm - J_fd
    max_abs = float(np.max(np.abs(diff)))
    max_rel = float(np.max(np.abs(diff) / np.maximum(np.abs(J_fd), 1e-12)))
    np.testing.assert_allclose(J_qm, J_fd, rtol=5e-5, atol=1e-8)
    print(f"[pull_jacobian FD] max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")


def check_body_jacobian_fd():
    L = 0.10
    order_x, order_y, order_z = 1, 1, 0
    gamma = 10
    m = np.array([0.4, -0.2, 0.3, 0.1, 0.05])
    s_tip = 1.0
    eps = 1e-6

    J_vbm, T0 = vw.body_jacobian_at_s(
        m, s_tip, gamma, L, order_x, order_y, order_z
    )
    T0_inv = np.linalg.inv(T0)
    J_fd = np.zeros_like(J_vbm)
    for j in range(len(m)):
        dm = np.zeros_like(m)
        dm[j] = eps
        Tp = vw.product_of_exponentials(
            m + dm, s_tip, gamma, L, order_x, order_y, order_z
        )
        Tm = vw.product_of_exponentials(
            m - dm, s_tip, gamma, L, order_x, order_y, order_z
        )
        dT = (Tp - Tm) / (2.0 * eps)
        Vb_hat = T0_inv @ dT
        J_fd[:, j] = vw.vee(Vb_hat)

    diff = J_vbm - J_fd
    max_abs = float(np.max(np.abs(diff)))
    max_rel = float(np.max(np.abs(diff) / np.maximum(np.abs(J_fd), 1e-12)))
    np.testing.assert_allclose(J_vbm, J_fd, rtol=5e-4, atol=5e-6)
    print(f"[body_jacobian FD] max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
    return J_vbm, T0


def check_known_direction_subspace(J_vbm):
    d_body = np.array([0.2, -0.4, 0.9])
    d_body = d_body / np.linalg.norm(d_body)
    S = np.zeros((6, 1))
    S[3:6, 0] = d_body

    A = J_vbm.T
    lambda_true = np.array([0.37])
    b = A @ S @ lambda_true
    lambda_hat = np.linalg.pinv(A @ S) @ b

    np.testing.assert_allclose(lambda_hat, lambda_true, rtol=1e-10, atol=1e-12)
    err = float(np.linalg.norm(lambda_hat - lambda_true))
    print(f"[known-direction subspace] err={err:.3e}")


def check_tip_origin_wrench_transform(T_tip):
    R = T_tip[:3, :3]
    block = np.zeros((6, 6))
    block[:3, :3] = R
    block[3:, 3:] = R

    F_body = np.array([0.01, -0.02, 0.03, 0.4, -0.5, 0.6])
    F_world_tip = block @ F_body

    np.testing.assert_allclose(F_world_tip[:3], R @ F_body[:3])
    np.testing.assert_allclose(F_world_tip[3:], R @ F_body[3:])
    print("[tip-origin wrench transform] passed")


def main():
    check_pull_jacobian_fd()
    J_vbm, T_tip = check_body_jacobian_fd()
    check_known_direction_subspace(J_vbm)
    check_tip_origin_wrench_transform(T_tip)
    print("All convention checks passed.")


if __name__ == "__main__":
    main()
