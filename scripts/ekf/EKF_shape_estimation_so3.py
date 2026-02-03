# -*- coding: utf-8 -*-
"""
EKF shape estimation with SO(3) measurement residual.

Residual per IMU:
  R_e = R_meas * R_pred^T
  r   = Log(R_e)^vee

Jacobian per IMU (analytic):
  H[:, p] = -Jr_inv(r) * (dR_dm * R_pred^T)^vee
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R


def skew(u):
    return np.array([[0.0, -u[2], u[1]],
                     [u[2], 0.0, -u[0]],
                     [-u[1], u[0], 0.0]])


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def phi(s):
    return np.array([[1, s, 0, 0, 0],
                     [0, 0, 1, s, 0],
                     [0, 0, 0, 0, 1]])


def twist(k, e3):
    return np.block([[skew(k), e3[:, None]],
                     [np.zeros((1, 3)), 0.0]])


def magnus_psi(s_i, h, m, e3):
    xi = np.array([0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6])
    c1, c2 = s_i - h + xi * h
    k1, k2 = phi(c1) @ m, phi(c2) @ m
    e1, e2 = twist(k1, e3), twist(k2, e3)
    return (h / 2) * (e1 + e2) + (np.sqrt(3) / 12) * h ** 2 * (e1 @ e2 - e2 @ e1)


def fwd_rotation(m, s, e3, gamma):
    T = np.eye(4)
    h = s / gamma
    for k in range(1, gamma + 1):
        T = T @ expm(magnus_psi(k * h, h, m, e3))
    return T[:3, :3]


def so3_log(Rm):
    return R.from_matrix(Rm).as_rotvec()


def jr_inv_so3(r):
    theta = np.linalg.norm(r)
    if theta < 1e-8:
        A = skew(r)
        return np.eye(3) + 0.5 * A + (1.0 / 12.0) * (A @ A)
    A = skew(r)
    a = (1.0 / (theta ** 2)) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) + 0.5 * A + a * (A @ A)


def compute_R_m_derivative(m_coeffs, s, gamma, e3):
    Nm = len(m_coeffs)
    Ns = gamma
    h = s / Ns
    Psi, e_Psi = [], []
    T_before = [np.eye(4)]
    T_after = [np.eye(4)] * (Ns + 1)
    xi = np.array([0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6])
    dPsi_dm = [np.zeros((4, 4, Nm)) for _ in range(Ns)]
    de_Psi_dm = [np.zeros((4, 4, Nm)) for _ in range(Ns)]

    for k in range(1, Ns + 1):
        s_i = k * h
        Psi_k = magnus_psi(s_i, h, m_coeffs, e3)
        Psi.append(Psi_k)
        e_Psi_k = expm(Psi_k)
        e_Psi.append(e_Psi_k)
        T_before.append(T_before[-1] @ e_Psi_k)

    T_after[Ns] = np.eye(4)
    for k in range(Ns - 1, 0, -1):
        T_after[k] = e_Psi[k] @ T_after[k + 1]

    for k in range(1, Ns + 1):
        s_i = k * h
        c1 = s_i - h + xi[0] * h
        c2 = s_i - h + xi[1] * h
        phi_c1 = phi(c1)
        phi_c2 = phi(c2)
        kappa_1 = phi_c1 @ m_coeffs
        kappa_2 = phi_c2 @ m_coeffs
        eta_1 = twist(kappa_1, e3)
        eta_2 = twist(kappa_2, e3)
        for i in range(Nm):
            dkappa_1 = phi_c1[:, i]
            dkappa_2 = phi_c2[:, i]
            deta_1 = twist(dkappa_1, np.zeros(3))
            deta_2 = twist(dkappa_2, np.zeros(3))
            comm_deta = (deta_1 @ eta_2 - eta_2 @ deta_1) + (eta_1 @ deta_2 - deta_2 @ eta_1)
            dPsi_k = (h / 2) * (deta_1 + deta_2) + (np.sqrt(3) / 12) * h ** 2 * comm_deta
            dPsi_dm[k - 1][:, :, i] = dPsi_k

    for k in range(Ns):
        Psi_k = Psi[k]
        e_Psi_k = e_Psi[k]
        for i in range(Nm):
            dPsi_k = dPsi_dm[k][:, :, i]
            # 1st-order dexp approximation: dexp(A)·dA ≈ dA + 0.5[A, dA]
            dexpinv = dPsi_k + 0.5 * (Psi_k @ dPsi_k - dPsi_k @ Psi_k)
            de_Psi_dm[k][:, :, i] = e_Psi_k @ dexpinv

    dT_dm = np.zeros((4, 4, Nm))
    for i in range(Nm):
        for k in range(Ns):
            dT_dm[:, :, i] += T_before[k] @ de_Psi_dm[k][:, :, i] @ T_after[k + 1]
    return dT_dm[:3, :3, :]


def measurement_jacobian_so3(m, s_vals, R_meas_list, e3, gamma):
    H_blocks, r_blocks = [], []
    for s_i, R_meas in zip(s_vals, R_meas_list):
        R_pred = fwd_rotation(m, s_i, e3, gamma)
        R_e = R_meas @ R_pred.T
        r = so3_log(R_e)
        Jr_inv = jr_inv_so3(r)
        dR_dm = compute_R_m_derivative(m, s_i, gamma, e3)
        H_i = np.zeros((3, m.size))
        for p in range(m.size):
            dR = dR_dm[:, :, p]
            A = dR @ R_pred.T
            A = 0.5 * (A - A.T)
            H_i[:, p] = -Jr_inv @ vee(A)
        H_blocks.append(H_i)
        r_blocks.append(r)
    return np.vstack(H_blocks), np.hstack(r_blocks)


if __name__ == "__main__":
    np.random.seed(44)
    s_vals = [0.2, 0.6, 1.0]
    m_true = np.random.uniform(-4, 4, 5)
    L_phys = 100.0
    gamma_int = 26
    sigma_rot = 0.005
    T_steps = 500
    e3 = np.array([0, 0, L_phys])

    R_true = [fwd_rotation(m_true, s, e3, gamma_int) for s in s_vals]

    meas_seq = []
    for _ in range(T_steps):
        Rs = []
        for Rg in R_true:
            dw = sigma_rot * np.random.randn(3)
            Rs.append(expm(skew(dw)) @ Rg)
        meas_seq.append(Rs)

    m_est = np.zeros(5)
    P_est = np.eye(5) * 0.2
    Q = np.eye(5) * 1e-6
    R_cov = np.eye(3 * len(s_vals)) * (3 * sigma_rot) ** 2

    hist = np.zeros((T_steps, 5))
    for t in range(T_steps):
        P_pred = P_est + Q
        H, r = measurement_jacobian_so3(m_est, s_vals, meas_seq[t], e3, gamma_int)
        R_big = R_cov
        S = H @ P_pred @ H.T + R_big
        K = P_pred @ H.T @ inv(S)
        m_est = m_est + K @ (-r)
        P_est = (np.eye(5) - K @ H) @ P_pred
        hist[t] = m_est

    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(6, 10))
    for i in range(5):
        axs[i].plot(hist[:, i], label=f"est m[{i}]")
        axs[i].axhline(m_true[i], ls="--", label="true")
        axs[i].legend(loc="upper right")
    axs[-1].set_xlabel("time step")
    plt.tight_layout()
    plt.show()
