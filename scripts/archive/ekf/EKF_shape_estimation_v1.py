# -*- coding: utf-8 -*-
"""ekf_curvature_estimation.py

Modified to use EKF for shape estimation with quaternion measurements.
Supports time-series random-walk data inputs.
"""

import numpy as np
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

##############################################################################
# Existing Functions (from original code)
##############################################################################

def curvature_kappa(s: float, m: np.ndarray) -> np.ndarray:
    kx = m[0] + m[1] * s
    ky = m[2] + m[3] * s
    kz = m[4]
    return np.array([kx, ky, kz])

def twist_matrix(kappa: np.ndarray) -> np.ndarray:
    def skew(u):
        return np.array([
            [0.0, -u[2], u[1]],
            [u[2], 0.0, -u[0]],
            [-u[1], u[0], 0.0]
        ])
    e3 = np.array([0.0, 0.0, 1.0])
    top_block = np.hstack([skew(kappa), e3.reshape(3, 1)])
    bottom_block = np.array([[0.0, 0.0, 0.0, 0.0]])
    return np.vstack([top_block, bottom_block])

def magnus_4th_subinterval(s_start: float, s_end: float, m: np.ndarray) -> np.ndarray:
    h = s_end - s_start
    c1 = s_start + h * (0.5 - np.sqrt(3.0) / 6.0)
    c2 = s_start + h * (0.5 + np.sqrt(3.0) / 6.0)
    eta1 = twist_matrix(curvature_kappa(c1, m))
    eta2 = twist_matrix(curvature_kappa(c2, m))
    part1 = (h / 2.0) * (eta1 + eta2)
    part2 = (h**2 * np.sqrt(3.0) / 12.0) * (eta1 @ eta2 - eta2 @ eta1)
    return part1 + part2

def product_of_exponentials(m: np.ndarray, s: float, gamma: int = 10, L: float = 100.0):
    d_sub = np.linspace(0.0, s, gamma + 1)
    T = np.eye(4)
    for k in range(1, gamma + 1):
        Psi_k = magnus_4th_subinterval(d_sub[k-1], d_sub[k], m)
        T = T @ expm(Psi_k)
    T[:3, 3] *= L
    return T

def forward_kinematics_multiple(m: np.ndarray, s_values, *, gamma: int = 10,
                               L: float = 100.0):
    rots, poss = [], []
    for s in s_values:
        T_s = product_of_exponentials(m, s, gamma=gamma, L=L)
        rots.append(T_s[:3, :3])
        poss.append(T_s[:3, 3])
    return rots, poss


def partial_product_of_exponentials(m: np.ndarray, s: float, iParam: int, gamma: int = 10):
    d_sub = np.linspace(0.0, s, gamma + 1)
    E_list = [expm(magnus_4th_subinterval(d_sub[k-1], d_sub[k], m)) for k in range(1, gamma+1)]
    Psi_list = [magnus_4th_subinterval(d_sub[k-1], d_sub[k], m) for k in range(1, gamma+1)]
    partial_T = np.zeros((4,4))
    for j in range(gamma):
        left = np.eye(4)
        for idx in range(j): left = left @ E_list[idx]
        right = np.eye(4)
        for idx in range(j+1, gamma): right = right @ E_list[idx]
        dA_j = partial_magnus_4th_subinterval(d_sub[j], d_sub[j+1], m, iParam)
        dExp_j = partial_expm(Psi_list[j], dA_j)
        partial_T += left @ dExp_j @ right
    return partial_T


def partial_magnus_4th_subinterval(s_start: float, s_end: float, m: np.ndarray, iParam: int) -> np.ndarray:
    h = s_end - s_start
    c1 = s_start + h * (0.5 - np.sqrt(3.0)/6.0)
    c2 = s_start + h * (0.5 + np.sqrt(3.0)/6.0)
    eta1 = twist_matrix(curvature_kappa(c1, m))
    eta2 = twist_matrix(curvature_kappa(c2, m))
    dEta1 = partial_twist_wrt_mi(c1, iParam)
    dEta2 = partial_twist_wrt_mi(c2, iParam)
    part1 = (h / 2.0) * (dEta1 + dEta2)
    part2 = (h**2 * np.sqrt(3.0)/12.0) * ((dEta1 @ eta2 + eta1 @ dEta2) - (dEta2 @ eta1 + eta2 @ dEta1))
    return part1 + part2


def partial_twist_wrt_mi(s: float, iParam: int) -> np.ndarray:
    dkappa = partial_kappa_wrt_mi(s, iParam)
    def skew(u):
        return np.array([
            [0.0, -u[2], u[1]],
            [u[2], 0.0, -u[0]],
            [-u[1], u[0], 0.0]
        ])
    dSkew = skew(dkappa)
    top_block = np.hstack([dSkew, np.zeros((3,1))])
    bottom_block = np.array([[0.0, 0.0, 0.0, 0.0]])
    return np.vstack([top_block, bottom_block])


def partial_kappa_wrt_mi(s: float, iParam: int) -> np.ndarray:
    out = np.zeros(3)
    if iParam == 0: out[0] = 1.0
    elif iParam == 1: out[0] = s
    elif iParam == 2: out[1] = 1.0
    elif iParam == 3: out[1] = s
    else: out[2] = 1.0
    return out


def dexp_negA_approx(A: np.ndarray, dA: np.ndarray) -> np.ndarray:
    return dA + 0.5 * (A @ dA - dA @ A)


def partial_expm(A: np.ndarray, dA: np.ndarray) -> np.ndarray:
    return expm(A) @ dexp_negA_approx(A, dA)

##############################################################################
# New EKF Functions
##############################################################################

def rotation_matrix_to_axis_angle(R_mat):
    angle = np.arccos(np.clip((np.trace(R_mat) - 1) / 2, -1.0, 1.0))
    if np.isclose(angle, 0):
        return 0.0, np.array([0.0, 0.0, 1.0])
    axis = np.array([R_mat[2,1] - R_mat[1,2], R_mat[0,2] - R_mat[2,0], R_mat[1,0] - R_mat[0,1]])
    axis /= (2 * np.sin(angle))
    return angle, axis


def compute_measurement_jacobian(m, s_values, R_obs_list, gamma, L):
    H, y = [], []
    for s, R_obs in zip(s_values, R_obs_list):
        T = product_of_exponentials(m, s, gamma=gamma, L=L)
        R_pred = T[:3, :3]
        R_err = R_obs @ R_pred.T
        angle, axis = rotation_matrix_to_axis_angle(R_err)
        theta = angle * axis
        grad = np.zeros(5)
        for i in range(5):
            dT = partial_product_of_exponentials(m, s, i, gamma=gamma)
            dR = dT[:3, :3]
            _, dAngle_dM = orientation_angle_and_gradM(R_pred, R_obs)
            grad[i] = np.sum(dAngle_dM * (dR @ R_obs.T))
        H.append(np.outer(axis, grad))
        y.append(theta)
    return np.vstack(H), np.concatenate(y)


def orientation_angle_and_gradM(R_pred, R_obs):
    M = R_pred @ R_obs.T
    tr_val = np.trace(M)
    cos_val = np.clip((tr_val - 1) / 2, -1.0, 1.0)
    if abs(cos_val) > 0.99999:
        return 0.0, np.zeros((3,3))
    scale = -1.0 / (2 * np.sqrt(1 - cos_val**2))
    return np.arccos(cos_val), scale * np.eye(3)


def ekf_update(m_pred, P_pred, s_values, R_obs_list, Q, R_mat, gamma, L):
    H, y = compute_measurement_jacobian(m_pred, s_values, R_obs_list, gamma, L)
    S = H @ P_pred @ H.T + R_mat
    K = P_pred @ H.T @ np.linalg.inv(S)
    m_est = m_pred + K @ y
    P_est = (np.eye(len(m_pred)) - K @ H) @ P_pred
    return m_est, P_est

##############################################################################
# Main Execution: Time-Series EKF
##############################################################################

if __name__ == "__main__":
    np.random.seed(0)
    # Ground-truth parameters
    m_true = np.random.uniform(-4.0, 4.0, 5)
    s_values = [0.0, 0.3, 0.7, 1.0]
    gamma_int, L_phys = 26, 100.0
    noise_level = 0.01

    # Precompute true rotations for measurement model
    rots_true, _ = forward_kinematics_multiple(m_true, s_values, gamma=gamma_int, L=L_phys)

    # Simulate time-series with small random-walk noise on each frame
    N_steps = 50
    R_obs_series = []
    for t in range(N_steps):
        R_obs_list = []
        for R_true in rots_true:
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            angle = noise_level * np.random.randn()
            R_noise = R.from_rotvec(axis * angle).as_matrix()
            R_obs_list.append(R_noise @ R_true)
        R_obs_series.append(R_obs_list)

    # EKF initialization
    m_est = np.zeros(5)
    P_est = np.eye(5) * 0.1
    Q = np.eye(5) * 1e-6
    R_mat = np.eye(3 * len(s_values)) * (noise_level ** 2)

    # Storage for history
    m_history = np.zeros((N_steps+1, 5))
    m_history[0] = m_est

    # Time-series EKF loop
    for t in range(N_steps):
        m_pred = m_est.copy()
        P_pred = P_est + Q
        m_est, P_est = ekf_update(m_pred, P_pred, s_values, R_obs_series[t], Q, R_mat, gamma_int, L_phys)
        m_history[t+1] = m_est

    # Plot parameter convergence
    plt.figure()
    for i in range(5):
        plt.plot(np.arange(N_steps+1), m_history[:, i], label=f"m[{i}]")
        plt.hlines(m_true[i], 0, N_steps, colors='k', linestyles='--')
    plt.xlabel("Time step")
    plt.ylabel("Parameter value")
    plt.legend()
    plt.title("EKF Parameter Estimates Over Time Series")
    plt.show()

    # Final 3D backbone comparison
    plot_3d_curves(m_true, m_est, s_values, gamma=gamma_int, L=L_phys)

