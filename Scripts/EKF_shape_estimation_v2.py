# -*- coding: utf-8 -*-
"""ekf_curvature_timeseries.py
Refined to support time-series data inputs with random-walk quaternion (rotation) measurements.
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

##############################################################################
# Core kinematics and EKF functions (unchanged from original)
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
    bottom_block = np.zeros((1,4))
    return np.vstack([top_block, bottom_block])

# Magnus 4th-order subinterval approximation
def magnus_4th_subinterval(s_start: float, s_end: float, m: np.ndarray) -> np.ndarray:
    h = s_end - s_start
    c1 = s_start + h * (0.5 - np.sqrt(3.0) / 6.0)
    c2 = s_start + h * (0.5 + np.sqrt(3.0) / 6.0)
    eta1 = twist_matrix(curvature_kappa(c1, m))
    eta2 = twist_matrix(curvature_kappa(c2, m))
    part1 = (h / 2.0) * (eta1 + eta2)
    part2 = (h**2 * np.sqrt(3.0) / 12.0) * (eta1 @ eta2 - eta2 @ eta1)
    return part1 + part2

# Product of exponentials for forward kinematics
def product_of_exponentials(m: np.ndarray, s: float, gamma: int = 10, L: float = 100.0):
    d_sub = np.linspace(0.0, s, gamma + 1)
    T = np.eye(4)
    for k in range(1, gamma + 1):
        Psi_k = magnus_4th_subinterval(d_sub[k-1], d_sub[k], m)
        T = T @ expm(Psi_k)
    T[:3, 3] *= L
    return T

# Vectorized forward kinematics for many s-values
def forward_kinematics_multiple(m: np.ndarray, s_values, *, gamma: int = 10, L: float = 100.0):
    rots, poss = [], []
    for s in s_values:
        T_s = product_of_exponentials(m, s, gamma=gamma, L=L)
        rots.append(T_s[:3, :3])
        poss.append(T_s[:3, 3])
    return rots, poss

# Compute measurement Jacobian and EKF update

def rotation_matrix_to_axis_angle(R):
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if np.isclose(angle, 0):
        return 0.0, np.array([0.0, 0.0, 1.0])
    axis = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
    axis /= (2 * np.sin(angle))
    return angle, axis

# Simplified partial derivatives for orientation
# (For brevity, use original implementations or suitable approximations)
def compute_measurement_jacobian(m, s_values, R_obs_list, gamma, L):
    # Placeholder: assume measurement theta = axis-angle error aggregated
    H = []
    y = []
    for s, R_obs in zip(s_values, R_obs_list):
        T = product_of_exponentials(m, s, gamma=gamma, L=L)
        R_pred = T[:3, :3]
        R_err = R_obs @ R_pred.T
        angle, axis = rotation_matrix_to_axis_angle(R_err)
        theta = angle * axis
        # Approximated gradient: numerical finite differences
        grad = np.zeros_like(m)
        eps = 1e-6
        for i in range(len(m)):
            m_eps = m.copy(); m_eps[i] += eps
            R_eps = product_of_exponentials(m_eps, s, gamma=gamma, L=L)[:3,:3]
            R_err_eps = R_obs @ R_eps.T
            angle_eps, axis_eps = rotation_matrix_to_axis_angle(R_err_eps)
            theta_eps = angle_eps * axis_eps
            grad[i] = ((theta_eps - theta) / eps).dot(axis)
        H.append(np.outer(axis, grad))
        y.append(theta)
    return np.vstack(H), np.concatenate(y)


def ekf_update(m_pred, P_pred, s_values, R_obs_list, Q, R, gamma, L):
    H, y = compute_measurement_jacobian(m_pred, s_values, R_obs_list, gamma, L)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    m_est = m_pred + K @ y
    P_est = (np.eye(len(m_pred)) - K @ H) @ P_pred
    return m_est, P_est

##############################################################################
# New: Time-series EKF on random-walk rotation measurements
##############################################################################

def skew(u: np.ndarray) -> np.ndarray:
    return np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])

def random_rotation_noise(noise_level: float) -> np.ndarray:
    """Generate a small random rotation matrix via exponential map."""
    w = noise_level * np.random.randn(3)
    return expm(skew(w))

if __name__ == "__main__":
    np.random.seed(44)
    # Ground-truth curvature parameters (static)
    m_true = np.random.uniform(-4.0, 4.0, 5)
    s_values = [0.0, 0.3, 0.7, 1.0]
    L_phys = 100.0
    gamma_int = 26
    noise_level = 0.005  # rotation noise magnitude per step
    num_steps = 500       # number of time steps

    # Pre-compute true rotations for each measurement index
    rots_true, _ = forward_kinematics_multiple(m_true, s_values, gamma=gamma_int, L=L_phys)

    # Generate time-series of noisy rotation measurements (random walk)
    measurement_seq = []
    prev_noisy = rots_true
    for t in range(num_steps):
        noisy_list = []
        for R_true in rots_true:
            R_noise = random_rotation_noise(noise_level)
            noisy_list.append(R_noise @ R_true)
        # ensure base frame remains identity
        noisy_list[0] = np.eye(3)
        measurement_seq.append(noisy_list)
    
    # EKF initialization
    m_est = np.zeros(5)
    P_est = np.eye(5) * 0.1
    Q = np.eye(5) * 1e-6
    R_block = np.eye(3 * len(s_values)) * (noise_level**2)

    # Storage for estimates
    m_estimates = np.zeros((num_steps, 5))

    # Run EKF over time-series
    for t in range(num_steps):
        # Predict (random-walk model)
        m_pred = m_est.copy()
        P_pred = P_est + Q
        # Update
        R_obs_list = measurement_seq[t]
        m_est, P_est = ekf_update(m_pred, P_pred,
                                  s_values, R_obs_list,
                                  Q, R_block,
                                  gamma_int, L_phys)
        m_estimates[t] = m_est

    # Plot parameter estimates over time
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(6, 10))
    time = np.arange(num_steps)
    for i in range(5):
        axs[i].plot(time, m_estimates[:, i], label=f'est m[{i}]')
        axs[i].axhline(m_true[i], linestyle='--', label='true')
        axs[i].legend(loc='upper right')
    axs[-1].set_xlabel('Time step')
    plt.tight_layout()
    plt.show()

    # Visualize final shape estimate vs. true
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # dense centreline for smooth curves
    s_dense = np.linspace(0.0, 1.0, 100)
    _, pos_true = forward_kinematics_multiple(m_true, s_dense, gamma=gamma_int, L=L_phys)
    _, pos_est = forward_kinematics_multiple(m_est, s_dense, gamma=gamma_int, L=L_phys)
    pos_true = np.asarray(pos_true)
    pos_est = np.asarray(pos_est)
    ax.plot(*pos_true.T, linestyle='-', label='True shape')
    ax.plot(*pos_est.T, linestyle='--', label='Estimated shape')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.show()
