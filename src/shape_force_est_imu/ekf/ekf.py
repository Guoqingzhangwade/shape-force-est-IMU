"""Measurement model and EKF update helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from .kinematics import forward_kinematics_multiple, rotation_from_poE
from .quat import q_fix_sign, quat_error, J_theta_q, J_q_q, J_q_R
from .jacobians import JRm_block


def measurement_jacobian(
    m: np.ndarray,
    s_vals,
    q_obs,
    *,
    gamma: int = 26,
    L: float = 100.0,
    model: str = "5d",
    e3: np.ndarray | None = None,
    mode: str = "exact",
):
    """Stacked measurement Jacobian and innovation vector."""
    H_blocks, y_blocks = [], []
    R_pred, _ = forward_kinematics_multiple(m, s_vals, gamma=gamma, L=L, model=model, e3=e3)
    q_pred = [q_fix_sign(R.from_matrix(Rp).as_quat()) for Rp in R_pred]

    for s_i, q_m, q_p, Rp in zip(s_vals, q_obs, q_pred, R_pred):
        q_m = q_fix_sign(np.asarray(q_m))
        theta = quat_error(q_m, q_p)
        H_blocks.append(J_theta_q(q_m) @ J_q_q(q_m) @ J_q_R(Rp) @ JRm_block(
            m, s_i, gamma=gamma, model=model, e3=e3, mode=mode
        ))
        y_blocks.append(-theta)
    return np.vstack(H_blocks), np.hstack(y_blocks)


def ekf_update(
    m: np.ndarray,
    P: np.ndarray,
    s_vals,
    q_obs,
    R_cov: np.ndarray,
    *,
    gamma: int = 26,
    L: float = 100.0,
    model: str = "5d",
    e3: np.ndarray | None = None,
    mode: str = "exact",
    iters: int = 1,
):
    """Iterated EKF update for quaternion measurements."""
    for _ in range(iters):
        H, y = measurement_jacobian(
            m, s_vals, q_obs, gamma=gamma, L=L, model=model, e3=e3, mode=mode
        )
        S = H @ P @ H.T + R_cov
        K = P @ H.T @ np.linalg.inv(S)
        m = m + K @ y
        P = (np.eye(len(m)) - K @ H) @ P
    return m, P


def theta_from_measurement(
    q_meas: np.ndarray,
    m: np.ndarray,
    s: float,
    *,
    gamma: int = 10,
    L: float = 100.0,
    model: str = "5d",
    e3: np.ndarray | None = None,
) -> np.ndarray:
    """Minimal orientation error for one measurement."""
    R_pred = rotation_from_poE(m, s, gamma=gamma, L=L, model=model, e3=e3)
    q_pred = q_fix_sign(R.from_matrix(R_pred).as_quat())
    q_meas = q_fix_sign(np.asarray(q_meas))
    return quat_error(q_meas, q_pred)


def numeric_jacobian_theta(
    q_meas: np.ndarray,
    m: np.ndarray,
    s: float,
    *,
    gamma: int = 10,
    L: float = 100.0,
    model: str = "5d",
    e3: np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Finite-difference Jacobian of theta wrt m."""
    J = np.zeros((3, m.size))
    for i in range(m.size):
        m_p, m_m = m.copy(), m.copy()
        m_p[i] += eps
        m_m[i] -= eps
        J[:, i] = (
            theta_from_measurement(q_meas, m_p, s, gamma=gamma, L=L, model=model, e3=e3)
            - theta_from_measurement(q_meas, m_m, s, gamma=gamma, L=L, model=model, e3=e3)
        ) / (2 * eps)
    return J
