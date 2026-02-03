# -*- coding: utf-8 -*-
"""
ekf_curvature_timeseries_quat_v5.py
-----------------------------------
Quaternion EKF for 5-parameter curvature model (analytic Jacobian fixed).

Key fixes vs v4:
  - Use a consistent quaternion chain rule for theta = 2 * vec(q_err).
  - Jacobian is w.r.t. q_pred (not q_meas), with proper conjugation.
  - Uses xyzw convention throughout (matches shape_force_est_imu.ekf).
"""

# ───────────── Imports ─────────────
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from shape_force_est_imu.ekf import (  # noqa: E402
    skew,
    q_fix_sign,
    quat_mul,
    quat_conj,
    forward_kinematics_multiple,
    JRm_block,
    J_q_R,
)


def left_quat_matrix_xyzw(q: np.ndarray) -> np.ndarray:
    """
    Left-multiplication matrix L(q) for xyzw quaternions.
    Returns L such that q ⊗ p = L(q) p, with p in xyzw order.
    """
    qv, q0 = q[:3], q[3]
    qv_hat = np.array(
        [[0.0, -qv[2], qv[1]], [qv[2], 0.0, -qv[0]], [-qv[1], qv[0], 0.0]]
    )
    top = np.hstack([q0 * np.eye(3) + qv_hat, qv.reshape(3, 1)])
    bottom = np.hstack([-qv.reshape(1, 3), np.array([[q0]])])
    return np.vstack([top, bottom])


def measurement_jacobian_analytic(
    m: np.ndarray,
    s_vals,
    q_obs,
    *,
    gamma: int = 26,
    L: float = 100.0,
    model: str = "5d",
):
    """
    Stacked measurement Jacobian and innovation vector using analytic chain rule.

    theta = 2 * vec(q_err),
    q_err = q_meas ⊗ q_pred^{-1} = q_meas ⊗ conj(q_pred).

    With xyzw convention, dtheta/dq_pred is:
      2 * [I3 0] * L(q_meas) * C,
    where C maps q_pred -> conj(q_pred) (diag([-1,-1,-1,1])).
    """
    R_pred, _ = forward_kinematics_multiple(m, s_vals, gamma=gamma, L=L, model=model)
    q_pred = [q_fix_sign(R.from_matrix(Rp).as_quat()) for Rp in R_pred]

    C = np.diag([-1.0, -1.0, -1.0, 1.0])
    E = np.hstack([np.eye(3), np.zeros((3, 1))])

    H_blocks, y_blocks = [], []
    for s_i, q_m, q_p, Rp in zip(s_vals, q_obs, q_pred, R_pred):
        q_m = q_fix_sign(np.asarray(q_m))
        q_p = q_fix_sign(np.asarray(q_p))

        q_err_raw = quat_mul(q_m, quat_conj(q_p))
        sign_err = 1.0 if q_err_raw[3] >= 0.0 else -1.0
        q_err = sign_err * q_err_raw
        theta = 2.0 * q_err[:3]

        dtheta_dqpred = sign_err * 2.0 * (E @ left_quat_matrix_xyzw(q_m) @ C)
        dqpred_dR = J_q_R(Rp)
        dR_dm = JRm_block(m, s_i, gamma=gamma, model=model)
        H_i = dtheta_dqpred @ dqpred_dR @ dR_dm

        H_blocks.append(H_i)
        y_blocks.append(-theta)

    return np.vstack(H_blocks), np.hstack(y_blocks)


def ekf_update_analytic(
    m: np.ndarray,
    P: np.ndarray,
    s_vals,
    q_obs,
    R_cov: np.ndarray,
    *,
    gamma: int = 26,
    L: float = 100.0,
    model: str = "5d",
    iters: int = 1,
):
    """Iterated EKF update using analytic Jacobian chain rule."""
    for _ in range(iters):
        H, y = measurement_jacobian_analytic(
            m, s_vals, q_obs, gamma=gamma, L=L, model=model
        )
        S = H @ P @ H.T + R_cov
        K = P @ H.T @ np.linalg.inv(S)
        m = m + K @ y
        P = (np.eye(len(m)) - K @ H) @ P
    return m, P


# ───────────── Demo / test harness ─────────────
if __name__ == "__main__":
    np.random.seed(44)

    # sensor positions – avoid s=0.0
    s_vals = [0.2, 0.6, 1.0]

    m_true = np.random.uniform(-4, 4, 5)
    L_phys = 100.0
    gamma_int = 26
    sigma_rot = 0.005
    T_steps = 500

    # ground truth
    R_true, _ = forward_kinematics_multiple(
        m_true, s_vals, gamma=gamma_int, L=L_phys, model="5d"
    )

    # noisy measurements (xyzw)
    meas_seq = []
    for _ in range(T_steps):
        q_list = []
        for Rg in R_true:
            dw = sigma_rot * np.random.randn(3)
            q_list.append(q_fix_sign(R.from_matrix(expm(skew(dw)) @ Rg).as_quat()))
        meas_seq.append(q_list)

    # EKF initial state
    m_est = np.zeros(5)
    P_est = np.eye(5) * 0.2
    Q = np.eye(5) * 1e-6
    R_cov = np.eye(3 * len(s_vals)) * (3 * sigma_rot) ** 2

    hist = np.zeros((T_steps, 5))
    for t in range(T_steps):
        P_pred = P_est + Q
        m_est, P_est = ekf_update_analytic(
            m_est,
            P_pred,
            s_vals,
            meas_seq[t],
            R_cov,
            gamma=gamma_int,
            L=L_phys,
            model="5d",
            iters=1,
        )
        hist[t] = m_est

    # ─ plots ─
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(6, 10))
    for i in range(5):
        axs[i].plot(hist[:, i], label=f"est m[{i}]")
        axs[i].axhline(m_true[i], ls="--", label="true")
        axs[i].legend(loc="upper right")
    axs[-1].set_xlabel("time step")
    plt.tight_layout()
    plt.show()
