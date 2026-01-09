# -*- coding: utf-8 -*-
"""
ekf_curvature_timeseries_quat_v4.py
-----------------------------------
Quaternion EKF for 5-parameter curvature model.
Now with sign-safe quaternions, non-singular sensor set,
and gentler gains.
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
    forward_kinematics_multiple,
    ekf_update,
)

# ───────────── Demo / test harness ─────────────
if __name__ == "__main__":
    np.random.seed(44)

    # ★ sensor positions – avoid s=0.0
    s_vals = [0.2, 0.6, 1.0]

    m_true   = np.random.uniform(-4,4,5)
    L_phys   = 100.0
    γ_int    = 26
    σ_rot    = 0.005
    T_steps  = 500

    # ground truth
    R_true, _ = forward_kinematics_multiple(
        m_true, s_vals, gamma=γ_int, L=L_phys, model="5d"
    )

    # noisy measurements
    meas_seq=[]
    for _ in range(T_steps):
        q=[]
        for Rg in R_true:
            δw = σ_rot*np.random.randn(3)
            q.append(q_fix_sign(R.from_matrix(expm(skew(δw)) @ Rg).as_quat()))
        meas_seq.append(q)

    # EKF initial state
    m_est = np.zeros(5)
    P_est = np.eye(5)*0.2
    Q     = np.eye(5)*1e-6          # ★ smaller Q
    R_cov = np.eye(3*len(s_vals))*(3*σ_rot)**2   # ★ larger R

    hist=np.zeros((T_steps,5))
    for t in range(T_steps):
        P_pred = P_est + Q
        # inner IEKF loop: set iters=2 for difficult cases
        m_est, P_est = ekf_update(
            m_est,
            P_pred,
            s_vals,
            meas_seq[t],
            R_cov,
            gamma=γ_int,
            L=L_phys,
            model="5d",
            iters=1,
        )
        hist[t]=m_est

    # ─ plots ─
    fig,axs=plt.subplots(5,1,sharex=True,figsize=(6,10))
    for i in range(5):
        axs[i].plot(hist[:,i],label=f'est m[{i}]')
        axs[i].axhline(m_true[i],ls='--',label='true')
        axs[i].legend(loc='upper right')
    axs[-1].set_xlabel('time step'); plt.tight_layout(); plt.show()
