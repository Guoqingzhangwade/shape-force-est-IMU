#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ekf_5d.py  –  EKF shape estimator (5-parameter curvature model)

State  m = [mx0, mx1,  my0, my1,  mz0]ᵀ
κx(s) = mx0 + mx1·s        (linear in s)
κy(s) = my0 + my1·s
κz(s) = mz0                (constant twist)
"""

# ---------- libraries ----------
import os
import sys

import numpy as np
from numpy.linalg import inv, norm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from shape_force_est_imu.ekf import (  # noqa: E402
    q_fix_sign,
    quat_mul,
    rotation_from_poE,
    theta_from_measurement,
    numeric_jacobian_theta,
)

# ---------- global parameters ----------
np.random.seed(128)
STATE_DIM  = 5
IMU_POS    = np.array([0.25, 0.50, 0.75])      # along arclength
L_PHYS     = 100.0                             # mm  (only for translation)
NUM_STEPS  = 100
GAMMA      = 20                                 # PoE subdivisions

# ground truth & initial guess
TRUE_M = np.array([1.0, 0.2, 1.5, 0.5, 1.7])
TRUE_M = np.random.uniform(-3,3,5)
INIT_M = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# process / measurement noise
PROC_STD0, PROC_STD1 = 1e-5, 1e-6              # σ for 0-th & 1-st order terms
Q = np.diag([PROC_STD0**2, PROC_STD1**2,
             PROC_STD0**2, PROC_STD1**2,
             PROC_STD0**2])
MEAS_STD_DEG = 0.5
R_SINGLE = (np.deg2rad(MEAS_STD_DEG)**2) * np.eye(3)

# ---------- generate synthetic IMU data ----------
meas = []
for _ in range(NUM_STEPS):
    frame = []
    for s in IMU_POS:
        R_clean = rotation_from_poE(TRUE_M, s, gamma=GAMMA, L=L_PHYS, model="5d")
        q_clean = q_fix_sign(R.from_matrix(R_clean).as_quat())
        noise = R.from_rotvec(np.random.randn(3) * np.deg2rad(MEAS_STD_DEG))
        q_meas = quat_mul(q_fix_sign(noise.as_quat()), q_clean)
        q_meas = q_fix_sign(q_meas)
        frame.append(q_meas)
    meas.append(frame)

# ---------- EKF loop ----------
m_est = INIT_M.copy()
P_est = 1e-2*np.eye(STATE_DIM)
est_hist = [m_est.copy()]          # include initial state at t = 0

for k in range(NUM_STEPS):
    # prediction
    m_pred, P_pred = m_est.copy(), P_est + Q

    # one EKF update pass
    H_stack, r_stack = [], []
    for i, s in enumerate(IMU_POS):
        q_meas = meas[k][i]
        r = -theta_from_measurement(
            q_meas, m_pred, s, gamma=GAMMA, L=L_PHYS, model="5d"
        )
        H = numeric_jacobian_theta(
            q_meas, m_pred, s, gamma=GAMMA, L=L_PHYS, model="5d"
        )
        H_stack.append(H)
        r_stack.append(r)
    H = np.vstack(H_stack)
    r = np.hstack(r_stack)
    R_big = np.kron(np.eye(len(IMU_POS)), R_SINGLE)

    S = H @ P_pred @ H.T + R_big
    K = P_pred @ H.T @ inv(S)

    m_est = m_pred + K @ r
    P_est = (np.eye(STATE_DIM) - K @ H) @ P_pred
    est_hist.append(m_est.copy())

est_hist = np.array(est_hist)      # length = NUM_STEPS+1

# ---------- plot ----------
t = np.arange(NUM_STEPS + 1)
lbl = ['mx0','mx1','my0','my1','mz0']
colors = ['r','g','b','c','m']

plt.figure(figsize=(11,7))
for i in range(STATE_DIM):
    plt.plot(t, TRUE_M[i]*np.ones_like(t),
             colors[i], lw=2, label='True' if i==0 else "")
    plt.plot(t, est_hist[:, i],
             ':', color=colors[i], lw=2, label='EKF' if i==0 else "")
    # draw initial marker for this coefficient
    plt.plot(0, INIT_M[i], marker='x', ms=9, color=colors[i],
             label='Init (t=0)' if i==0 else "")

plt.xlabel('time-step'); plt.ylabel('modal coefficient')
plt.title('EKF (5-D model) – initial + estimates')
plt.grid(True); plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')
plt.tight_layout(); plt.show()
