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
import numpy as np
from numpy.linalg import inv, norm
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm
import matplotlib.pyplot as plt

# ---------- global parameters ----------
np.random.seed(24)
STATE_DIM  = 5
IMU_POS    = np.array([0.25, 0.50, 0.75])      # along arclength
L_PHYS     = 100.0                             # mm  (only for translation)
NUM_STEPS  = 100
GAMMA      = 20                                 # PoE subdivisions

# ground truth & initial guess
TRUE_M = np.array([1.0, 0.2, 1.5, 0.5, 1.7])
TRUE_M = np.random.uniform(-2,2,5)
INIT_M = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# process / measurement noise
PROC_STD0, PROC_STD1 = 1e-5, 1e-6              # σ for 0-th & 1-st order terms
Q = np.diag([PROC_STD0**2, PROC_STD1**2,
             PROC_STD0**2, PROC_STD1**2,
             PROC_STD0**2])
MEAS_STD_DEG = 0.5
R_SINGLE = (np.deg2rad(MEAS_STD_DEG)**2) * np.eye(3)

# ---------- helper functions ----------
def skew(u):
    return np.array([[0, -u[2], u[1]],
                     [u[2],  0, -u[0]],
                     [-u[1], u[0], 0]])

def phi(s):
    """Basis Φ(s) ∈ ℝ^(3×5)."""
    return np.array([[1, s, 0, 0, 0],
                     [0, 0, 1, s, 0],
                     [0, 0, 0, 0, 1]])

def twist(k):
    e3 = np.array([0, 0, 1.0])
    return np.block([[skew(k), e3[:, None]],
                     [np.zeros((1, 3)), 0]])

def magnus_psi(s_i, h, m):
    xi = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])
    c1, c2 = s_i - h + xi*h
    k1, k2 = phi(c1)@m, phi(c2)@m
    e1, e2 = twist(k1), twist(k2)
    return (h/2)*(e1+e2) + (np.sqrt(3)/12)*h**2*(e1@e2 - e2@e1)

def fwd_rotation(m, s):
    """Rotation matrix at arclength s via piece-wise Magnus PoE."""
    T = np.eye(4); h = s/GAMMA
    for k in range(1, GAMMA+1):
        T = T @ expm(magnus_psi(k*h, h, m))
    T[:3, 3] *= L_PHYS          # scale translation once
    return T[:3, :3]

def q_xyzw_to_wxyz(q):
    x, y, z, w = q
    return np.array([w, x, y, z])

def q_mul(a, b):
    w1,x1,y1,z1 = a
    w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

def q_inv(q):
    w,x,y,z = q
    return np.array([w, -x, -y, -z]) / np.dot(q, q)

def theta(q_meas, m, s):
    """Minimal 3-vector orientation error."""
    q_pred = q_xyzw_to_wxyz(R.from_matrix(fwd_rotation(m, s)).as_quat())
    if q_pred[0] < 0: q_pred = -q_pred
    q_e = q_mul(q_meas, q_inv(q_pred))
    if q_e[0] < 0: q_e = -q_e
    return 2*q_e[1:]

def jac_num(q_meas, m, s, eps=1e-6):
    J = np.zeros((3, STATE_DIM))
    for i in range(STATE_DIM):
        m_p, m_m = m.copy(), m.copy()
        m_p[i] += eps; m_m[i] -= eps
        J[:, i] = (theta(q_meas, m_p, s) -
                   theta(q_meas, m_m, s)) / (2*eps)
    return J

# ---------- generate synthetic IMU data ----------
meas = []
for _ in range(NUM_STEPS):
    frame = []
    for s in IMU_POS:
        q_clean = q_xyzw_to_wxyz(R.from_matrix(fwd_rotation(TRUE_M, s)).as_quat())
        noise   = R.from_rotvec(np.random.randn(3)*np.deg2rad(MEAS_STD_DEG))
        q_meas  = q_mul(q_xyzw_to_wxyz(noise.as_quat()), q_clean)
        if q_meas[0] < 0: q_meas = -q_meas
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
        r = -theta(q_meas, m_pred, s)          # residual (z = 0, h = θ)
        H = jac_num(q_meas, m_pred, s)
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
