#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shape_iekf_demo.py
──────────────────
• 5-parameter polynomial-curvature model on SE(3)
• 4th-order Magnus expansion + product-of-exponentials
• Three orientation-only IMU sensors along the backbone
• Left-invariant Iterated EKF (IEKF) for shape estimation
"""

# ────────────── Imports ──────────────
import numpy as np
from numpy.linalg import norm, inv
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm
import matplotlib.pyplot as plt

# ─────────────── Globals ─────────────
np.random.seed(24)           # reproducibility
STATE_DIM = 5                # [mx0, mx1,  my0, my1,  mz0]  (linear poly in x,y + const in z)
IMU_POS   = np.array([0.25, 0.50, 0.75])   # normalised arc-length positions
L         = 100.0            # physical length [mm] – used only for translation scaling
NUM_STEPS = 300
MAGNUS_GAMMA = 20            # sub-intervals for PoE integration

# noise (tune as needed)
PROC_STD_0, PROC_STD_1 = 1e-5, 1e-6
MEAS_STD_DEG = 0.5
Q = np.diag([PROC_STD_0**2, PROC_STD_1**2,
             PROC_STD_0**2, PROC_STD_1**2,
             PROC_STD_0**2])
R_single = (np.deg2rad(MEAS_STD_DEG)**2) * np.eye(3)

# ───────────── Utility functions ─────────────
def skew(u):
    return np.array([[   0, -u[2],  u[1]],
                     [ u[2],    0, -u[0]],
                     [-u[1],  u[0],    0]])

def quat_mul(q1, q2):
    # hamilton product, scalar-first quats
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

def quat_inv(q):
    w,x,y,z = q
    return np.array([w, -x, -y, -z]) / np.dot(q,q)

# ───────────── Model functions ─────────────
def phi_func(s):
    """Basis matrix Φ(s) ∈ ℝ^{3×5} mapping modal coeffs → curvature κ(s)."""
    return np.array([[1, s, 0, 0, 0],
                     [0, 0, 1, s, 0],
                     [0, 0, 0, 0, 1]])

def twist_matrix(kappa):
    """4×4 body-twist matrix  Ê(κ)  for given curvature vector κ."""
    e3 = np.array([0,0,1.0])                # unit tangent
    return np.block([[skew(kappa),  e3[:,None]],
                     [np.zeros((1,3)), 0]])

def magnus_psi_i(s_i, h, m):
    """4th-order Magnus term over sub-interval [s_i-h, s_i]."""
    xi  = np.array([0.5-np.sqrt(3)/6, 0.5+np.sqrt(3)/6])
    c1, c2 = s_i - h + xi*h
    k1, k2 = phi_func(c1)@m,  phi_func(c2)@m
    e1, e2 = twist_matrix(k1), twist_matrix(k2)
    return (h/2)*(e1+e2) + (np.sqrt(3)/12)*h**2*(e1@e2 - e2@e1)

def product_of_exponentials(m, s, gamma=MAGNUS_GAMMA):
    """Homogeneous transform T(s) via piecewise Magnus expansion."""
    h = s/gamma
    T = np.eye(4)
    for k in range(1, gamma+1):
        psi = magnus_psi_i(k*h, h, m)
        T = T @ expm(psi)
    T[:3,3] *= L           # scale translation once
    return T

def forward_kinematics(m, s):
    """Return rotation (3×3) and position (3,) at arc-length s∈[0,1]."""
    T = product_of_exponentials(m, s)
    return T[:3,:3], T[:3,3]

# ───────── Quaternion error & Jacobian ─────────
def theta_error(q_meas, m, s):
    """
    Minimal-vector orientation error between measurement and prediction.
    θ = 2 * vec(q_e)  with  q_e = q_meas ⊗ q_pred⁻¹  (scalar-first).
    """
    R_pred,_ = forward_kinematics(m, s)
    q_pred   = R.from_matrix(R_pred).as_quat(scalar_first=True)
    if q_pred[0] < 0: q_pred = -q_pred          # enforce unique sign

    q_e = quat_mul(q_meas, quat_inv(q_pred))
    if q_e[0] < 0: q_e = -q_e
    return 2*q_e[1:]                            # 3×1

def theta_jac_numeric(q_meas, m, s, eps=1e-6):
    """Finite-difference Jacobian ∂θ/∂m (3×5)."""
    J = np.zeros((3, STATE_DIM))
    theta0 = theta_error(q_meas, m, s)
    for i in range(STATE_DIM):
        m_p = m.copy(); m_p[i] += eps
        m_m = m.copy(); m_m[i] -= eps
        J[:,i] = (theta_error(q_meas, m_p, s) -
                  theta_error(q_meas, m_m, s)) / (2*eps)
    return J

# ───────────── Synthetic data generation ─────────────
TRUE_M = np.array([1.0, 0.2, 1.5, 0.5, 1.7])   # ground truth modal coeffs
IMU_R  = []
for s in IMU_POS:
    R_true,_ = forward_kinematics(TRUE_M, s)
    IMU_R.append(R_true)

def add_measurement_noise(R_clean):
    rotvec_noise = np.random.randn(3) * np.deg2rad(MEAS_STD_DEG)
    return R.from_matrix(R_clean)*R.from_rotvec(rotvec_noise)

measurements = []
for _ in range(NUM_STEPS):
    frame = []
    for R_clean in IMU_R:
        q  = add_measurement_noise(R_clean).as_quat(scalar_first=True)
        if q[0] < 0: q = -q
        frame.append(q)
    measurements.append(frame)

# ───────────── IEKF initialisation ─────────────
m_est = np.array([0.5, -0.1, 1.0,  0.0, 2.0])   # deliberately off
m_est = np.array([0.0, 0.0, 0.0,  0.0, 0.0])   # deliberately off
P_est = 1e-2*np.eye(STATE_DIM)

# storage for plotting
est_hist = []

# ───────────── IEKF loop ─────────────
for k in range(NUM_STEPS):

    # 1) prediction (state is constant)
    m_pred, P_pred = m_est.copy(), P_est + Q

    # 2) iterated measurement update
    m_iter, P_iter = m_pred.copy(), P_pred.copy()
    for _ in range(5):            # usually ≤5 iterations suffice
        H_list, r_list = [], []

        for i,s in enumerate(IMU_POS):
            q_meas = measurements[k][i]
            r  = -theta_error(q_meas, m_iter, s)             # z=0, h=θ
            H  = theta_jac_numeric(q_meas, m_iter, s)
            H_list.append(H)
            r_list.append(r)

        H = np.vstack(H_list)                                # 3×N_IMU rows
        r = np.hstack(r_list)

        R_big = np.kron(np.eye(len(IMU_POS)), R_single)
        S = H @ P_iter @ H.T + R_big
        K = P_iter @ H.T @ inv(S)

        delta = K @ r
        m_new = m_iter + delta
        if norm(delta) < 1e-6:
            m_iter = m_new
            break
        m_iter = m_new

        # recompute covariance around new linearisation
        P_iter = (np.eye(STATE_DIM) - K @ H) @ P_iter

    # 3) commit iteration result
    m_est, P_est = m_iter, P_iter
    est_hist.append(m_est.copy())

# ───────────── Plot results ─────────────
est_hist = np.array(est_hist)
t = np.arange(NUM_STEPS)

plt.figure(figsize=(10,6))
colors = ['r','g','b','c','m']
labels = ['mx0','mx1','my0','my1','mz0']
for i in range(STATE_DIM):
    plt.plot(t, TRUE_M[i]*np.ones_like(t), colors[i], lw=2, label=f'True {labels[i]}')
    plt.plot(t, est_hist[:,i], '--', color=colors[i], label=f'IEKF {labels[i]}')
plt.xlabel('time-step'); plt.ylabel('modal coefficient value')
plt.title('True vs IEKF-estimated modal coefficients (5-D model)')
plt.grid(True); plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')
plt.tight_layout(); plt.show()
