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
np.random.seed(128)
STATE_DIM  = 6
IMU_POS    = np.array([0.25, 0.50, 0.75])      # along arclength
L_PHYS     = 100.0                             # mm  (only for translation)
NUM_STEPS  = 100
GAMMA      = 20                                 # PoE subdivisions

# ground truth & initial guess
TRUE_M = np.array([1.0, 0.2, 1.5, 0.5, 1.7, 0.3])
TRUE_M = np.random.uniform(-2,2,6)
INIT_M = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# process / measurement noise
PROC_STD0, PROC_STD1 = 1e-5, 1e-6              # σ for 0-th & 1-st order terms
Q = np.diag([PROC_STD0**2, PROC_STD1**2,
             PROC_STD0**2, PROC_STD1**2,
             PROC_STD0**2, PROC_STD1**2])
MEAS_STD_DEG = 0.5
R_SINGLE = (np.deg2rad(MEAS_STD_DEG)**2) * np.eye(3)

e3 = np.array([0, 0, L_PHYS])

# ---------- helper functions ----------
def skew(u):
    return np.array([[0, -u[2], u[1]],
                     [u[2],  0, -u[0]],
                     [-u[1], u[0], 0]])

def phi(s):
    """Basis Φ(s) ∈ ℝ^(3×5)."""
    return np.array([[1, s, 0, 0, 0, 0],
                     [0, 0, 1, s, 0, 0],
                     [0, 0, 0, 0, 1, s]])

def twist(k, e3):
    return np.block([[skew(k), e3[:, None]],
                     [np.zeros((1, 3)), 0]])

def magnus_psi(s_i, h, m, e3):
    xi = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])
    c1, c2 = s_i - h + xi*h
    k1, k2 = phi(c1)@m, phi(c2)@m
    e1, e2 = twist(k1,e3), twist(k2,e3)
    return (h/2)*(e1+e2) + (np.sqrt(3)/12)*h**2*(e1@e2 - e2@e1)

def fwd_rotation(m, s, e3):
    """Rotation matrix at arclength s via piece-wise Magnus PoE."""
    T = np.eye(4); h = s/GAMMA
    for k in range(1, GAMMA+1):
        T = T @ expm(magnus_psi(k*h, h, m, e3))
    T[:3, 3] *= 1          # scale translation once
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
    q_pred = q_xyzw_to_wxyz(R.from_matrix(fwd_rotation(m, s, e3)).as_quat())
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

def compute_dtheta_dqhat(q_meas):
    """
    Compute the derivative of the minimal error vector theta with respect to the predicted quaternion q_hat.

    Args:
        q_meas (numpy.ndarray): Measured quaternion [q0, q1, q2, q3] (scalar-first convention)

    Returns:
        numpy.ndarray: Derivative matrix of shape (3, 4)
    """
    # Ensure the quaternion is normalized
    q_meas = q_meas / np.linalg.norm(q_meas)

    if q_meas[0] < 0:
        q_meas = -q_meas
    
    # Extract scalar and vector parts
    q0 = q_meas[0]  # Scalar part
    qv = q_meas[1:]  # Vector part
    
    # Skew-symmetric matrix of qv
    qv_hat = np.array([
        [ 0,       -qv[2],  qv[1]],
        [ qv[2],    0,     -qv[0]],
        [-qv[1],  qv[0],     0   ]
    ])
    
    # Compute the derivative matrix with dimension 3x4
    # first column is qv / ||q||, second-four columns is -q0 * I - qv_hat
    derivative = np.zeros((3, 4))
    derivative[:, 0] = qv 
    derivative[:, 1:] = -q0 * np.eye(3) - qv_hat
    
    derivative = 2 * derivative

    return derivative

def compute_q_R_derivative(R_matrix):
    """
    Compute the derivative of the predicted quaternion with respect to the predicted rotation matrix.

    Args:
        R_matrix (numpy.ndarray): Rotation matrix of shape (3, 3)

    Returns:
        numpy.ndarray: Derivative tensor of shape (4, 3, 3)
    """
    # Ensure the rotation matrix is valid
    rot = R.from_matrix(R_matrix)
    q = rot.as_quat(scalar_first=True)  # [w, x, y, z]
    q = q / np.linalg.norm(q)  # Normalize the quaternion

    if q[0] < 0:
        q = -q

    # Extract quaternion components
    q0, q1, q2, q3 = q

    # Initialize the Jacobian tensor
    J = np.zeros((4, 3, 3))  # Shape: (4, 3, 3)

    # Compute derivatives for n = 0 (q0)
    for i in range(3):
        J[0, i, i] = 1 / (8 * q0)

    # Compute derivatives for n = 1 (q1)
    # Positive Index: (2, 1) -> R32
    J[1, 2, 1] = (1 / (4 * q0)) - (q1 / (8 * q0**2))
    # Negative Index: (1, 2) -> R23
    J[1, 1, 2] = (-1 / (4 * q0)) - (q1 / (8 * q0**2))
    # Diagonal terms
    for i in range(3):
        J[1, i, i] += (-q1 / (8 * q0**2))

    # Compute derivatives for n = 2 (q2)
    # Positive Index: (0, 2) -> R13
    J[2, 0, 2] = (1 / (4 * q0)) - (q2 / (8 * q0**2))
    # Negative Index: (2, 0) -> R31
    J[2, 2, 0] = (-1 / (4 * q0)) - (q2 / (8 * q0**2))
    # Diagonal terms
    for i in range(3):
        J[2, i, i] += (-q2 / (8 * q0**2))

    # Compute derivatives for n = 3 (q3)
    # Positive Index: (1, 0) -> R21
    J[3, 1, 0] = (1 / (4 * q0)) - (q3 / (8 * q0**2))
    # Negative Index: (0, 1) -> R12
    J[3, 0, 1] = (-1 / (4 * q0)) - (q3 / (8 * q0**2))
    # Diagonal terms
    for i in range(3):
        J[3, i, i] += (-q3 / (8 * q0**2))

    return J

def compute_R_m_derivative(m_coeffs, s, gamma):
    
    Nm = len(m_coeffs)  # Number of modal coefficients
    Ns = gamma          # Number of subdivisions
    h = s / Ns          # Length of each subdivision
    
    # Constants
    # L = 100.0  # Total arc length
    e3 = np.array([0, 0, L_PHYS])  # Local tangent vector
    
    # Preallocate arrays
    Psi = []         # List to store Psi_k matrices
    e_Psi = []       # List to store e^{Psi_k} matrices
    T_before = [np.eye(4)]  # List to store cumulative products before k
    T_after = [np.eye(4)] * (Ns + 1)  # List to store cumulative products after k
    
    # Quadrature points for 2-point Gaussian quadrature
    xi = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])  # Quadrature points in [0, 1]
    
    # Preallocate derivative arrays
    dPsi_dm = [np.zeros((4, 4, Nm)) for _ in range(Ns)]  # List of 4x4xNm arrays
    de_Psi_dm = [np.zeros((4, 4, Nm)) for _ in range(Ns)]
    
    # Compute Psi_k and e^{Psi_k}
    for k in range(1, Ns + 1):
        s_i = k * h  # Start of the interval (we skip s = 0)
        
        # Compute Ψ_k using magnus_psi_i
        Psi_k = magnus_psi(s_i, h, m_coeffs, e3)  
        Psi.append(Psi_k)
        
        # Compute e^{Ψ_k}
        e_Psi_k = expm(Psi_k)
        e_Psi.append(e_Psi_k)
        
        # Update cumulative product T_before
        T_before.append(T_before[-1] @ e_Psi_k)
    
    # Compute T_after
    T_after[Ns] = np.eye(4)
    for k in range(Ns - 1, 0, -1):
        T_after[k] = e_Psi[k] @ T_after[k + 1]
    
    # Compute derivatives of Psi_k
    for k in range(1, Ns + 1):
        s_i = k * h
        c1 = s_i - h + xi[0] * h
        c2 = s_i - h + xi[1] * h
        
        # Compute phi at quadrature points
        phi_c1 = phi(c1)
        phi_c2 = phi(c2)
        
        # Compute kappa at quadrature points
        kappa_1 = phi_c1 @ m_coeffs
        kappa_2 = phi_c2 @ m_coeffs
        
        # Compute eta at quadrature points
        eta_1 = twist(kappa_1, e3)
        eta_2 = twist(kappa_2, e3)
        
        # Precompute commutators
        commutator_eta = eta_1 @ eta_2 - eta_2 @ eta_1
        
        for i in range(Nm):
            # Compute derivative of kappa with respect to m_i
            dkappa_1_dmi = phi_c1[:, i]
            dkappa_2_dmi = phi_c2[:, i]
            
            # Compute derivative of eta with respect to m_i
            deta_1_dmi = twist(dkappa_1_dmi, np.zeros(3))
            deta_2_dmi = twist(dkappa_2_dmi, np.zeros(3))
            
            # Compute derivative of commutators
            commutator_deta = (deta_1_dmi @ eta_2 - eta_2 @ deta_1_dmi) + (eta_1 @ deta_2_dmi - deta_2_dmi @ eta_1)
            
            # Compute derivative of Psi_k
            dPsi_k_dmi = (h / 2) * (deta_1_dmi + deta_2_dmi) + (np.sqrt(3) / 12) * h**2 * commutator_deta
            dPsi_dm[k - 1][:, :, i] = dPsi_k_dmi  # Adjusted index k - 1
    
    # Compute derivatives of e^{Psi_k}
    for k in range(Ns):
        Psi_k = Psi[k]
        e_Psi_k = e_Psi[k]
        
        for i in range(Nm):
            dPsi_k_dmi = dPsi_dm[k][:, :, i]
            
            # Compute the approximation of dexpinv
            dexpinv_approx = dPsi_k_dmi + 0.5 * ((Psi_k) @ dPsi_k_dmi - dPsi_k_dmi @ (Psi_k))
            
            # Compute derivative of e^{Psi_k}
            de_Psi_k_dmi = e_Psi_k @ dexpinv_approx
            de_Psi_dm[k][:, :, i] = de_Psi_k_dmi
    
    # Assemble the derivative dT_dm
    dT_dm = np.zeros((4, 4, Nm))
    
    for i in range(Nm):
        for k in range(Ns):
            term = T_before[k] @ de_Psi_dm[k][:, :, i] @ T_after[k + 1]
            dT_dm[:, :, i] += term
    
    # Extract the derivative of R with respect to m_i
    dR_dm = dT_dm[:3, :3, :]  # Dimensions: 3 x 3 x Nm
    
    return dR_dm

def jac_analy(q_meas, m, s, gamma, e3):
    """
    Compute the derivative of the minimal error vector theta with respect to the modal coefficients m.

    Args:
        q_meas (numpy.ndarray): Measured quaternion [q0, q1, q2, q3] (scalar-first convention)
        R_matrix (numpy.ndarray): Predicted rotation matrix of shape (3, 3)
        m_coeffs (numpy.ndarray): Modal coefficients vector of shape (5,)
        s (float): Total arclength
        gamma (int): Number of subdivisions

    Returns:
        numpy.ndarray: Derivative of theta with respect to m, shape (3, 6)
    """
    import numpy as np

    # Step 1: Compute derivative of theta with respect to q_hat
    dtheta_dqhat = compute_dtheta_dqhat(q_meas)  # Shape: (3, 4)

    # calculate R_matrix based on m
    R_matrix = fwd_rotation(m, s, e3)
    # Step 2: Compute derivative of q_hat with respect to R
    dqhat_dR = compute_q_R_derivative(R_matrix)  # Shape: (4, 3, 3)

    # Step 3: Compute derivative of R with respect to m
    dR_dm = compute_R_m_derivative(m, s, gamma)  # Shape: (3, 3, 6)

    # Step 4: Compute derivative of theta with respect to R
    # Using Einstein summation to contract over q_hat indices
    dtheta_dR = np.einsum('ik,kjl->ijl', dtheta_dqhat, dqhat_dR)  # Shape: (3, 3, 3)

    # Step 5: Compute derivative of theta with respect to m
    # Contracting over R indices to get derivative with respect to m
    dtheta_dm = np.einsum('ijl,jln->in', dtheta_dR, dR_dm)  # Shape: (3, 6)

    return dtheta_dm

# ---------- generate synthetic IMU data ----------
meas = []
for _ in range(NUM_STEPS):
    frame = []
    for s in IMU_POS:
        q_clean = q_xyzw_to_wxyz(R.from_matrix(fwd_rotation(TRUE_M, s, e3)).as_quat())
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
        # H = jac_analy(q_meas, m_pred, s, 10, e3)
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
lbl = ['mx0','mx1','my0','my1','mz0','mz1']
colors = ['r','g','b','c','m', 'y']

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
plt.title('EKF (6-D model) – initial + estimates')
plt.grid(True); plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')
plt.tight_layout(); plt.show()
