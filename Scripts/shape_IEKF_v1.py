import numpy as np
from numpy.linalg import norm, inv
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm
import matplotlib.pyplot as plt

# ---------------- Parameters ---------------- #
np.random.seed(24)
STATE_DIM = 5
IMU_POS = np.array([0.25, 0.50, 0.75])
L = 100.0
NUM_STEPS = 300
MAGNUS_GAMMA = 20

# noise
PROC_STD_0, PROC_STD_1 = 1e-5, 1e-6
MEAS_STD_DEG = 0.5
Q = np.diag([PROC_STD_0**2, PROC_STD_1**2,
             PROC_STD_0**2, PROC_STD_1**2,
             PROC_STD_0**2])
R_single = (np.deg2rad(MEAS_STD_DEG)**2) * np.eye(3)

# true and initial modal coefficients
TRUE_M = np.array([1.0, 0.2, 1.5, 0.5, 1.7])
INIT_M = np.array([0.5, -0.1, 1.0, 0.0, 2.0])   # deliberately off


# -------------- Utility ---------------- #
def skew(u):
    return np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

def quat_mul(q1, q2):
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

def quat_scipy_to_scalar_first(q_xyzw):
    """Convert scipy's (x,y,z,w) to (w,x,y,z)."""
    x,y,z,w = q_xyzw
    return np.array([w,x,y,z])

def quat_scalar_first_to_xyzw(q_wxyz):
    """Convert (w,x,y,z) to (x,y,z,w) for scipy."""
    w,x,y,z = q_wxyz
    return np.array([x,y,z,w])

def phi_func(s):
    return np.array([[1, s, 0, 0, 0],
                     [0, 0, 1, s, 0],
                     [0, 0, 0, 0, 1]])

def twist_matrix(kappa):
    e3 = np.array([0,0,1.0])
    return np.block([[skew(kappa), e3[:,None]],
                     [np.zeros((1,3)), 0]])

def magnus_psi_i(s_i, h, m):
    xi = np.array([0.5-np.sqrt(3)/6, 0.5+np.sqrt(3)/6])
    c1, c2 = s_i - h + xi*h
    k1, k2 = phi_func(c1)@m, phi_func(c2)@m
    e1, e2 = twist_matrix(k1), twist_matrix(k2)
    return (h/2)*(e1+e2) + (np.sqrt(3)/12)*h**2*(e1@e2 - e2@e1)

def product_of_exponentials(m, s, gamma=MAGNUS_GAMMA):
    T = np.eye(4)
    h = s/gamma
    for k in range(1, gamma+1):
        psi = magnus_psi_i(k*h, h, m)
        T = T @ expm(psi)
    T[:3,3] *= L
    return T

def forward_kinematics(m, s):
    T = product_of_exponentials(m, s)
    return T[:3,:3], T[:3,3]

def theta_error(q_meas_wxyz, m, s):
    R_pred,_ = forward_kinematics(m, s)
    q_pred_xyzw = R.from_matrix(R_pred).as_quat()
    q_pred = quat_scipy_to_scalar_first(q_pred_xyzw)
    if q_pred[0] < 0: q_pred = -q_pred
    q_e = quat_mul(q_meas_wxyz, quat_inv(q_pred))
    if q_e[0] < 0: q_e = -q_e
    return 2*q_e[1:]

def theta_jac_numeric(q_meas_wxyz, m, s, eps=1e-6):
    J = np.zeros((3, STATE_DIM))
    for i in range(STATE_DIM):
        m_p, m_m = m.copy(), m.copy()
        m_p[i]+=eps; m_m[i]-=eps
        J[:,i] = (theta_error(q_meas_wxyz, m_p, s) - theta_error(q_meas_wxyz, m_m, s))/(2*eps)
    return J

# -------------- Generate synthetic measurements -------------- #
def add_noise_rot(R_clean):
    rotvec_noise = np.random.randn(3)*np.deg2rad(MEAS_STD_DEG)
    return R.from_matrix(R_clean)*R.from_rotvec(rotvec_noise)

clean_rotations = [forward_kinematics(TRUE_M, s)[0] for s in IMU_POS]
measurements = []
for _ in range(NUM_STEPS):
    frame = []
    for Rc in clean_rotations:
        q_xyzw = add_noise_rot(Rc).as_quat()
        q = quat_scipy_to_scalar_first(q_xyzw)
        if q[0] < 0: q = -q
        frame.append(q)
    measurements.append(frame)

# -------------- Filtering (EKF & IEKF) -------------- #
def filter_loop(iterated=False):
    m_est = INIT_M.copy()
    P_est = 1e-2*np.eye(STATE_DIM)
    est_hist = []
    for k in range(NUM_STEPS):
        m_pred, P_pred = m_est.copy(), P_est + Q

        m_iter, P_iter = m_pred.copy(), P_pred.copy()
        niters = 5 if iterated else 1
        for _ in range(niters):
            H_list, r_list = [], []
            for i,s in enumerate(IMU_POS):
                q_meas = measurements[k][i]
                r = -theta_error(q_meas, m_iter, s)
                H = theta_jac_numeric(q_meas, m_iter, s)
                H_list.append(H)
                r_list.append(r)
            H = np.vstack(H_list)
            r = np.hstack(r_list)
            R_big = np.kron(np.eye(len(IMU_POS)), R_single)
            S = H @ P_iter @ H.T + R_big
            K = P_iter @ H.T @ inv(S)
            delta = K @ r
            m_iter += delta
            P_iter = (np.eye(STATE_DIM)-K@H)@P_iter
            if norm(delta)<1e-6: break

        m_est, P_est = m_iter, P_iter
        est_hist.append(m_est.copy())
    return np.array(est_hist)

ekf_hist  = filter_loop(iterated=False)
iekf_hist = filter_loop(iterated=True)

# -------------- Plot -------------- #
t = np.arange(NUM_STEPS)
labels = ['mx0','mx1','my0','my1','mz0']
colors = ['r','g','b','c','m']

plt.figure(figsize=(11,7))
for i in range(STATE_DIM):
    plt.plot(t, TRUE_M[i]*np.ones_like(t), colors[i], lw=2, label=f'True {labels[i]}' if i==0 else "")
    plt.plot(t, ekf_hist[:,i], ':', color=colors[i], lw=2, label=f'EKF {labels[i]}' if i==0 else "")
    plt.plot(t, iekf_hist[:,i], '--', color=colors[i], lw=1.5, label=f'IEKF {labels[i]}' if i==0 else "")
    plt.plot(0, INIT_M[i], marker='x', color=colors[i], ms=9, label=f'Init {labels[i]}' if i==0 else "")
plt.xlabel('time-step'); plt.ylabel('modal coefficient value')
plt.title('True vs Estimated Modal Coefficients (EKF vs IEKF)')
plt.grid(True); plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')
plt.tight_layout()
plt.show()
