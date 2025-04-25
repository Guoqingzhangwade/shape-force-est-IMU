# -*- coding: utf-8 -*-
"""
ekf_curvature_timeseries_quat.py
--------------------------------
Quaternion–driven EKF for 5-parameter curvature model using
analytic measurement Jacobian:
            H = J_{θq} · J_{qq} · J_{qR} · J_{Rm}
Only J_{Rm} (∂R/∂m) comes from a helper you already wrote.
Everything else is closed-form.
"""
# ───────────────────────── Imports ──────────────────────────
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# ─────────────────── Quaternion utilities ───────────────────
def skew(u):          # 3-vector → 3×3 skew
    return np.array([[   0, -u[2],  u[1]],
                     [ u[2],    0, -u[0]],
                     [-u[1],  u[0],    0]])

def quat_mul(q1, q2):               # Hamilton product
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quat_conj(q):                     # inverse because unit
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_error(q_meas, q_pred):
    q_err = quat_mul(q_meas, quat_conj(q_pred))
    if q_err[3] < 0:        # keep small-angle convention
        q_err = -q_err
    return 2.0 * q_err[:3]  # minimal 3-vector residual θ

# ─── J_theta_q (3×4) and J_q_q (4×4) per manuscript ─────────
def J_theta_q(q_meas):
    qv, q0 = q_meas[:3], q_meas[3]
    return 2 * np.hstack([
        -qv.reshape(3, 1),             # 3×1 column
        -q0 * np.eye(3) - skew(qv)     # 3×3 block
    ])

def J_q_q(q_meas):
    qv, q0 = q_meas[:3], q_meas[3]
    row0  = np.hstack([q0,  qv])
    rows3 = np.hstack([qv.reshape(3,1), -q0*np.eye(3) - skew(qv)])
    return np.vstack([row0, rows3])

# ─── J_q_R : 4×9 analytic from R→q conversion table ─────────
def J_q_R(Rm):
    R11,R12,R13,R21,R22,R23,R31,R32,R33 = Rm.reshape(-1)
    q0 = 0.5*np.sqrt(1+R11+R22+R33)
    q  = np.empty(4)
    q[3] = q0
    q[0] = (R32-R23)/(4*q0)
    q[1] = (R13-R31)/(4*q0)
    q[2] = (R21-R12)/(4*q0)

    J = np.zeros((4,9))
    for idx in (0,4,8):                # diagonal entries
        J[3,idx] = 1/(8*q0)
    pos = {(0,7),(1,2),(2,3)}          # (n, flat-idx)
    neg = {(0,5),(1,6),(2,1)}
    for n in range(3):
        for idx in range(9):
            if (n,idx) in pos:
                J[n,idx] =  1/(4*q0) - q[n]/(8*q0**2)
            elif (n,idx) in neg:
                J[n,idx] = -1/(4*q0) - q[n]/(8*q0**2)
            elif idx in (0,4,8):
                J[n,idx] = - q[n]/(8*q0**2)
    return J

# ─────────────────── Curvature model & kinematics ───────────
def curvature_kappa(s, m):
    return np.array([m[0]+m[1]*s,  m[2]+m[3]*s,  m[4]])

def twist_matrix(kappa):
    e3 = np.array([0,0,1.])
    top = np.hstack([skew(kappa), e3.reshape(3,1)])
    return np.vstack([top, np.zeros((1,4))])

def magnus_4th_subinterval(s0, s1, m):
    h   = s1-s0
    c1  = s0 + h*(0.5 - np.sqrt(3)/6)
    c2  = s0 + h*(0.5 + np.sqrt(3)/6)
    η1, η2 = twist_matrix(curvature_kappa(c1,m)), twist_matrix(curvature_kappa(c2,m))
    return (h/2)*(η1+η2) + (h**2*np.sqrt(3)/12)*(η1@η2 - η2@η1)

def product_of_exponentials(m, s, *, gamma=10, L=100.0):
    d = np.linspace(0,s,gamma+1)
    T = np.eye(4)
    for k in range(1, gamma+1):
        T = T @ expm(magnus_4th_subinterval(d[k-1], d[k], m))
    T[:3,3] *= L
    return T

def forward_kinematics_multiple(m, s_vals, *, gamma=10, L=100.0):
    R_list, p_list = [], []
    for s in s_vals:
        T = product_of_exponentials(m, s, gamma=gamma, L=L)
        R_list.append(T[:3,:3])
        p_list.append(T[:3,3])
    return R_list, p_list

# ───── Analytic ∂κ/∂m and ∂η/∂m (for ∂Ψ/∂m) ────────────────
def d_kappa_dm(s, j):
    v = np.zeros(3)
    if   j==0: v[0]=1
    elif j==1: v[0]=s
    elif j==2: v[1]=1
    elif j==3: v[1]=s
    else:      v[2]=1
    return v

def d_eta_dm(s, j):
    k = d_kappa_dm(s,j)
    return np.vstack([np.hstack([skew(k), np.zeros((3,1))]),
                      np.zeros((1,4))])

def dPsi_dm(s0, s1, m, j):
    h  = s1-s0
    c1 = s0 + h*(0.5 - np.sqrt(3)/6)
    c2 = s0 + h*(0.5 + np.sqrt(3)/6)
    η1, η2  = twist_matrix(curvature_kappa(c1,m)), twist_matrix(curvature_kappa(c2,m))
    dη1, dη2 = d_eta_dm(c1,j), d_eta_dm(c2,j)
    part1 = (h/2)*(dη1+dη2)
    part2 = (h**2*np.sqrt(3)/12)*((dη1@η2+η1@dη2) - (dη2@η1+η2@dη1))
    return part1+part2

# def dexp_first(A,dA):                    # 1st-order Bernoulli
#     return dA + 0.5*(A@dA - dA@A)

# def dExp_dm(A,dA):
#     return expm(A) @ dexp_first(A,dA)

def dexp_so3(A, dA):
    """
    Exact right-Jacobian for so(3): returns J · dA
    A, dA are 3×3 skew matrices.
    """
    a = np.array([A[2,1], A[0,2], A[1,0]])
    θ = np.linalg.norm(a)
    if θ < 1e-9:
        return dA + 0.5*(A@dA - dA@A)  # fall back to 1st order
    hat_a = A
    J = ( (np.sin(θ)/θ)   * np.eye(3)
        + ((1-np.sin(θ)/θ)) * (a[:,None]@a[None,:])/(θ**2)
        + ((1-np.cos(θ))/θ) * hat_a/θ )
    return J @ dA

def dexp_se3(A,dA):
    """
    Cheap SE(3) lift: apply exact SO(3) Jacobian to rotation block
    and 1st-order term to translation (good enough for κ·L <~ 0.4 rad).
    """
    dR = dexp_so3(A[:3,:3], dA[:3,:3])
    dP = dA[:3,3:] + 0.5*(A[:3,:3]@dA[:3,3:] - dA[:3,:3]@A[:3,3:])
    out = np.zeros_like(dA)
    out[:3,:3] = dR
    out[:3,3:] = dP
    return out

def dExp_dm(A,dA):
    return expm(A) @ dexp_se3(A,dA)


def dT_dm(m,s,j,*,gamma=10):
    d = np.linspace(0,s,gamma+1)
    Es, Ps = [], []
    for k in range(1,gamma+1):
        P = magnus_4th_subinterval(d[k-1],d[k],m)
        Es.append(expm(P))
        Ps.append(P)
    left = np.eye(4)
    dT   = np.zeros((4,4))
    for k in range(gamma):
        right = np.eye(4)
        for r in range(k+1,gamma):
            right = right @ Es[r]
        dA  = dPsi_dm(d[k],d[k+1],m,j)
        dEk = dExp_dm(Ps[k], dA)
        dT += left @ dEk @ right
        left = left @ Es[k]
    return dT

def JRm_block(m,s,*,gamma=10):
    JRm = np.zeros((9,m.size))
    for j in range(m.size):
        JRm[:,j] = dT_dm(m,s,j,gamma=gamma)[:3,:3].reshape(9)
    return JRm

# ─────────────── Measurement Jacobian H ─────────────────────
def measurement_jacobian(m, s_vals, q_obs, *, gamma=26, L=100.0):
    H_blocks, y_blocks = [], []
    R_pred, _ = forward_kinematics_multiple(m, s_vals, gamma=gamma, L=L)
    q_pred = [R.from_matrix(Rp).as_quat() for Rp in R_pred]
    if q_pred[3] <0:
        q_pred = -q_pred

    for i,(q_m,q_p,Rp) in enumerate(zip(q_obs, q_pred, R_pred)):
        θ   = quat_error(q_m, q_p)                 # residual 3×1
        Jθq = J_theta_q(q_m)
        Jqq = J_q_q(q_m)
        JqR = J_q_R(Rp)
        JRm = JRm_block(m, s_vals[i], gamma=gamma) # 9×5

        H_blocks.append(Jθq @ Jqq @ JqR @ JRm)     # 3×5
        y_blocks.append(-θ)
    return np.vstack(H_blocks), np.hstack(y_blocks)

# ──────────────── EKF update (random walk) ──────────────────
def ekf_update(m_est,P_est, s_vals, q_obs, R_cov, *, gamma=26, L=100.0):
    H,y = measurement_jacobian(m_est, s_vals, q_obs, gamma=gamma, L=L)
    S   = H @ P_est @ H.T + R_cov
    K   = P_est @ H.T @ np.linalg.inv(S)
    m_new = m_est + K @ y
    P_new = (np.eye(len(m_est)) - K @ H) @ P_est
    return m_new, P_new

##############################################################################
# 5) Plot
##############################################################################

def plot_3d_curves(m_true, m_est, s_values, gamma=10, L=100.0):
    s_dense = np.linspace(0,1,100)
    rots_true, pos_true = forward_kinematics_multiple(m_true, s_dense, gamma=gamma, L=L)
    rots_est,  pos_est  = forward_kinematics_multiple(m_est,  s_dense, gamma=gamma, L=L)
    pos_true = np.array(pos_true)
    pos_est  = np.array(pos_est)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_true[:,0], pos_true[:,1], pos_true[:,2],
            color='blue', label='True shape')
    ax.plot(pos_est[:,0],  pos_est[:,1],  pos_est[:,2],
            color='green', linestyle='--', label='Est shape')

    rotsT, posT = forward_kinematics_multiple(m_true, s_values, gamma=gamma, L=L)
    rotsE, posE = forward_kinematics_multiple(m_est,  s_values, gamma=gamma, L=L)
    for (s_i, R_t, p_t, R_e, p_e) in zip(s_values, rotsT, posT, rotsE, posE):
        plot_frame(ax, p_t, R_t, f'True s={s_i}', 'blue')
        plot_frame(ax, p_e, R_e, f'Est s={s_i}', 'green')

    ax.set_xlabel('X(mm)')
    ax.set_ylabel('Y(mm)')
    ax.set_zlabel('Z(mm)')
    ax.legend()
    set_equal_axis(ax)
    plt.show()

def plot_frame(ax, origin, R, label=None, color='black'):
    scale = 5.0
    x_axis = R[:,0]*scale
    y_axis = R[:,1]*scale
    z_axis = R[:,2]*scale
    ax.quiver(origin[0], origin[1], origin[2],
              x_axis[0], x_axis[1], x_axis[2],
              color=color)
    ax.quiver(origin[0], origin[1], origin[2],
              y_axis[0], y_axis[1], y_axis[2],
              color=color)
    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0], z_axis[1], z_axis[2],
              color=color)
    if label:
        ax.text(origin[0], origin[1], origin[2], label, color=color)

def set_equal_axis(ax):
    xlims = ax.get_xlim3d()
    ylims = ax.get_ylim3d()
    zlims = ax.get_zlim3d()
    spans = [xlims[1]-xlims[0], ylims[1]-ylims[0], zlims[1]-zlims[0]]
    max_span = max(spans)
    x_center = 0.5*(xlims[0]+xlims[1])
    y_center = 0.5*(ylims[0]+ylims[1])
    z_center = 0.5*(zlims[0]+zlims[1])
    ax.set_xlim3d([x_center - max_span/2, x_center + max_span/2])
    ax.set_ylim3d([y_center - max_span/2, y_center + max_span/2])
    ax.set_zlim3d([z_center - max_span/2, z_center + max_span/2])


# ────────────────────────── Demo ────────────────────────────
if __name__ == "__main__":
    np.random.seed(44)

    # true parameters and virtual IMU positions
    m_true   = np.random.uniform(-4,4,5)
    s_vals   = [0.0, 0.3, 0.7, 1.0]
    L_phys   = 100.0
    gamma_int= 26
    σ_rot    = 0.005          # rad
    T_steps  = 500

    # ground-truth rotations
    R_true,_ = forward_kinematics_multiple(m_true, s_vals,
                                           gamma=gamma_int, L=L_phys)

    # generate noisy quaternion measurements
    meas_seq = []
    for _ in range(T_steps):
        q_list=[]
        for Rg in R_true:
            δw  = σ_rot*np.random.randn(3)
            Rn  = expm(skew(δw)) @ Rg
            q   = R.from_matrix(Rn).as_quat()
            if q[3] < 0: q = -q        # keep scalar part positive
            q_list.append(q)
        q_list[0] = np.array([0,0,0,1])  # base frame exactly known
        meas_seq.append(q_list)

    # EKF initialisation
    m_est = np.zeros(5)
    m_est = m_true
    P_est = np.eye(5)*0.1
    Q     = np.eye(5)*1e-6
    R_cov = np.eye(3*len(s_vals))*σ_rot**2

    m_hist = np.zeros((T_steps,5))
    for t in range(T_steps):
        # predict (random walk)
        P_pred = P_est + Q
        # update
        m_est,P_est = ekf_update(m_est,P_pred, s_vals,
                                 meas_seq[t], R_cov,
                                 gamma=gamma_int,L=L_phys)
        m_hist[t]=m_est

    # ─── plots ───────────────────────────────────────────────
    fig,axs = plt.subplots(5,1,sharex=True,figsize=(6,10))
    t = np.arange(T_steps)
    for i in range(5):
        axs[i].plot(t,m_hist[:,i],label=f'est m[{i}]')
        axs[i].axhline(m_true[i],ls='--',label='true')
        axs[i].legend(loc='upper right')
    axs[-1].set_xlabel('time step'); plt.tight_layout()

    # final shape vs true
    plot_3d_curves(m_true, m_est, s_vals, gamma=gamma_int, L=L_phys)
    # s_dense=np.linspace(0,1,100)
    # _,p_true=forward_kinematics_multiple(m_true,s_dense,gamma=gamma_int,L=L_phys)
    # _,p_est =forward_kinematics_multiple(m_est ,s_dense,gamma=gamma_int,L=L_phys)
    # p_true,p_est=np.asarray(p_true),np.asarray(p_est)

    # fig=plt.figure(); ax=fig.add_subplot(111,projection='3d')
    # ax.plot(*p_true.T,label='true'); ax.plot(*p_est.T,'--',label='est')
    # ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.legend()
    # plt.show()
