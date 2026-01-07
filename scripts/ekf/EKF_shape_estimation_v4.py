# -*- coding: utf-8 -*-
"""
ekf_curvature_timeseries_quat_v4.py
-----------------------------------
Quaternion EKF for 5-parameter curvature model.
Now with sign-safe quaternions, non-singular sensor set,
and gentler gains.
"""

# ───────────── Imports ─────────────
import numpy as np, matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R

# ─────────── utilities ─────────────
def skew(u): return np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])

def quat_mul(q1,q2):
    x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
    return np.array([ w1*x2+x1*w2+y1*z2-z1*y2,
                      w1*y2-x1*z2+y1*w2+z1*x2,
                      w1*z2+x1*y2-y1*x2+z1*w2,
                      w1*w2-x1*x2-y1*y2-z1*z2 ])

def quat_conj(q): return np.array([-q[0],-q[1],-q[2], q[3]])

def q_fix_sign(q):                       # ★ keep scalar part positive
    return -q if q[3] < 0 else q

def quat_error(qm, qp):
    q_err = quat_mul(qm, quat_conj(qp))
    q_err = q_fix_sign(q_err)
    return 2*q_err[:3]                   # θ ∈ ℝ³

# J_theta_q  (3×4)
def J_theta_q(qm):
    qv, q0 = qm[:3], qm[3]
    return 2*np.hstack([-qv.reshape(3,1), -q0*np.eye(3)-skew(qv)])

# J_q_q  (4×4)
def J_q_q(qm):
    qv,q0 = qm[:3],qm[3]
    return np.vstack([ np.hstack([q0,qv]),
                       np.hstack([qv.reshape(3,1),-q0*np.eye(3)-skew(qv)]) ])

# J_q_R  (4×9)
def J_q_R(Rm):
    R11,R12,R13,R21,R22,R23,R31,R32,R33 = Rm.reshape(-1)
    q0 = 0.5*np.sqrt(max(1+R11+R22+R33,1e-12))   # num-safe
    q  = np.array([(R32-R23)/(4*q0),
                   (R13-R31)/(4*q0),
                   (R21-R12)/(4*q0),
                   q0])
    J = np.zeros((4,9))
    for idx in (0,4,8): J[3,idx] = 1/(8*q0)
    pos={(0,7),(1,2),(2,3)}; neg={(0,5),(1,6),(2,1)}
    for n in range(3):
        for idx in range(9):
            if (n,idx) in pos:   J[n,idx]= 1/(4*q0)-q[n]/(8*q0**2)
            elif (n,idx) in neg:J[n,idx]=-1/(4*q0)-q[n]/(8*q0**2)
            elif idx in (0,4,8): J[n,idx]=         -q[n]/(8*q0**2)
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

def dexp_first(A,dA):                    # 1st-order Bernoulli
    return dA + 0.5*(A@dA - dA@A)

def dExp_dm(A,dA):
    return expm(A) @ dexp_first(A,dA)

# def dexp_so3(A, dA):
#     """
#     Exact right-Jacobian for so(3): returns J · dA
#     A, dA are 3×3 skew matrices.
#     """
#     a = np.array([A[2,1], A[0,2], A[1,0]])
#     θ = np.linalg.norm(a)
#     if θ < 1e-9:
#         return dA + 0.5*(A@dA - dA@A)  # fall back to 1st order
#     hat_a = A
#     J = ( (np.sin(θ)/θ)   * np.eye(3)
#         + ((1-np.sin(θ)/θ)) * (a[:,None]@a[None,:])/(θ**2)
#         + ((1-np.cos(θ))/θ) * hat_a/θ )
#     return J @ dA

# def dexp_se3(A,dA):
#     """
#     Cheap SE(3) lift: apply exact SO(3) Jacobian to rotation block
#     and 1st-order term to translation (good enough for κ·L <~ 0.4 rad).
#     """
#     dR = dexp_so3(A[:3,:3], dA[:3,:3])
#     dP = dA[:3,3:] + 0.5*(A[:3,:3]@dA[:3,3:] - dA[:3,:3]@A[:3,3:])
#     out = np.zeros_like(dA)
#     out[:3,:3] = dR
#     out[:3,3:] = dP
#     return out

# def dExp_dm(A,dA):
#     return expm(A) @ dexp_se3(A,dA)


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

# ───────────── H and innovation ─────────────
def measurement_jacobian(m, s_vals, q_obs, *, γ=26, L=100.0):
    H,Y=[],[]
    R_pred,_ = forward_kinematics_multiple(m,s_vals,gamma=γ,L=L)
    q_pred = [q_fix_sign(R.from_matrix(Rp).as_quat()) for Rp in R_pred]

    for s_i,qm,qp,Rp in zip(s_vals,q_obs,q_pred,R_pred):
        θ   = quat_error(qm, qp)
        H.append( J_theta_q(qm) @ J_q_q(qm) @ J_q_R(Rp) @ JRm_block(m,s_i,gamma=γ) )
        Y.append(-θ)                            # innovation = 0 − θ
    return np.vstack(H), np.hstack(Y)

# IEKF one-step corrector -----------------------------------
def ekf_update(m,P,s_vals,q_obs,R_cov,*,γ=26,L=100.0,iters=1):
    for _ in range(iters):
        H,y = measurement_jacobian(m,s_vals,q_obs,γ=γ,L=L)
        S   = H @ P @ H.T + R_cov
        K   = P @ H.T @ np.linalg.inv(S)
        m  += K @ y
        P   = (np.eye(len(m))-K@H)@P
    return m,P

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
    R_true,_ = forward_kinematics_multiple(m_true,s_vals,gamma=γ_int,L=L_phys)

    # noisy measurements
    meas_seq=[]
    for _ in range(T_steps):
        q=[]
        for Rg in R_true:
            δw = σ_rot*np.random.randn(3)
            q.append(q_fix_sign(R.from_matrix(expm(skew(δw))@Rg).as_quat()))
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
        m_est,P_est = ekf_update(m_est,P_pred,s_vals,
                                 meas_seq[t],R_cov,
                                 γ=γ_int,L=L_phys,iters=1)
        hist[t]=m_est

    # ─ plots ─
    fig,axs=plt.subplots(5,1,sharex=True,figsize=(6,10))
    for i in range(5):
        axs[i].plot(hist[:,i],label=f'est m[{i}]')
        axs[i].axhline(m_true[i],ls='--',label='true')
        axs[i].legend(loc='upper right')
    axs[-1].set_xlabel('time step'); plt.tight_layout(); plt.show()
