#!/usr/bin/env python3
"""
shape_wrench_one_step.py
------------------------
6-state (linear curvature + torsion) IEKF with one-iteration tension
feedback and tip-wrench estimation for a single TDCR sample.

* Geometry & stiffness pulled from CosseratRodModel (SI units).
* Exact analytic tendon-length Jacobian J_qm (bending + twist).
* Wrench is solved in the tip body frame, then transformed to the world
  frame for comparison with simulator ground truth.
"""
from __future__ import annotations
import argparse, numpy as np
from numpy.linalg import norm, pinv
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# ────────────────── import your Cosserat-rod class ───────────────────────
from cosserat_rod_model import CosseratRodModel

ROD = CosseratRodModel(length=0.10, backbone_radius=5e-4,   # metres
                       youngs_modulus=60e9, tendon_count=4, # pascals
                       num_disks=40)

# ────────────────── derived constants (SI) ────────────────────────────────
L_PHYS     = ROD.L
EIx,EIy    = ROD.Kbt[0,0], ROD.Kbt[1,1]      # N·m²
GJ         = ROD.Kbt[2,2]                    # N·m²

STATE_DIM  = 6                               # mx0 mx1 my0 my1 mz0 mz1
IMU_S      = np.array([12/39, 25/39, 1.0])
GAMMA      = 20
PROC_STD   = np.array([1e-5,1e-6]*3)
Q          = np.diag(PROC_STD**2)
R_SINGLE   = (np.deg2rad(1.0)**2)*np.eye(3)
INIT_M     = np.zeros(STATE_DIM)
R_TAU      = (0.03**2)*np.eye(ROD.n_tendon)  # (N)²

# ───────── helper maths ───────────────────────────────────────────────────
def skew(v):
    return np.array([[  0,-v[2], v[1]],
                     [ v[2],  0,-v[0]],
                     [-v[1], v[0], 0]])

def phi(s):
    return np.array([[1,s,0,0,0,0],
                     [0,0,1,s,0,0],
                     [0,0,0,0,1,s]])

def twist(k,e3):
    return np.block([[skew(k), e3[:,None]],
                     [np.zeros((1,3)),0]])

def magnus_psi(si,h,m,e3):
    xi=np.array([.5-np.sqrt(3)/6, .5+np.sqrt(3)/6])
    c1,c2=si-h+xi*h
    k1,k2=phi(c1)@m, phi(c2)@m
    e1,e2=twist(k1,e3), twist(k2,e3)
    return (h/2)*(e1+e2)+(np.sqrt(3)/12)*h**2*(e1@e2-e2@e1)

def fwd_frame(m,s,e3,γ=GAMMA):
    T=np.eye(4); h=s/γ
    for j in range(1,γ+1):
        T@=expm(magnus_psi(j*h,h,m,e3))
    return T

# quaternion helpers -------------------------------------------------------
def q_xyzw_to_wxyz(q): x,y,z,w=q; return np.array([w,x,y,z])
def q_mul(a,b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                     w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2])
def q_inv(q): w,x,y,z=q; return np.array([w,-x,-y,-z])/np.dot(q,q)

def theta(qm,m,s,e3):
    q_pred=q_xyzw_to_wxyz(R.from_matrix(fwd_frame(m,s,e3)[:3,:3]).as_quat())
    if q_pred[0]<0:q_pred=-q_pred
    q_e=q_mul(qm,q_inv(q_pred))
    if q_e[0]<0:q_e=-q_e
    return 2*q_e[1:]

def jac_num(qm,m,s,e3,eps=1e-6):
    J=np.zeros((3,STATE_DIM))
    for i in range(STATE_DIM):
        mp,mm=m.copy(),m.copy(); mp[i]+=eps; mm[i]-=eps
        J[:,i]=(theta(qm,mp,s,e3)-theta(qm,mm,s,e3))/(2*eps)
    return J

# strain-energy gradient ----------------------------------------------------
def grad_U(m):
    mx0,mx1,my0,my1,mz0,mz1=m
    L=L_PHYS
    return L*np.array([
        EIx*(mx0+mx1/4),
        EIx*(mx0/4+mx1/3),
        EIy*(my0+my1/4),
        EIy*(my0/4+my1/3),
        GJ *(mz0+mz1/4),
        GJ *(mz0/4+mz1/3)
    ])

# modal-body Jacobian via FD -----------------------------------------------
def vee(X):
    return np.array([X[2,1],X[0,2],X[1,0],
                     X[0,3],X[1,3],X[2,3]])

def J_Vbm_fd(m,eps=1e-6):
    e3=np.array([0,0,L_PHYS])
    J=np.zeros((6,STATE_DIM))
    for i in range(STATE_DIM):
        mp,mm=m.copy(),m.copy(); mp[i]+=eps; mm[i]-=eps
        Tp=fwd_frame(mp,1.0,e3); Tm=fwd_frame(mm,1.0,e3)
        Xi=(logm(np.linalg.inv(Tm)@Tp))/(2*eps)
        J[:,i]=vee(Xi)
    return J

# tendon-length Jacobian (bending + torsion) -------------------------------
def tendon_xyz():                       # (Nt,3) in metres
    return np.asarray(ROD.r)

def J_qm_exact():
    ρ = tendon_xyz()
    x,y,z = ρ[:,0], ρ[:,1], ρ[:,2]
    L = L_PHYS
    Nt=len(x)
    J=np.zeros((Nt,STATE_DIM))
    # bending
    J[:,0]=  y*L ; J[:,1]=  y*L/2
    J[:,2]= -x*L ; J[:,3]= -x*L/2
    # torsion via axial offset
    J[:,4]=  z*L ; J[:,5]=  z*L/2
    return J                             # (Nt×6)

# tension Jacobian via FD ---------------------------------------------------
def tension_jac_fd(m,F0,J_qm,eps=1e-6):
    rhs0=grad_U(m)-J_Vbm_fd(m).T@F0
    Jτ=np.zeros((J_qm.shape[0],STATE_DIM))
    for i in range(STATE_DIM):
        mp,mm=m.copy(),m.copy(); mp[i]+=eps; mm[i]-=eps
        τp=pinv(J_qm.T, rcond=1e-4) @ (grad_U(mp)-J_Vbm_fd(mp).T@F0)
        τm=pinv(J_qm.T, rcond=1e-4) @ (grad_U(mm)-J_Vbm_fd(mm).T@F0)
        Jτ[:,i]=(τp-τm)/(2*eps)
    return Jτ

# wrench solve --------------------------------------------------------------
def solve_wrench(m,τ,J_qm):
    rhs=grad_U(m)-J_qm.T@τ
    JT =J_Vbm_fd(m).T
    return (pinv(JT, rcond=1e-4) @ rhs).ravel()

# SE(3) adjoint wrench transform -------------------------------------------
def adjoint_wrench(T, F_body):
    R = T[:3,:3]
    p = T[:3,3]
    px = np.array([[   0,-p[2], p[1]],
                   [ p[2],   0,-p[0]],
                   [-p[1], p[0], 0]])
    Ad = np.block([[R,            np.zeros((3,3))],
                   [px @ R,       R            ]])
    return Ad @ F_body

# IEKF + one tension update -------------------------------------------------
def run_one_iter_ekf(q_meas,τ_meas,max_iter=30,tol=1e-6):
    e3=np.array([0,0,L_PHYS])
    m,P=INIT_M.copy(),1e-1*np.eye(STATE_DIM)
    R_big=np.kron(np.eye(len(IMU_S)),R_SINGLE)
    J_qm=J_qm_exact()

    for _ in range(max_iter):
        m_pred,P_pred=m,P+Q
        r,H=[],[]
        for s,q in zip(IMU_S,q_meas):
            r.append(-theta(q,m_pred,s,e3))
            H.append(jac_num(q,m_pred,s,e3))
        r=np.hstack(r); H=np.vstack(H)
        S=H@P_pred@H.T+R_big
        K=P_pred@H.T@np.linalg.inv(S)
        m=m_pred+K@r; P=(np.eye(STATE_DIM)-K@H)@P_pred
        if norm(K@r)<tol: break

    F0 = solve_wrench(m,τ_meas,J_qm)
    rhs   = grad_U(m)-J_Vbm_fd(m).T@F0
    τ_pred= pinv(J_qm.T, rcond=1e-4) @ rhs
    rτ    = τ_meas-τ_pred
    Hτ    = tension_jac_fd(m,F0,J_qm)
    Sτ    = Hτ@P@Hτ.T + R_TAU
    Kτ    = P@Hτ.T@np.linalg.inv(Sτ)
    m     = m + Kτ@rτ
    P     = (np.eye(STATE_DIM)-Kτ@Hτ)@P

    F_body = solve_wrench(m,τ_meas,J_qm)
    return m, F_body

# plotting helpers ----------------------------------------------------------
def fwd_positions(m,s):
    e3=np.array([0,0,L_PHYS])
    return np.array([fwd_frame(m,si,e3)[:3,3] for si in s])

def plot_frame(ax,p,R,scale=0.02,c='k'):
    p=np.asarray(p,float); R=np.asarray(R,float)
    cols=['r','g','b'] if c=='k' else ['darkred','darkgreen','navy']
    for i in range(3):
        ax.quiver(p[0],p[1],p[2],
                  R[0,i]*scale,R[1,i]*scale,R[2,i]*scale,
                  color=cols[i],linewidth=1.1)

def set_equal(ax):
    lim=[ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()]
    span=max(b-a for a,b in lim)
    ctr=[0.5*sum(l) for l in lim]
    ax.set_xlim3d([ctr[0]-span/2,ctr[0]+span/2])
    ax.set_ylim3d([ctr[1]-span/2,ctr[1]+span/2])
    ax.set_zlim3d([ctr[2]-span/2,ctr[2]+span/2])

# ───────── main ───────────────────────────────────────────────────────────
def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--idx',type=int,default=0)
    pa.add_argument('--gt',  default='tdcr_gt_samples_100.npz')
    pa.add_argument('--meas',default='tdcr_meas_samples_100.npz')
    args=pa.parse_args()

    gt   = np.load(args.gt,  allow_pickle=True)
    meas = np.load(args.meas,allow_pickle=True)

    T_gt   = gt['T'][args.idx]               # (n_disks,4,4) world frame
    q_meas = meas['q_meas'][args.idx]        # (3,4)
    τ_meas = (meas['tau_meas'] if 'tau_meas' in meas.files
              else meas['tau'])[args.idx]    # (Nt,)

    m_hat, F_body = run_one_iter_ekf(q_meas, τ_meas)

    # transform to world for comparison
    tip_T_world = T_gt[-1]
    F_world_hat = adjoint_wrench(tip_T_world, F_body)

    print("cond(J_qm.T):", np.linalg.cond(J_qm_exact().T))
    print("Wrench est (world) :", np.round(F_world_hat,4))

    if {'f_ext','l_ext'}.issubset(gt.files):
        wrench_gt = np.hstack([gt['f_ext'][args.idx], gt['l_ext'][args.idx]])
        err = F_world_hat - wrench_gt
        print("Wrench GT  (world) :", np.round(wrench_gt,4))
        print(f"‖Force‖  error : {norm(err[:3]):.4f} N")
        print(f"‖Moment‖ error : {norm(err[3:]):.4f} N·m")

    # plot backbone
    s=np.linspace(0,1,160)
    pos_est=fwd_positions(m_hat,s)
    pos_gt =T_gt[:,:3,3]
    fig=plt.figure(figsize=(6,5)); ax=fig.add_subplot(111,projection='3d')
    ax.plot(*pos_gt.T,'k-',lw=2,label='GT')
    ax.plot(*pos_est.T,'r--',lw=2,label='IEKF+τ')
    e3=np.array([0,0,L_PHYS])
    for s_i in IMU_S:
        i=int(round(s_i*(len(T_gt)-1)))
        plot_frame(ax,T_gt[i][:3,3],T_gt[i][:3,:3],'k')
        Te=fwd_frame(m_hat,s_i,e3); plot_frame(ax,Te[:3,3],Te[:3,:3],'r')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.set_title('IEKF shape & world-frame wrench')
    ax.legend(); ax.view_init(20,-60); set_equal(ax); plt.tight_layout(); plt.show()

if __name__ == '__main__':
    main()
