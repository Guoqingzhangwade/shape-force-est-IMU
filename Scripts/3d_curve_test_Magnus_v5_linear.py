import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

##############################################################################
# 1) 5-Parameter Curvature Model
##############################################################################
#
#   kappa_x(s) = m[0] + m[1]*s
#   kappa_y(s) = m[2] + m[3]*s
#   kappa_z(s) = m[4]   (constant)
#
# => total 5 parameters: m = [m0, m1, m2, m3, m4]

def curvature_kappa(s, m):
    """
    Return kappa_x, kappa_y, kappa_z at position s.
    m: array of length 5 => [m0, m1, m2, m3, m4].
    """
    kx = m[0] + m[1]*s
    ky = m[2] + m[3]*s
    kz = m[4]
    return np.array([kx, ky, kz])

def twist_matrix(kappa):
    """
    4x4 se(3) matrix from curvature kappa (3-vector).
    twist = [ [skew(kappa), e3], [0,0] ],
    with e3 = [0,0,1].
    """
    def skew(u):
        return np.array([
            [0,     -u[2],   u[1]],
            [u[2],   0,     -u[0]],
            [-u[1],  u[0],   0   ]
        ])
    e3 = np.array([0,0,1])
    top_left = skew(kappa)
    top_block = np.hstack([top_left, e3.reshape(3,1)])
    bottom_block = np.array([[0,0,0,0]])
    return np.vstack([top_block, bottom_block])

##############################################################################
# 2) 4th-order Magnus subinterval + product of exponentials
##############################################################################

def magnus_4th_subinterval(s_start, s_end, m):
    """
    4th-order Magnus for [s_start, s_end], 2-pt Gauss-Legendre.
    Return Psi_k in 4x4 se(3).
    """
    h = s_end - s_start
    c1 = s_start + h*(0.5 - np.sqrt(3)/6)
    c2 = s_start + h*(0.5 + np.sqrt(3)/6)

    kappa1 = curvature_kappa(c1, m)
    kappa2 = curvature_kappa(c2, m)
    eta1 = twist_matrix(kappa1)
    eta2 = twist_matrix(kappa2)

    # standard 4th-order formula:
    part1 = (h/2)*(eta1 + eta2)
    part2 = (h**2 * np.sqrt(3)/12)*(eta1@eta2 - eta2@eta1)
    return part1 + part2

def product_of_exponentials(m, s, gamma=10, L=100.0):
    """
    Integrate T(s) from 0..s in gamma subintervals, each with magnus_4th_subinterval.
    Then scale translation by L.
    """
    d_sub = np.linspace(0, s, gamma+1)
    T = np.eye(4)
    for k in range(1, gamma+1):
        Psi_k = magnus_4th_subinterval(d_sub[k-1], d_sub[k], m)
        T = T @ expm(Psi_k)
    T[:3,3] *= L
    return T

def forward_kinematics_multiple(m, s_values, gamma=10, L=100.0):
    rots = []
    poss = []
    for s in s_values:
        T_s = product_of_exponentials(m, s, gamma, L)
        rots.append(T_s[:3,:3])
        poss.append(T_s[:3,3])
    return rots, poss

##############################################################################
# 3) Partial Derivatives wrt m
##############################################################################

def partial_kappa_wrt_mi(s, iParam):
    """
    We have 5 params => iParam in [0..4].
      kappa_x(s) = m[0] + m[1]*s
      kappa_y(s) = m[2] + m[3]*s
      kappa_z(s) = m[4]

    => partial wrt m[0] => [1,0,0]
       partial wrt m[1] => [s,0,0]
       partial wrt m[2] => [0,1,0]
       partial wrt m[3] => [0,s,0]
       partial wrt m[4] => [0,0,1]
    """
    out = np.zeros(3)
    if iParam==0:
        out[0] = 1.0
    elif iParam==1:
        out[0] = s
    elif iParam==2:
        out[1] = 1.0
    elif iParam==3:
        out[1] = s
    else:
        out[2] = 1.0
    return out

def partial_twist_wrt_mi(s, iParam):
    """
    partial( twist_matrix(kappa(s)) ) wrt m[iParam].
    twist_matrix(kappa) => [ [skew(kappa), e3], [0,0] ].
    partial => [ [ skew(dkappa), 0], [0,0] ].
    """
    dkappa = partial_kappa_wrt_mi(s, iParam)
    def skew(u):
        return np.array([
            [0,    -u[2],  u[1]],
            [u[2],   0,   -u[0]],
            [-u[1],  u[0],  0  ]
        ])
    dSkew = skew(dkappa)
    top_block = np.hstack([dSkew, np.zeros((3,1))])
    bottom_block = np.array([[0,0,0,0]])
    return np.vstack([top_block, bottom_block])

def partial_magnus_4th_subinterval(s_start, s_end, m, iParam):
    """
    partial( Psi_k ) wrt m[iParam] in [s_start, s_end].
    Psi_k = (h/2)(eta1+eta2) + (h^2 sqrt(3)/12)(eta1@eta2 - eta2@eta1).
    """
    h = s_end - s_start
    c1 = s_start + h*(0.5 - np.sqrt(3)/6)
    c2 = s_start + h*(0.5 + np.sqrt(3)/6)

    eta1 = twist_matrix(curvature_kappa(c1, m))
    eta2 = twist_matrix(curvature_kappa(c2, m))

    dEta1 = partial_twist_wrt_mi(c1, iParam)
    dEta2 = partial_twist_wrt_mi(c2, iParam)

    part1 = (h/2)*(dEta1 + dEta2)
    part2 = (h**2*np.sqrt(3)/12)*(
        (dEta1@eta2 + eta1@dEta2) - (dEta2@eta1 + eta2@dEta1)
    )
    return part1 + part2

def dexp_negA_approx(A, dA):
    """
    Placeholder for dexp_{-A}(dA). 
    A real approach would implement your Bernoulli expansions for large angles.
    """
    return dA

def partial_expm(A, dA):
    """
    partial( e^A ) => e^A * dexp_{-A}(dA).
    We'll do naive => e^A @ dA.
    """
    expA = expm(A)
    dA_tilde = dexp_negA_approx(A, dA)
    return expA @ dA_tilde

def partial_product_of_exponentials(m, s, iParam, gamma=10):
    """
    partial( T(s) ) wrt m[iParam].
    T(s) = product_{k=1..gamma} exp( Psi_k ).
    """
    d_sub = np.linspace(0, s, gamma+1)
    E_list = []
    Psi_list = []
    for k in range(1, gamma+1):
        A_k = magnus_4th_subinterval(d_sub[k-1], d_sub[k], m)
        E_list.append(expm(A_k))
        Psi_list.append(A_k)

    partial_T = np.zeros((4,4))
    for j in range(gamma):
        left = np.eye(4)
        for idx in range(j):
            left = left @ E_list[idx]
        dA_j = partial_magnus_4th_subinterval(d_sub[j], d_sub[j+1], m, iParam)
        dExp_j = partial_expm(Psi_list[j], dA_j)
        right = np.eye(4)
        for idx in range(j+1, gamma):
            right = right @ E_list[idx]
        partial_T += left @ dExp_j @ right
    return partial_T

##############################################################################
# 4) Orientation-Angle Cost & Gradient
##############################################################################

def orientation_angle_and_gradM(R_pred, R_obs):
    """
    angle = arccos( (trace(M)-1)/2 ), M = R_pred R_obs^T
    => return angle, partial(angle)/partial(M) in 3x3
    """
    M = R_pred @ R_obs.T
    tr_val = np.trace(M)
    cos_val = (tr_val -1)/2
    cos_val = np.clip(cos_val, -1, 1)
    angle = np.arccos(cos_val)
    if abs(cos_val) > 0.99999:
        dAngle_dM = np.zeros((3,3))
    else:
        scale = -1.0/(2.0 * np.sqrt(1 - cos_val**2))
        dAngle_dM = scale * np.eye(3)
    return angle, dAngle_dM

def cost_and_grad(m, s_values, R_obs_list, gamma=10, L=100.0):
    """
    Return (total_cost, grad[5]) for 5 param shape.
    """
    total_cost = 0.0
    grad = np.zeros(5)
    for (R_obs, s_val) in zip(R_obs_list, s_values):
        T_s = product_of_exponentials(m, s_val, gamma, L)
        R_pred = T_s[:3,:3]
        angle_i, dAngle_dM = orientation_angle_and_gradM(R_pred, R_obs)
        total_cost += angle_i
        # chain rule: M=R_pred R_obs^T => dM = dR_pred @ R_obs^T
        for iParam in range(5):
            dT_s = partial_product_of_exponentials(m, s_val, iParam, gamma)
            dR_pred = dT_s[:3,:3]
            dM = dR_pred @ R_obs.T
            contrib = np.sum(dAngle_dM * dM)
            grad[iParam] += contrib
    return total_cost, grad

def objective_with_grad(m, s_values, R_obs_list, gamma=10, L=100.0):
    c, g = cost_and_grad(m, s_values, R_obs_list, gamma, L)
    return c, g

def solve_with_analytic_gradient(m_init, s_values, R_obs_list, gamma=10, L=100.0):
    def fun_and_jac(m_vec):
        c, grad_ = objective_with_grad(m_vec, s_values, R_obs_list, gamma, L)
        return c, grad_
    res = minimize(
        fun_and_jac,
        m_init,
        method='BFGS',
        jac=True,
        options={'maxiter':200, 'disp':True}
    )
    return res

##############################################################################
# 5) Plot
##############################################################################

def plot_3d_curves(m_true, m_est, s_values, gamma=10, L=100.0):
    s_dense = np.linspace(0,1,100)
    rots_true, pos_true = forward_kinematics_multiple(m_true, s_dense, gamma, L)
    rots_est,  pos_est  = forward_kinematics_multiple(m_est,  s_dense, gamma, L)
    pos_true = np.array(pos_true)
    pos_est  = np.array(pos_est)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_true[:,0], pos_true[:,1], pos_true[:,2],
            color='blue', label='True shape')
    ax.plot(pos_est[:,0],  pos_est[:,1],  pos_est[:,2],
            color='green', linestyle='--', label='Est shape')

    rotsT, posT = forward_kinematics_multiple(m_true, s_values, gamma, L)
    rotsE, posE = forward_kinematics_multiple(m_est,  s_values, gamma, L)
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

##############################################################################
#                             Main
##############################################################################

if __name__=="__main__":
    np.random.seed(23)

    # 1) True shape: 5 param => [m0, m1, m2, m3, m4]
    #    kappa_x(s) = m0 + m1*s
    #    kappa_y(s) = m2 + m3*s
    #    kappa_z(s) = m4
    m_true = np.random.uniform(-4,4,5)

    # 2) measurement points
    s_values = [0.0, 0.3, 0.7, 1.0]

    # 3) Observed rotations w/ small noise, except base => identity
    rotsT, posT = forward_kinematics_multiple(m_true, s_values, gamma=26, L=100.0)
    R_obs_list = []
    noise_level = 0.001
    for i, R_ in enumerate(rotsT):
        if i==0:
            R_obs_list.append(np.eye(3))
        else:
            R_obs_list.append(R_ + noise_level*np.random.randn(3,3))

    # 4) Solve w/ analytic gradient
    m_init = np.zeros(5)
    res = solve_with_analytic_gradient(m_init, s_values, R_obs_list, gamma=26, L=100.0)
    print("\nSolver result:", res.message)
    print("Final cost:", res.fun)
    print("True m:", m_true)
    print("Est. m:", res.x)

    # 5) Plot
    plot_3d_curves(m_true, res.x, s_values, gamma=26, L=100.0)
