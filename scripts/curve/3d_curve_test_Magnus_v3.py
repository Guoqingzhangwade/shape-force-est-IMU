import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

##############################################################################
#                          Polynomial Curvature (7 param)
##############################################################################

def phi_func(s):
    """
    3 x 7 matrix.

    x-axis curvature: m[0] + m[1]*s + m[2]*s^2
    y-axis curvature: m[3] + m[4]*s + m[5]*s^2
    z-axis torsion:   m[6] (constant)

    => kappa_x(s) = m[0] + m[1]*s + m[2]*s^2
       kappa_y(s) = m[3] + m[4]*s + m[5]*s^2
       kappa_z(s) = m[6]
    """
    return np.array([
        [1, s, s**2, 0, 0,   0,   0],
        [0, 0,    0,  1, s, s**2, 0],
        [0, 0,    0,  0, 0,   0,   1]
    ])

def curvature_kappa(s, m):
    """
    kappa(s) = phi_func(s) @ m, where m in R^7
    """
    return phi_func(s) @ m

def skew(u):
    return np.array([
        [0,    -u[2],  u[1]],
        [u[2],   0,   -u[0]],
        [-u[1],  u[0],  0  ]
    ])

def twist_matrix(kappa):
    """
    4x4 twist in se(3):
      [ skew(kappa), e3 ]
      [    0,         0 ]
    where e3 = [0,0,1].
    """
    e3 = np.array([0,0,1])
    top_left = skew(kappa)
    top_block = np.hstack([top_left, e3.reshape(3,1)])
    bot_block = np.array([[0,0,0,0]])
    return np.vstack([top_block, bot_block])

##############################################################################
#               4th-order Magnus + Subdivision (Forward Kinematics)
##############################################################################

def magnus_4th_subinterval(s_start, s_end, m):
    """
    4th-order Magnus sub-step in [s_start, s_end], 2-pt Gauss-Legendre.
    Return Psi_k as 4x4 in se(3).
    """
    h = s_end - s_start
    c1 = s_start + h*(0.5 - np.sqrt(3)/6)
    c2 = s_start + h*(0.5 + np.sqrt(3)/6)

    kappa1 = curvature_kappa(c1, m)
    kappa2 = curvature_kappa(c2, m)

    eta1 = twist_matrix(kappa1)
    eta2 = twist_matrix(kappa2)

    part1 = (h/2)*(eta1 + eta2)
    part2 = (h**2 * np.sqrt(3)/12)*(eta1@eta2 - eta2@eta1)
    return part1 + part2

def product_of_exponentials(m, s, gamma=10, L=100.0):
    """
    Integrate from 0..s via gamma subintervals, each 4th-order Magnus.
    Then scale T[:3,3] by L for mm output.
    T(s) in SE(3).
    """
    d_sub = np.linspace(0, s, gamma+1)
    T = np.eye(4)
    for k in range(1, gamma+1):
        Psi_k = magnus_4th_subinterval(d_sub[k-1], d_sub[k], m)
        T = T @ expm(Psi_k)
    # scale translation
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
#          Partial Derivatives of the 4th-order Magnus (analytic gradient)
##############################################################################

def partial_kappa_wrt_mi(s, iParam):
    """
    partial(kappa) wrt m[iParam].
    We have 7 params => iParam in [0..6].

    - iParam in [0..2] => x curvature
    - iParam in [3..5] => y curvature
    - iParam == 6     => z curvature

    kappa_x(s) = m[0] + m[1]*s + m[2]*s^2
    kappa_y(s) = m[3] + m[4]*s + m[5]*s^2
    kappa_z(s) = m[6] (constant)
    """
    out = np.zeros(3)
    if iParam < 3:
        # x
        sub_i = iParam
        if sub_i==0:
            out[0] = 1
        elif sub_i==1:
            out[0] = s
        else:
            out[0] = s**2
    elif iParam < 6:
        # y
        sub_i = iParam - 3
        if sub_i==0:
            out[1] = 1
        elif sub_i==1:
            out[1] = s
        else:
            out[1] = s**2
    else:
        # z => iParam=6
        out[2] = 1.0
    return out

def partial_twist_wrt_mi(s, iParam):
    """
    partial( twist_matrix(kappa) ) wrt m[iParam].
    twist_matrix(kappa) = [ [skew(kappa), e3], [0,0] ].

    => partial( twist ) = [ [ skew(dkappa),  0 ], [0,0] ]
    """
    dkappa = partial_kappa_wrt_mi(s, iParam)
    d_skew = skew(dkappa)
    top = np.hstack([d_skew, np.zeros((3,1))])
    bot = np.array([[0,0,0,0]])
    return np.vstack([top, bot])

def partial_magnus_4th_subinterval(s_start, s_end, m, iParam):
    """
    partial( Psi_k ) wrt m[iParam] in subinterval [s_start, s_end].
    """
    h = s_end - s_start
    c1 = s_start + h*(0.5 - np.sqrt(3)/6)
    c2 = s_start + h*(0.5 + np.sqrt(3)/6)

    # normal
    kappa1 = curvature_kappa(c1, m)
    kappa2 = curvature_kappa(c2, m)
    eta1 = twist_matrix(kappa1)
    eta2 = twist_matrix(kappa2)
    # partial
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
    For a real system, you might use a Bernoulli series expansion.
    """
    return dA  # naive

def partial_expm(A, dA):
    """
    partial( e^A ) in direction dA => e^A * dexp_{-A}(dA).
    """
    eA = expm(A)
    dA_tilde = dexp_negA_approx(A, dA)
    return eA @ dA_tilde

def partial_product_of_exponentials(m, s, iParam, gamma=10):
    """
    partial( T(s) ) wrt m[iParam], T(s) = product_{k=1..gamma} exp(Psi_k).
    """
    d_sub = np.linspace(0, s, gamma+1)
    # first gather each exp(Psi_k)
    E_list = []
    Psi_list = []
    for k in range(1, gamma+1):
        A_k = magnus_4th_subinterval(d_sub[k-1], d_sub[k], m)
        E_list.append(expm(A_k))
        Psi_list.append(A_k)

    partial_T = np.zeros((4,4))
    for j in range(gamma):
        # left product
        left = np.eye(4)
        for idx in range(j):
            left = left @ E_list[idx]

        # partial(A_j)
        dA_j = partial_magnus_4th_subinterval(d_sub[j], d_sub[j+1], m, iParam)
        # partialExp_j
        dExp_j = partial_expm(Psi_list[j], dA_j)

        # right product
        right = np.eye(4)
        for idx in range(j+1, gamma):
            right = right @ E_list[idx]

        partial_T += left @ dExp_j @ right
    return partial_T

##############################################################################
#               Orientation Angle Cost + Jacobian wrt m
##############################################################################

def orientation_angle_and_gradM(R_pred, R_obs):
    """
    Return (angle, dAngle/dM) with M = R_pred R_obs^T, in a simplified manner.
    angle = arccos( (trace(M)-1)/2 ).
    dAngle/dM => 3x3
    """
    M = R_pred @ R_obs.T
    tr_val = np.trace(M)
    cos_val = (tr_val - 1)/2
    cos_val = np.clip(cos_val, -1, 1)
    angle = np.arccos(cos_val)
    if abs(cos_val) > 0.999999:
        dAngle_dM = np.zeros((3,3))
    else:
        scale = -1.0/(2.*np.sqrt(1. - cos_val**2))
        dAngle_dM = scale * np.eye(3)
    return angle, dAngle_dM

def cost_and_grad(m, s_values, R_obs_list, gamma=10, L=100.0):
    """
    Sum of orientation angles. Return (cost, 7-dim gradient).
    """
    total_cost = 0.0
    grad = np.zeros(7)
    for (R_obs, s) in zip(R_obs_list, s_values):
        # forward kinematics
        T_s = product_of_exponentials(m, s, gamma, L)
        R_pred = T_s[:3,:3]

        angle_i, dAngle_dM = orientation_angle_and_gradM(R_pred, R_obs)
        total_cost += angle_i

        # chain rule: M = R_pred * R_obs^T => dM = dR_pred @ R_obs^T
        for iParam in range(7):
            dT_s = partial_product_of_exponentials(m, s, iParam, gamma)
            # (Note: scaling the translation by L doesn't affect orientation)
            dR_pred = dT_s[:3,:3]
            dM = dR_pred @ R_obs.T
            # partial(angle)/partial(m[i]) = frobenius( dAngle_dM, dM )
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
#                           Plot
##############################################################################

def plot_3d_curves(m_true, m_est, s_values, gamma=10, L=100.0):
    s_dense = np.linspace(0,1,100)
    rots_true, poss_true = forward_kinematics_multiple(m_true, s_dense, gamma, L)
    rots_est,  poss_est  = forward_kinematics_multiple(m_est,  s_dense, gamma, L)

    poss_true = np.array(poss_true)
    poss_est  = np.array(poss_est)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(poss_true[:,0], poss_true[:,1], poss_true[:,2],
            label='True shape', color='blue')
    ax.plot(poss_est[:,0],  poss_est[:,1],  poss_est[:,2],
            label='Est shape', color='green', linestyle='--')

    # frames at measurement points
    rotsT, posT = forward_kinematics_multiple(m_true, s_values, gamma, L)
    rotsE, posE = forward_kinematics_multiple(m_est,  s_values, gamma, L)
    for s_i, Rt, pt, Re, pe in zip(s_values, rotsT, posT, rotsE, posE):
        plot_frame(ax, pt, Rt, f'True s={s_i}', 'blue')
        plot_frame(ax, pe, Re, f'Est s={s_i}', 'green')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
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
#                          Main
##############################################################################

if __name__=="__main__":
    np.random.seed(25)

    # 1) True shape: 7 param
    m_true = np.random.uniform(-3,3,7)

    # 2) s_values
    s_values = [0.0, 0.3, 0.7, 1.0]

    # 3) Observed rotations (with small noise except at s=0 => identity)
    rotsT, posT = forward_kinematics_multiple(m_true, s_values, gamma=16, L=100.0)
    R_obs_list = []
    noise_level = 0.001
    for i, R_ in enumerate(rotsT):
        if i==0:
            R_obs_list.append(np.eye(3))
        else:
            R_obs_list.append(R_ + noise_level*np.random.randn(3,3))

    # 4) Solve via analytic gradient
    m_init = np.zeros(7)
    res = solve_with_analytic_gradient(m_init, s_values, R_obs_list, gamma=16, L=100.0)
    print("\nSolver result:", res.message)
    print("Final cost:", res.fun)
    print("True m:", m_true)
    print("Est. m:", res.x)

    # 5) Plot
    plot_3d_curves(m_true, res.x, s_values, gamma=16, L=100.0)
