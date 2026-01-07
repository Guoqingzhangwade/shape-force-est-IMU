import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

##############################################################################
#                          Polynomial Curvature
##############################################################################

def phi_func(s):
    """
    The 3x9 polynomial basis matrix:
      kappa_x(s) = m[0] + m[1]*s + m[2]*s^2
      kappa_y(s) = m[3] + m[4]*s + m[5]*s^2
      kappa_z(s) = m[6] + m[7]*s + m[8]*s^2
    """
    return np.array([
        [1, s, s**2,  0, 0,   0,   0, 0,   0],
        [0, 0,   0,   1, s, s**2,  0, 0,   0],
        [0, 0,   0,   0, 0,   0,   1, s, s**2]
    ])

def curvature_kappa(s, m):
    """
    Return the 3D curvature = phi_func(s) @ m
    """
    return phi_func(s) @ m

def skew(u):
    return np.array([
        [0,    -u[2],  u[1]],
        [u[2],   0,   -u[0]],
        [-u[1],  u[0], 0   ]
    ])

def twist_matrix(kappa):
    """
    Build 4x4 twist = [ [skew(kappa), e3],
                        [ 0,          0 ] ],
    with e3 = [0,0,1].
    """
    e3 = np.array([0,0,1])
    top_left = skew(kappa)
    top_block = np.hstack([top_left, e3.reshape(3,1)])
    bottom_block = np.array([[0,0,0,0]])
    return np.vstack([top_block, bottom_block])

##############################################################################
#               4th-order Magnus + Subdivision (Forward Kinematics)
##############################################################################

def magnus_4th_subinterval(s_start, s_end, m):
    """
    4th-order Magnus sub-step in [s_start, s_end], with 2-pt Gauss-Legendre.
    Return Psi (4x4 in se(3)).
    """
    h = s_end - s_start
    # c1, c2
    c1 = s_start + h*(0.5 - np.sqrt(3)/6)
    c2 = s_start + h*(0.5 + np.sqrt(3)/6)

    kappa1 = curvature_kappa(c1, m)
    kappa2 = curvature_kappa(c2, m)

    eta1 = twist_matrix(kappa1)
    eta2 = twist_matrix(kappa2)

    part1 = (h/2)*(eta1 + eta2)
    part2 = (h**2*np.sqrt(3)/12)*(eta1@eta2 - eta2@eta1)
    return part1 + part2

def product_of_exponentials(m, s, gamma=10, L=100.0):
    """
    Integrate from 0..s via gamma subintervals, 4th-order Magnus each step.
    Then scale T[:3,3] by L for actual mm.

    Return a 4x4 matrix T(s).
    """
    d_sub = np.linspace(0, s, gamma+1)
    T = np.eye(4)
    for k in range(1, gamma+1):
        Psi_k = magnus_4th_subinterval(d_sub[k-1], d_sub[k], m)
        T = T @ expm(Psi_k)
    # scale the final translation
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
#           Partial Derivatives of the 4th-order Magnus (analytic gradient)
##############################################################################

def partial_kappa_wrt_mi(s, iParam):
    """
    Return partial of kappa(s) wrt m[iParam], which is a 3D vector.
    """
    # mapping:  iParam in [0,1,2] => x part
    #           iParam in [3,4,5] => y
    #           iParam in [6,7,8] => z
    out = np.zeros(3)
    if iParam < 3:
        # x
        sub_i = iParam - 0
        if sub_i==0: out[0] = 1
        elif sub_i==1: out[0] = s
        else: out[0] = s**2
    elif iParam < 6:
        # y
        sub_i = iParam - 3
        if sub_i==0: out[1] = 1
        elif sub_i==1: out[1] = s
        else: out[1] = s**2
    else:
        # z
        sub_i = iParam - 6
        if sub_i==0: out[2] = 1
        elif sub_i==1: out[2] = s
        else: out[2] = s**2
    return out

def partial_twist_wrt_mi(s, iParam):
    """
    partial( twist(kappa) ) wrt m[iParam].
    twist(kappa) = [ [skew(kappa), e3],
                     [ 0,          0 ] ],
    partial => [ [skew(dkappa), 0],
                 [ 0,         0 ] ].
    """
    dkappa = partial_kappa_wrt_mi(s, iParam)
    block = np.hstack([skew(dkappa), np.zeros((3,1))])
    bottom = np.array([[0,0,0,0]])
    dTwist = np.vstack([block, bottom])
    return dTwist

def partial_magnus_4th_subinterval(s_start, s_end, m, iParam):
    """
    partial( Psi_k ) wrt m[iParam], where
      Psi_k = 4th-order in [s_start, s_end].
    """
    h = s_end - s_start
    c1 = s_start + h*(0.5 - np.sqrt(3)/6)
    c2 = s_start + h*(0.5 + np.sqrt(3)/6)

    # Evaluate normal twist:
    kappa1 = curvature_kappa(c1, m)
    kappa2 = curvature_kappa(c2, m)
    eta1 = twist_matrix(kappa1)
    eta2 = twist_matrix(kappa2)

    # partial wrt m_i:
    dEta1 = partial_twist_wrt_mi(c1, iParam)
    dEta2 = partial_twist_wrt_mi(c2, iParam)

    # Psi_k = (h/2)(eta1+eta2) + (h^2 sqrt(3)/12)(eta1@eta2 - eta2@eta1)
    # => dPsi = (h/2)(dEta1 + dEta2)
    #         + (h^2 sqrt(3)/12) [ (dEta1@eta2 + eta1@dEta2) - (dEta2@eta1 + eta2@dEta1) ]
    part1 = (h/2)*(dEta1 + dEta2)
    part2 = (h**2 * np.sqrt(3)/12)*(
        (dEta1@eta2 + eta1@dEta2) - (dEta2@eta1 + eta2@dEta1)
    )
    return part1 + part2

def dexp_negA_approx(A, dA):
    """
    Placeholder for dexp_{-A}(dA). 
    For small A, we can do dA or a short series. 
    A real approach would implement your Bernoulli-based expansions.
    """
    # We'll do the naive approach:  dexp_{-A}(dA) ~ dA
    # => partial_expm(A, dA) ~ exp(A) @ dA
    return dA

def partial_expm(A, dA):
    """
    partial( e^A ) in direction dA => e^A * dexp_{-A}(dA).
    """
    expA = expm(A)
    dA_tilde = dexp_negA_approx(A, dA)
    return expA @ dA_tilde

def partial_product_of_exponentials(m, s, iParam, gamma=10):
    """
    partial( T(s) ) wrt m[iParam], 
    where T(s) = product_{k=1..gamma} exp( Psi_k ), 
    Psi_k in subintervals [d_sub[k-1], d_sub[k]].
    """
    d_sub = np.linspace(0, s, gamma+1)
    # E_list to store each exp(Psi_k)
    E_list = []
    Psi_list = []
    for k in range(1, gamma+1):
        A_k = magnus_4th_subinterval(d_sub[k-1], d_sub[k], m)
        E_k = expm(A_k)
        E_list.append(E_k)
        Psi_list.append(A_k)

    # partial T = sum_j( E1..E_{j-1} * partialExp_j * E_{j+1}..E_gamma )
    # partialExp_j = e^{A_j} * dexp_{-A_j}( partial(A_j) ).
    partial_T = np.eye(4)
    # build final T, but we need partial. We'll do the product rule carefully:
    # We won't store all partialExp_j yet. We'll do a loop.

    # We can do "left products" and "right products" for each j:
    T_part = np.zeros((4,4))
    for j in range(gamma):
        # left product
        left = np.eye(4)
        for idx in range(j):
            left = left @ E_list[idx]
        # partial A_j
        dA_j = partial_magnus_4th_subinterval(d_sub[j], d_sub[j+1], m, iParam)
        # partialExp_j
        dExp_j = partial_expm(Psi_list[j], dA_j)

        # right product
        right = np.eye(4)
        for idx in range(j+1, gamma):
            right = right @ E_list[idx]

        T_part += left @ dExp_j @ right

    return T_part

##############################################################################
#          Orientation Angle Cost:  cost & gradient wrt m in R^9
##############################################################################

def orientation_angle_error_and_grad(R_pred, R_obs):
    """
    Return (angle, gradient wrt M=R_pred R_obs^T) in a naive approach.
    angle = arccos( (trace(M)-1)/2 ), M=R_pred R_obs^T
    We'll produce d(angle)/d(M) as a 3x3 matrix.
    """
    M = R_pred @ R_obs.T
    tr_val = np.trace(M)
    cos_val = (tr_val -1)/2
    cos_val = np.clip(cos_val, -1, 1)
    angle = np.arccos(cos_val)

    # derivative wrt trace(M):
    # d(angle)/d(trace) = -1/sqrt(1-cos_val^2) * 1/2
    # => -1 / (2 sqrt(1-cos^2))
    if abs(cos_val) > 0.999999:
        dAngle_dM = np.zeros((3,3))  # near singular
    else:
        scale = -1.0/(2.0 * np.sqrt(1 - cos_val**2))
        # derivative of trace(M) wrt M is identity( 1 => M[i,i]=1 ), i.e. a matrix with ones on diag.
        # But more precisely, partial trace(AB)/partial A = B^T, etc.
        # We'll approximate: dAngle_dM = scale * I_3
        dAngle_dM = scale * np.eye(3)
    return angle, dAngle_dM

def cost_and_grad(m, s_values, R_obs_list, gamma=10, L=100.0):
    """
    Returns (total_cost, grad_vector in R^9) for the sum of orientation angles
    at each s in s_values.
    """
    total_cost = 0.0
    grad = np.zeros(9)

    for (R_obs, s) in zip(R_obs_list, s_values):
        # forward kinematics => T(s)
        T_s = product_of_exponentials(m, s, gamma=gamma, L=L)
        R_pred = T_s[:3,:3]
        # angle_i, grad wrt M
        angle_i, dAngle_dM = orientation_angle_error_and_grad(R_pred, R_obs)
        total_cost += angle_i

        # chain rule: M = R_pred R_obs^T => dM = dR_pred * R_obs^T
        # so partial(angle)/partial(R_pred) => sum_{a,b} dAngle_dM[a,b]* dM[a,b]
        # => sum_{a,b} dAngle_dM[a,b] * (dR_pred[a,b]* R_obs^T[b,b]?  Actually it's a bit more 
        # "matrix multiplication" wise. We'll do a simpler approach:

        # for each iParam, partial(R_pred)/partial(m_i) => top-left 3x3 of partial_T
        for iParam in range(9):
            dT_s = partial_product_of_exponentials(m, s, iParam, gamma=gamma)
            # note we haven't scaled the partial translation, but let's recall we do T[:3,3]*=L at the end
            # => We'll replicate that step:
            # Actually we must do partial of that scaling too. But let's skip that for orientation cost. 
            dR_pred = dT_s[:3,:3]

            # dM = dR_pred * R_obs^T
            M_obsT = R_obs.T
            dM = dR_pred @ M_obsT
            # Then partial(angle)/partial(m_i) = sum( dAngle_dM * dM ) in Frobenius sense
            # => np.sum( dAngle_dM * dM )
            contrib = np.sum(dAngle_dM * dM)
            grad[iParam] += contrib

    return total_cost, grad

def objective_with_grad(m, s_values, R_obs_list, gamma=10, L=100.0):
    """
    The function passed to SciPy's minimize with jac=True.
    Return (cost, gradient_vector).
    """
    c, g = cost_and_grad(m, s_values, R_obs_list, gamma=gamma, L=L)
    return c, g

##############################################################################
#                             Optimization Demo
##############################################################################

def solve_with_analytic_gradient(m_init, s_values, R_obs_list, gamma=10, L=100.0):
    def fun_and_jac(m_vec):
        c, g = objective_with_grad(m_vec, s_values, R_obs_list, gamma=gamma, L=L)
        return c, g
    res = minimize(
        fun_and_jac,
        m_init,
        method='BFGS',
        jac=True,
        options={'maxiter':200, 'disp':True}
    )
    return res

##############################################################################
#                           Plotting Utility
##############################################################################
def plot_3d_curves(m_true, m_est, s_values, gamma=10, L=100.0):
    s_dense = np.linspace(0,1,100)
    rots_true, poss_true = forward_kinematics_multiple(m_true, s_dense, gamma, L)
    rots_est, poss_est   = forward_kinematics_multiple(m_est,  s_dense, gamma, L)

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
    for i, (s_i, R_t, p_t, R_e, p_e) in enumerate(zip(s_values, rotsT, posT, rotsE, posE)):
        plot_frame(ax, p_t, R_t, label=f'True s={s_i}', color='blue')
        plot_frame(ax, p_e, R_e, label=f'Est  s={s_i}', color='green')

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
#                           Main Test
##############################################################################
if __name__=="__main__":
    np.random.seed(23)

    # 1) Generate random "true" shape
    m_true = np.random.uniform(-3.0,3.0,9)
    

    # 2) Measurement points
    s_values = [0.0, 0.3, 0.7, 1.0]

    # 3) "Observed" rotations
    rotsT, posT = forward_kinematics_multiple(m_true, s_values, gamma=16, L=100.0)
    # add small noise except s=0 => R=I
    R_obs_list = []
    noise_level = 0.001
    for i, R_ in enumerate(rotsT):
        if i==0:
            R_obs_list.append(np.eye(3))
        else:
            R_obs_list.append(R_ + noise_level*np.random.randn(3,3))

    # 4) Solve with analytic gradient
    m_init = np.zeros(9)
    result = solve_with_analytic_gradient(m_init, s_values, R_obs_list, gamma=16, L=100.0)
    print("Final cost:", result.fun)
    print("Message:", result.message)
    print("True m:", m_true)
    print("Est  m:", result.x)

    # 5) Plot
    plot_3d_curves(m_true, result.x, s_values, gamma=16, L=100.0)
