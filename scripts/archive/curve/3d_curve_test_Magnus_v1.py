import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

##############################################################################
#               Polynomial Curvature + 4th-Order Magnus Integration
##############################################################################

def skew_symmetric(u):
    """
    Return the 3x3 skew-symmetric matrix from a 3D vector u.
    """
    return np.array([
        [0,    -u[2],  u[1]],
        [u[2],   0,   -u[0]],
        [-u[1],  u[0],  0  ]
    ])

def twist_matrix(curvature_vector):
    """
    Build a 4x4 twist matrix in se(3) from the 3D curvature (omega) plus the local z-axis.
    Following your text:
      eta = [ [ skew(kappa), e3 ],
              [ 0,          0   ] ],
    where e3 = [0,0,1] in normalized length units, but we will multiply by total length later.
    """
    e3 = np.array([0, 0, 1])  # local tangent direction
    top_left = skew_symmetric(curvature_vector)
    top_block = np.hstack([top_left, e3.reshape(3,1)])
    bot_block = np.array([[0,0,0,0]])
    return np.vstack([top_block, bot_block])

def phi_func(s):
    """
    Polynomial basis matrix \bs{\Phi}(s) in R^{3x9}, for:
      kappa_x(s) = m[0] + m[1]*s + m[2]*s^2
      kappa_y(s) = m[3] + m[4]*s + m[5]*s^2
      kappa_z(s) = m[6] + m[7]*s + m[8]*s^2
    """
    return np.array([
        [1,    s,    s**2,  0,   0,    0,    0,   0,    0],
        [0,    0,    0,     1,   s,    s**2, 0,   0,    0],
        [0,    0,    0,     0,   0,    0,    1,   s,    s**2]
    ])

def local_twist_eta(s, m):
    """
    Local twist in se(3), i.e. \hat(eta)(s) = [skew(kappa(s)), e3; 0, 0],
    with kappa(s) = phi_func(s)*m.
    """
    kappa = phi_func(s) @ m  # 3D curvature vector
    return twist_matrix(kappa)


def magnus_4th_subinterval(d_start, d_end, m):
    """
    Compute the 4th-order Magnus approximation \Psi in one subinterval [d_start, d_end].
    Follows eqn:
       \Psi = (h/2)*(eta1 + eta2) + (h^2*sqrt(3)/12)*(eta1@eta2 - eta2@eta1)
    where h = d_end - d_start, 
    and we pick c1, c2 in that subinterval via 2-pt Gauss-Legendre.

    Return a 4x4 matrix \Psi in se(3).
    """
    h = d_end - d_start
    # Quadrature points c1, c2 within [d_start, d_end]
    c1 = d_start + h*(0.5 - np.sqrt(3)/6)  # (1/2 - sqrt(3)/6)*h
    c2 = d_start + h*(0.5 + np.sqrt(3)/6)

    # Evaluate local twists at c1, c2
    eta1 = local_twist_eta(c1, m)
    eta2 = local_twist_eta(c2, m)

    # 4th order formula
    part1 = (h/2)*(eta1 + eta2)
    part2 = (h**2 * np.sqrt(3)/12)*(eta1 @ eta2 - eta2 @ eta1)
    return part1 + part2

def forward_kinematics_single(m, s, gamma=10, total_length=100.0):
    """
    Compute \mathbf{T}(s) from s=0 to s (normalized) using:
      - \gamma subintervals
      - 4th order magnus approx in each subinterval
      - product of exponentials of \Psi_k

    The final T[:3,3] is then scaled by 'total_length' in mm.
    """
    # Build subdivision d0..d_gamma
    # If s is in [0,1], step size = s/gamma
    d_sub = np.linspace(0, s, gamma+1)

    # Start from identity
    T = np.eye(4)

    for k in range(1, gamma+1):
        d_start = d_sub[k-1]
        d_end   = d_sub[k]
        # compute \Psi_k in subinterval
        Psi_k = magnus_4th_subinterval(d_start, d_end, m)
        # exponentiate
        expPsi_k = expm(Psi_k)
        # accumulate
        T = T @ expPsi_k

    # Finally, scale the translation by total_length
    T[:3,3] *= total_length
    return T

def forward_kinematics_rotations(m, s_values, gamma=10, total_length=100.0):
    """
    Return the rotation and position for a list of s in [0,1].
    """
    rots = []
    poss = []
    for s in s_values:
        T_s = forward_kinematics_single(m, s, gamma=gamma, total_length=total_length)
        rots.append(T_s[:3,:3])
        poss.append(T_s[:3, 3])
    return rots, poss

##############################################################################
#               Orientation-Angle Cost + Inverse Kinematics
##############################################################################

def orientation_angle_error(R_pred, R_obs):
    """
    \theta = arccos( (trace(R_pred * R_obs^T) - 1)/2 ), in [0, pi].
    """
    M = R_pred @ R_obs.T
    tr_val = np.trace(M)
    cos_val = (tr_val - 1.0)/2.0
    # clamp for numerical safety
    cos_val = max(min(cos_val, 1.0), -1.0)
    angle = np.arccos(cos_val)
    return angle

def objective_function(m, s_values, observed_rots, gamma=10, total_length=100.0):
    """
    Sum of orientation-angle errors at each measurement point.
    """
    rots_pred, _ = forward_kinematics_rotations(m, s_values, gamma=gamma, total_length=total_length)
    cost = 0.0
    for R_obs, R_pred in zip(observed_rots, rots_pred):
        cost += orientation_angle_error(R_pred, R_obs)
    return cost

def inverse_kinematics(s_values, observed_rots,
                       gamma=10, total_length=100.0,
                       init_guess=None):
    """
    Use a standard optimizer (L-BFGS-B) to find m in R^9 that minimizes
    the sum of orientation errors at s_values.
    """
    if init_guess is None:
        init_guess = np.zeros(9)  # or random, e.g. np.random.randn(9)

    def cost_fun(m_vec):
        return objective_function(m_vec, s_values, observed_rots,
                                  gamma=gamma, total_length=total_length)

    res = minimize(
        cost_fun,
        init_guess,
        method='L-BFGS-B',
        options={'maxiter':300, 'disp':True}
    )
    return res.x, res

##############################################################################
#                           Visualization
##############################################################################
def plot_3d_curves(m_true, m_est, s_values, gamma=10, total_length=100.0):
    """
    Plot the 3D backbone for the "true" shape and the "estimated" shape,
    plus frames at measurement points.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    s_dense = np.linspace(0,1,100)
    rots_true, poss_true = forward_kinematics_rotations(m_true, s_dense, gamma, total_length)
    rots_est,  poss_est  = forward_kinematics_rotations(m_est,  s_dense, gamma, total_length)

    poss_true = np.array(poss_true)
    poss_est  = np.array(poss_est)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(poss_true[:,0], poss_true[:,1], poss_true[:,2],
            label='True shape', color='blue')
    ax.plot(poss_est[:,0],  poss_est[:,1],  poss_est[:,2],
            label='Est shape', color='green', linestyle='--')

    # Also show frames at measurement s_values
    rots_true_s, poss_true_s = forward_kinematics_rotations(m_true, s_values, gamma, total_length)
    rots_est_s,  poss_est_s  = forward_kinematics_rotations(m_est,  s_values, gamma, total_length)

    for i, (s_i, R_t, p_t, R_e, p_e) in enumerate(zip(s_values, rots_true_s, poss_true_s,
                                                      rots_est_s, poss_est_s)):
        plot_frame(ax, p_t, R_t, label=f'True s={s_i:.2f}', color='blue')
        plot_frame(ax, p_e, R_e, label=f'Est  s={s_i:.2f}', color='green')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend()
    set_equal_axis(ax)
    plt.title("4th-order Magnus w/ Polynomial Curvature Model")
    plt.show()

def plot_frame(ax, origin, R, label=None, color='black'):
    """
    Plot a small coordinate frame at 'origin', orientation 'R'.
    """
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
    """
    Make the 3D axes of a plot have equal scale so that spheres appear as spheres, etc.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    spans = [x_limits[1]-x_limits[0], y_limits[1]-y_limits[0], z_limits[1]-z_limits[0]]
    max_span = max(spans)
    x_center = 0.5*(x_limits[0]+x_limits[1])
    y_center = 0.5*(y_limits[0]+y_limits[1])
    z_center = 0.5*(z_limits[0]+z_limits[1])
    ax.set_xlim3d([x_center - max_span/2, x_center + max_span/2])
    ax.set_ylim3d([y_center - max_span/2, y_center + max_span/2])
    ax.set_zlim3d([z_center - max_span/2, z_center + max_span/2])

##############################################################################
#                            Main Example
##############################################################################

if __name__=="__main__":
    np.random.seed(3)

    # 1) True shape state in R^9
    m_true = np.random.uniform(-3, 3, 9)

    # 2) Measurement points (normalized s)
    s_values = np.array([0.0, 0.3, 0.7, 1.0])

    # 3) "Observed" rotations from forward kinematics w/ small noise
    rots_true, _ = forward_kinematics_rotations(m_true, s_values, gamma=8, total_length=100.0)

    noise_level = 0.001
    observed_rots = []
    for i, R_t in enumerate(rots_true):
        if i==0:
            # for s=0, let's fix orientation = Identity exactly
            # to anchor the base
            observed_rots.append(np.eye(3))
        else:
            # add small random noise to the 3x3 matrix
            observed_rots.append(R_t + noise_level*np.random.randn(3,3))

    # 4) Solve inverse kinematics
    init_guess = np.zeros(9)
    m_est, result = inverse_kinematics(s_values, observed_rots,
                                       gamma=8, total_length=100.0,
                                       init_guess=init_guess)
    print("Solver final message:", result.message)
    print("True shape:", m_true)
    print("Estimated shape:", m_est)
    print("Final cost:", result.fun)

    # 5) Visualize
    plot_3d_curves(m_true, m_est, s_values, gamma=8, total_length=100.0)

