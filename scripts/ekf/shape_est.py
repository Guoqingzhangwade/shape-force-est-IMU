# main.py

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt
import ipdb

# Random seed for reproducibility
RANDOM_SEED = 42
RANDOM_SEED = 24

# Number of time steps
NUM_STEPS = 500

# State dimension (modal coefficients)
STATE_DIM = 5

# True modal coefficients (constant state)
# TRUE_MODAL_COEFF = np.random.randn(STATE_DIM) * 0.1  # Small random coefficients

# TRUE_MODAL_COEFF = np.array([1, 0.2, 0.3, 1.5, 0.5, 0.6, 1.7, 0.8, 2])
TRUE_MODAL_COEFF = np.array([1, 0.2, 1.5, 0.5, 1.7])
# TRUE_MODAL_COEFF = np.random.uniform(-2,2,5)
# TRUE_MODAL_COEFF = np.array([1, 0.2, 1.5, 0.5, 1.7])
print(TRUE_MODAL_COEFF)
# ipdb.set_trace()

# Process noise covariance 
# set STD for different order coefficients
PROCESS_NOISE_STD_0th = 0.1  # Standard deviation of process noise 
PROCESS_NOISE_STD_1st = 0.01  # Standard deviation of process noise
PROCESS_NOISE_STD_2nd = 0.001  # Standard deviation of process noise  
# oth oder indices: 0, 3, 6; 1st order indices: 1, 4, 7; 2nd order indices: 2, 5, 8
# PROCESS_NOISE_COV = np.diag([PROCESS_NOISE_STD_0th**2, PROCESS_NOISE_STD_1st**2, PROCESS_NOISE_STD_2nd**2]*3)
PROCESS_NOISE_COV = np.diag([PROCESS_NOISE_STD_0th**2, PROCESS_NOISE_STD_1st**2, PROCESS_NOISE_STD_0th**2, PROCESS_NOISE_STD_1st**2,PROCESS_NOISE_STD_0th**2])
# PROCESS_NOISE_COV = np.diag([PROCESS_NOISE_STD**2]*STATE_DIM)

# set process noise as zero
# PROCESS_NOISE_COV = np.zeros((STATE_DIM, STATE_DIM))

# Measurement noise covariance (for the quaternion measurements)
MEASUREMENT_NOISE_STD = 0.01  # Standard deviation of measurement noise
MEASUREMENT_NOISE_COV = np.diag([MEASUREMENT_NOISE_STD**2]*3)
# ipdb.set_trace()

# Positions where IMU measurements are taken (e.g., along a beam or structure) 0.2, 0.5, 1.0

IMU_POSITIONS = np.array([0.2, 0.5, 1.0])

np.random.seed(RANDOM_SEED)

# Other simulation parameters can be added here as needed
# Define the total arc length (in mm)
L = 100.0  # 100 mm

# Define the skew-symmetric matrix for u(s)
def skew_symmetric(u):
    u = u.flatten()
    return np.array([[0, -u[2], u[1]],
                     [u[2], 0, -u[0]],
                     [-u[1], u[0], 0]])

# Define the twist matrix for eta(s)
def twist_matrix(u, e3):
    u = u.flatten()
    return np.block([[skew_symmetric(u), e3.reshape(3, 1)],
                     [np.zeros((1, 3)), 0]])


# Example modal basis function, using normalized arc length s
def phi_func(s):
    # return np.array([[1, s, s**2, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 1, s, s**2, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 1, s, s**2]])
    return np.array([[1, s, 0, 0, 0],
                     [0, 0, 1, s, 0],
                     [0, 0, 0, 0, 1]])

    # return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 1, 0, 0, 0, 0, 0],
    #                  [0, 0, 0, 0, 0, 0, 1, 0, 0]])

# Compute the product of exponentials
def product_of_exponentials(m, s, k=10):
  
    T = np.eye(4)  # Start with the identity matrix
    e3 = np.array([0, 0, L])  # Local tangent vector


    # define the number of subdivison points as k
    # the length of subinterval 
    h = s/k
    

    for i in range(k):
        s_i = i/ k * s
        psi_i = magnus_psi_i(s_i, h, m, e3)
        T = T @ expm(psi_i)  # Multiply each small transformation

    return T

# Define 4th order Magnus expansion approximantion term
def magnus_psi_i(s_i, h, m, e3):
    # calculate qudrature points
    c1 = h*(1/2 - sqrt(3)/6)
    c2 = h*(1/2 + sqrt(3)/6)

    kappa_1 = phi_func(s_i- h + c1) @ m
    kappa_2 = phi_func(s_i- h + c2) @ m

    eta_1 = twist_matrix(kappa_1, e3)
    eta_2 = twist_matrix(kappa_2, e3)

    phi_i = h/2*(eta_1 + eta_2) + h**2*sqrt(3)/12*(eta_1 @ eta_2 - eta_2 @ eta_1)

    return phi_i

# Function to compute rotation matrices and positions for a list of s_values
def forward_kinematics(m, s, k=10):
    rotation = []
    position = []
    
    T = product_of_exponentials(m, s, k)
    rotation = T[:3, :3]  # Extract the rotation matrix
    position = T[:3, 3]   # Extract the position vector
    return rotation, position

def quaternion_multiply(quaternion0, quaternion1):
    # normalizing the quaternion
    quaternion0 = quaternion0/np.linalg.norm(quaternion0)
    quaternion1 = quaternion1/np.linalg.norm(quaternion1)
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ], dtype=np.float64)



def quaternion_inverse(q):
    q = q/np.linalg.norm(q)
    w, x, y, z = q
    return np.array([w, -x, -y, -z])



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
    e3 = np.array([0, 0, L])  # Local tangent vector
    
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
        Psi_k = magnus_psi_i(s_i, h, m_coeffs, e3)  
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
        phi_c1 = phi_func(c1)
        phi_c2 = phi_func(c2)
        
        # Compute kappa at quadrature points
        kappa_1 = phi_c1 @ m_coeffs
        kappa_2 = phi_c2 @ m_coeffs
        
        # Compute eta at quadrature points
        eta_1 = twist_matrix(kappa_1, e3)
        eta_2 = twist_matrix(kappa_2, e3)
        
        # Precompute commutators
        commutator_eta = eta_1 @ eta_2 - eta_2 @ eta_1
        
        for i in range(Nm):
            # Compute derivative of kappa with respect to m_i
            dkappa_1_dmi = phi_c1[:, i]
            dkappa_2_dmi = phi_c2[:, i]
            
            # Compute derivative of eta with respect to m_i
            deta_1_dmi = twist_matrix(dkappa_1_dmi, np.zeros(3))
            deta_2_dmi = twist_matrix(dkappa_2_dmi, np.zeros(3))
            
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

def compute_theta_m_derivative(q_meas, R_matrix, m_coeffs, s, gamma):
    """
    Compute the derivative of the minimal error vector theta with respect to the modal coefficients m.

    Args:
        q_meas (numpy.ndarray): Measured quaternion [q0, q1, q2, q3] (scalar-first convention)
        R_matrix (numpy.ndarray): Predicted rotation matrix of shape (3, 3)
        m_coeffs (numpy.ndarray): Modal coefficients vector of shape (9,)
        s (float): Total arclength
        gamma (int): Number of subdivisions

    Returns:
        numpy.ndarray: Derivative of theta with respect to m, shape (3, 9)
    """
    import numpy as np

    # Step 1: Compute derivative of theta with respect to q_hat
    dtheta_dqhat = compute_dtheta_dqhat(q_meas)  # Shape: (3, 4)

    # Step 2: Compute derivative of q_hat with respect to R
    dqhat_dR = compute_q_R_derivative(R_matrix)  # Shape: (4, 3, 3)

    # Step 3: Compute derivative of R with respect to m
    dR_dm = compute_R_m_derivative(m_coeffs, s, gamma)  # Shape: (3, 3, 9)

    # Step 4: Compute derivative of theta with respect to R
    # Using Einstein summation to contract over q_hat indices
    dtheta_dR = np.einsum('ik,kjl->ijl', dtheta_dqhat, dqhat_dR)  # Shape: (3, 3, 3)

    # Step 5: Compute derivative of theta with respect to m
    # Contracting over R indices to get derivative with respect to m
    dtheta_dm = np.einsum('ijl,jln->in', dtheta_dR, dR_dm)  # Shape: (3, 9)

    return dtheta_dm


def compute_theta(q_meas, m_coeffs, s, gamma):
    """
    Compute the minimal error vector theta between the measured quaternion q_meas
    and the predicted quaternion q_hat given m_coeffs.
    """
    # Ensure the quaternions are normalized
    q_meas = q_meas / np.linalg.norm(q_meas)
    if q_meas[0] < 0:
        q_meas = -q_meas
    m_coeffs = m_coeffs.reshape(STATE_DIM, 1)  # Ensure m_coeffs is a column vector
    
    # Compute predicted quaternion q_hat
    R_pred, _ = forward_kinematics(m_coeffs, s, gamma)
    # ipdb.set_trace()
    q_hat = R.from_matrix(R_pred).as_quat(scalar_first=True)
    
    # Compute error quaternion q_e = q_meas * q_hat^{-1}
    q_hat_inv = np.array([q_hat[0], -q_hat[1], -q_hat[2], -q_hat[3]])  # Inverse of q_hat
    q_e = quaternion_multiply(q_meas, q_hat_inv)
    

    
    # Ensure q_e[0] >= 0 to avoid ambiguity
    if q_e[0] < 0:
        q_e = -q_e
  
    # Extract vector part and compute theta
    q_e_vector = q_e[1:]  # Shape: (3,)
    theta = 2 * q_e_vector  # Minimal error vector

    return theta  # Shape: (3,)

def compute_theta_m_derivative_numerical(q_meas, m_coeffs, s, gamma, delta=1e-6):
    """
    Compute the numerical derivative of theta with respect to m using finite differences.
    """
    m_coeffs = m_coeffs.reshape(STATE_DIM, 1)  # Ensure m_coeffs is a column vector
    Nm = m_coeffs.shape[0]
    dtheta_dm_num = np.zeros((3, Nm))
    theta0 = compute_theta(q_meas, m_coeffs, s, gamma)  # Reference theta
    
    for i in range(Nm):
        # Perturb m_i by delta
        m_plus = m_coeffs.copy()
        m_minus = m_coeffs.copy()
        m_plus[i, 0] += delta
        m_minus[i, 0] -= delta
        
        # Compute theta for perturbed m
        theta_plus = compute_theta(q_meas, m_plus, s, gamma)
        theta_minus = compute_theta(q_meas, m_minus, s, gamma)

        # ipdb.set_trace()
        
        # Compute numerical derivative
        dtheta_dm_num[:, i] = (theta_plus - theta_minus) / (2 * delta)
    
    return dtheta_dm_num



def validate_theta_m_derivative(q_meas, m_coeffs, s, gamma, delta=1e-6):
    """
    Validate the analytical derivative of theta with respect to m by comparing it to the numerical derivative.
    """
    m_coeffs = m_coeffs.reshape(STATE_DIM, 1)  # Ensure m_coeffs is a column vector
    
    # Compute the predicted rotation matrix
    R_matrix, _ = forward_kinematics(m_coeffs, s, gamma)
    
    # Compute the analytical derivative ∂θ/∂m
    dtheta_dm_analytical = compute_theta_m_derivative(q_meas, R_matrix, m_coeffs, s, gamma)
    
    # Compute the numerical derivative ∂θ/∂m
    dtheta_dm_numerical = compute_theta_m_derivative_numerical(q_meas, m_coeffs, s, gamma, delta)
    
    # Compute relative error
    diff = dtheta_dm_analytical - dtheta_dm_numerical
    relative_error = np.linalg.norm(diff) / np.linalg.norm(dtheta_dm_numerical)
    
    print("Relative error between analytical and numerical derivatives of theta w.r.t m:", relative_error)
    
    # Optionally, print the analytical and numerical derivatives for comparison
    print("\nAnalytical derivative dtheta/dm:")
    print(dtheta_dm_analytical)
    
    print("\nNumerical derivative dtheta/dm:")
    print(dtheta_dm_numerical)



# # Test the derivative computation
m = np.random.randn(STATE_DIM)*0.1
s = 1.0
gamma = 20
R_true, _ = forward_kinematics(m, s, k=10)
q_true = R.from_matrix(R_true).as_quat(scalar_first=True)
if q_true[0] < 0:
    q_true = -q_true
q_meas = q_true + np.random.randn(4) * 0.01
q_meas = q_meas/np.linalg.norm(q_meas)
if q_meas[0] < 0:
    q_meas = -q_meas
validate_theta_m_derivative(q_meas, m, s, gamma, delta=1e-7)
# ipdb.set_trace()

# # test the function
# m = np.random.randn(9)
# s = 1.0
# gamma = 10
# R_true, _ = forward_kinematics(m, s, k=10)
# q_true = R.from_matrix(R_true).as_quat(scalar_first=True)
# q_meas = q_true + np.random.randn(4) * 0.01
# dtheta_dm = compute_theta_m_derivative(q_meas, R_true, m, s, gamma)
# print(dtheta_dm)
# ipdb.set_trace()




# True modal coefficients over time (constant)
modal_coeffs_true = np.tile(TRUE_MODAL_COEFF, (NUM_STEPS, 1))

# Generate measurements
measurements = []
for k in range(NUM_STEPS):
    quat_measurements = []
    for s in IMU_POSITIONS:
        R_true, _ = forward_kinematics(TRUE_MODAL_COEFF, s, k=20)
        # quaternion with scalar-first convention
        q_true = R.from_matrix(R_true).as_quat(scalar_first=True)
        if q_true[0] < 0:
            q_true = -q_true
        noise_rotvec = np.random.multivariate_normal(np.zeros(3), MEASUREMENT_NOISE_COV)
        noise_rotation = R.from_rotvec(noise_rotvec)
        q_meas = R.from_quat(q_true, scalar_first=True) * noise_rotation
        # q_meas = R.from_quat(q_true, scalar_first=True) 
        q_meas = q_meas.as_quat(scalar_first=True)
        q_meas = q_meas/np.linalg.norm(q_meas)
        if q_meas[0] < 0:
            q_meas = -q_meas
        # q_meas = R.from_quat(q_true, scalar_first=True) 
        quat_measurements.append(q_meas)
    measurements.append(quat_measurements)


# EKF Initialization
# m_est = np.ones(STATE_DIM)
# m_est = np.random.randn(STATE_DIM) * 0.101
m_est = np.array([1, 0.2, 0.3, 1.5, 0.5, 0.6, 1.7, 0.8, 0.9]) * 1.3
# m_est = np.array([1, 0.0, 0.0, 1.5, 0.0, 0.0, 1.7, 0.0, 0.0]) * 1.3
# m_est = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]) * 1.3
m_est = np.array([1, 0.2, 1.5, 0.5, 1.7]) * 1.3
# m_est = np.zeros(5)



P_est = np.eye(STATE_DIM) * 0.1
Q = PROCESS_NOISE_COV
R_cov_single = MEASUREMENT_NOISE_COV

# Storage for estimates
modal_coeffs_est = []

for k in range(NUM_STEPS):
    # Prediction
    m_pred = m_est
    P_pred = P_est + Q

    # Update
    H_list = []
    residuals_list = []
    for i, s in enumerate(IMU_POSITIONS):
        R_pred, _  = forward_kinematics(m_pred, s, k=20)
        q_pred = R.from_matrix(R_pred).as_quat(scalar_first=True)
        if q_pred[0] < 0:
            q_pred = -q_pred
        q_meas = measurements[k][i]
        # q_meas_R = R.from_quat(q_meas, scalar_first=True)
        # q_pred_R = R.from_quat(q_pred, scalar_first=True)
        # q_err_R = quaternion_multiply(q_meas, quaternion_inverse(q_pred))

        # residual = 2 * q_err_R[1:]
        residual = -compute_theta(q_meas, m_pred, s, 20)
        residuals_list.append(residual)
        # H_i = compute_theta_m_derivative_numerical(q_meas, m_pred, s, 20, 1e-7)
        H_i = compute_theta_m_derivative(q_meas, R_pred, m_pred, s, 20)
        H_list.append(H_i)

    residuals = np.hstack(residuals_list)
    # ipdb.set_trace()
    H = np.vstack(H_list)
    R_cov = np.kron(np.eye(len(IMU_POSITIONS)), R_cov_single)
    # ipdb.set_trace()
    S = H @ P_pred @ H.T + R_cov
    K = P_pred @ H.T @ np.linalg.inv(S)
    # ipdb.set_trace()
    m_est = m_pred + K @ residuals
    # m_est = m_pred - K @ residuals
    P_est = (np.eye(STATE_DIM) - K @ H) @ P_pred
    modal_coeffs_est.append(m_est)



# # ───────────────────────── IEKF loop ─────────────────────────
# for k in range(NUM_STEPS):
#     # 1) prediction (model is constant → identity state-transition)
#     m_pred = m_est.copy()
#     P_pred = P_est + Q

#     # 2) iterated measurement update
#     m_iter = m_pred.copy()
#     P_iter = P_pred.copy()          # gain is recomputed around *each* linearisation
#     max_iters = 5                   # usually 3–5 is enough
#     tol       = 1e-6

#     for _ in range(max_iters):
#         H_list, r_list = [], []

#         # build stacked residual and Jacobian for every IMU
#         for s in IMU_POSITIONS:
#             R_pred, _ = forward_kinematics(m_iter, s, k=20)
#             q_meas     = measurements[k][IMU_POSITIONS.tolist().index(s)]

#             # residual   r = z − h(x)   (here: −θ because h(x)=0 in minimal-error form)
#             r  = -compute_theta(q_meas, m_iter, s, 20)
#             H  = compute_theta_m_derivative(q_meas, R_pred, m_iter, s, 20)

#             r_list.append(r)
#             H_list.append(H)

#         r = np.hstack(r_list)              # 3 × N_IMU
#         H = np.vstack(H_list)              # 3 N_IMU × STATE_DIM
#         R_big = np.kron(np.eye(len(IMU_POSITIONS)), R_cov_single)

#         S = H @ P_iter @ H.T + R_big
#         K = P_iter @ H.T @ np.linalg.inv(S)

#         delta = K @ r                      # state increment
#         m_new = m_iter + delta

#         # convergence check
#         if np.linalg.norm(delta) < tol:
#             m_iter = m_new
#             break
#         m_iter = m_new                     # re–linearise around updated state

#     # 3) covariance after final iteration (once, not inside the loop)
#     P_est = (np.eye(STATE_DIM) - K @ H) @ P_iter
#     m_est = m_iter
#     modal_coeffs_est.append(m_est)
# # ───────────────────── end IEKF loop ─────────────────────


print('True Modal Coefficients:')
print(modal_coeffs_true)
print('Estimated Modal Coefficients:')
print(modal_coeffs_est)

# Plotting
modal_coeffs_est = np.array(modal_coeffs_est)
plt.figure(figsize=(12, 8))
color_vec = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple']
for i in range(STATE_DIM):
    plt.plot(modal_coeffs_true[:, i], color=color_vec[i], label=f'True Modal Coeff {i+1}')
    plt.plot(modal_coeffs_est[:, i], '--', color=color_vec[i], label=f'Estimated Modal Coeff {i+1}')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Time Step')
plt.ylabel('Modal Coefficient Value')
plt.title('True vs Estimated Modal Coefficients')
plt.tight_layout()
plt.grid(True)






plt.show()



