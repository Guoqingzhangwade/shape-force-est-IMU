import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt

# Define the total arc length (in mm)
total_arc_length = 100.0  # 100 mm

# Define the skew-symmetric matrix for u(s)
def skew_symmetric(u):
    return np.array([[0, -u[2], u[1]],
                     [u[2], 0, -u[0]],
                     [-u[1], u[0], 0]])

# Define the twist matrix for eta(s)
def twist_matrix(u, e3):
    return np.block([[skew_symmetric(u), e3.reshape(3, 1)],
                     [np.zeros((1, 3)), 0]])

# Compute the product of exponentials
def product_of_exponentials(m, s, phi_func, k=10):
  
    T = np.eye(4)  # Start with the identity matrix
    e3 = np.array([0, 0, 1])  # Local tangent unit vector


    # define the number of subdivison points as k
    # the length of subinterval 
    h = s/(k-1)
    

    
    for i in range(k):
        s_i = i/ (k-1) * s
        psi_i = magnus_psi_i(s_i, h, m, e3)
        T = T @ expm(psi_i)  # Multiply each small transformation

    return T

# Define 4th order Magnus expansion approximantion term
def magnus_psi_i(s_i, h, m, e3):
    # calculate qudrature points
    c1 = h/2*(1/2 - sqrt(3)/6)
    c2 = h/2*(1/2 + sqrt(3)/6)

    kappa_1 = phi_func(s_i + c1) @ m
    kappa_2 = phi_func(s_i + c2) @ m

    eta_1 = twist_matrix(kappa_1, e3)
    eta_2 = twist_matrix(kappa_2, e3)

    phi_i = h/2*(eta_1 + eta_2) + h**2*sqrt(3)/12*(eta_1 @ eta_2 - eta_2 @ eta_1)

    return phi_i


# Forward kinematics using product of exponentials
def forward_kinematics_single(m, s, phi_func, k=10):
    T = product_of_exponentials(m, s, phi_func, k)
    T[:3, 3] *= total_arc_length  # Scale the position by the total arc length (100 mm)
    return T

# Example modal basis function, using normalized arc length s
def phi_func(s):
    return np.array([[1, s, s**2, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, s, s**2, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, s, s**2]])

# Function to compute rotation matrices and positions for a list of s_values
def forward_kinematics_rotations(m, s_values, phi_func, k=10):
    rotations = []
    positions = []
    for s in s_values:
        T = forward_kinematics_single(m, s, phi_func, k)
        rotations.append(T[:3, :3])  # Extract the rotation matrix
        positions.append(T[:3, 3])   # Extract the position vector
    return rotations, positions

# Inverse kinematics objective function
def objective_function(m, s_values, observed_rotations, phi_func, weights, k=10):
    predicted_rotations, _ = forward_kinematics_rotations(m, s_values, phi_func, k)
    error = 0

    for observed, predicted in zip(observed_rotations, predicted_rotations):
        error += np.linalg.norm(observed - predicted, 'fro')  # Frobenius norm of the difference

    # for observed, predicted in zip(observed_rotations, predicted_rotations):
    #     diff = observed - predicted
       
    #     error += np.linalg.norm(weights* diff, 'fro')  # Frobenius norm of the difference
    
    return error

# Inverse kinematics to estimate the shape state vector m
def inverse_kinematics(s_values, observed_rotations, phi_func, initial_guess=None, bounds=None, k=10):
    if initial_guess is None:
        # Use a small non-zero initial guess if not provided
        initial_guess = np.random.uniform(-0.05, 0.05, 9)  # Smaller range for better stability

    # Optimize the state vector m
    # result = minimize(objective_function, initial_guess, args=(s_values, observed_rotations, phi_func, k), method='BFGS', options={'maxiter': 500})
    # result = least_squares(objective_function, initial_guess, args=(s_values, observed_rotations, phi_func, k))
    result = minimize(objective_function, initial_guess, args=(s_values, observed_rotations, phi_func, k), method='L-BFGS-B', bounds=bounds, options={'maxiter': 500})

    # Extract the optimized state vector m
    return result.x

# Plot the true and estimated 3D curve with frames
def plot_3d_true_estimated_curves(m_true, m_estimated, s_values, phi_func, k=10):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Calculate the full 3D curve for true and estimated shapes
    s_full = np.linspace(0, 1, 100)
    _, true_positions = forward_kinematics_rotations(m_true, s_full, phi_func, k)
    _, estimated_positions = forward_kinematics_rotations(m_estimated, s_full, phi_func, k)

    # Extract positions for plotting
    true_positions = np.array(true_positions)
    estimated_positions = np.array(estimated_positions)

    ax.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], label='True 3D Curve', color='blue')
    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated 3D Curve', color='green', linestyle='--')

    # Plot the frames at measurement points for true and estimated shapes
    true_rotations, true_positions_measure = forward_kinematics_rotations(m_true, s_values, phi_func, k)
    estimated_rotations, estimated_positions_measure = forward_kinematics_rotations(m_estimated, s_values, phi_func, k)

    for i, (s, true_rot, est_rot, true_pos, est_pos) in enumerate(zip(s_values, true_rotations, estimated_rotations, true_positions_measure, estimated_positions_measure)):
        # Plot true frame
        plot_frame(ax, true_pos, true_rot, label=f'True Frame at s={s}', color='blue', alpha=0.6)
        # Plot estimated frame
        plot_frame(ax, est_pos, est_rot, label=f'Estimated Frame at s={s}', color='green', alpha=0.6)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('True vs. Estimated 3D Curve with Frames')

    # Equal scaling for all axes
    set_equal_axis_scale(ax)

    ax.legend(loc='upper right')
    plt.show()

# Helper function to plot a coordinate frame
def plot_frame(ax, origin, rotation_matrix, label=None, color='black', alpha=1.0):
    # Scale factor for the length of the axes
    scale = 5  # Adjust the scale for better visualization

    # Define the unit vectors for the axes
    x_axis = rotation_matrix[:, 0] * scale
    y_axis = rotation_matrix[:, 1] * scale
    z_axis = rotation_matrix[:, 2] * scale

    # Plot the axes as arrows
    ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2], color=color, alpha=alpha)
    ax.quiver(origin[0], origin[1], origin[2], y_axis[0], y_axis[1], y_axis[2], color=color, alpha=alpha)
    ax.quiver(origin[0], origin[1], origin[2], z_axis[0], z_axis[1], z_axis[2], color=color, alpha=alpha)

    if label:
        ax.text(origin[0], origin[1], origin[2], label, color=color)

# Helper function to set equal scaling on all axes
def set_equal_axis_scale(ax):
    # Get limits for all axes
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Calculate the span for each axis
    x_span = x_limits[1] - x_limits[0]
    y_span = y_limits[1] - y_limits[0]
    z_span = z_limits[1] - z_limits[0]

    # Find the maximum span
    max_span = max(x_span, y_span, z_span)

    # Find the center of each axis
    x_center = (x_limits[0] + x_limits[1]) / 2
    y_center = (y_limits[0] + y_limits[1]) / 2
    z_center = (z_limits[0] + z_limits[1]) / 2

    # Set the limits for each axis to be centered around the center with the max span
    ax.set_xlim3d([x_center - max_span / 2, x_center + max_span / 2])
    ax.set_ylim3d([y_center - max_span / 2, y_center + max_span / 2])
    ax.set_zlim3d([z_center - max_span / 2, z_center + max_span / 2])

# Main execution to perform inverse kinematics and plot results
if __name__ == "__main__":
    # Set a random true shape state m_true
    np.random.seed(123)  # For reproducibility
    np.random.seed(23)  # For reproducibility
    m_true = np.random.uniform(-4, 4, 9)

    # Compute the true transformations using m_true
    s_values = np.array([0.2, 0.5, 1.0])  # 3 measurement points along the normalized arc length
    true_rotations, true_positions = forward_kinematics_rotations(m_true, s_values, phi_func, k=10)

    # Add small random noise to the rotation matrices to simulate measurements
    noise_level = 0.001
    observed_rotations = [R + noise_level * np.random.randn(3, 3) for R in true_rotations]

    # Define weights, giving more dominance to lower-order coefficients (first few elements of m)
    weights = np.array([10, 5, 1, 10, 5, 1, 10, 5, 1])  # Example weights

    # Estimate the shape state vector m using the noisy observations with an improved initial guess
    initial_guess = np.random.uniform(-4, 4, 9)  # Example of a better-informed initial guess
    m_estimated = inverse_kinematics(s_values, observed_rotations, phi_func, initial_guess, bounds=[(-4, 4)] * 9)

    # Compare the true and estimated shape state vectors
    print("True shape state vector m_true:")
    print(m_true)

    print("\nEstimated shape state vector m_estimated:")
    print(m_estimated)

    # Plot the true vs. estimated 3D curves and frames
    plot_3d_true_estimated_curves(m_true, m_estimated, s_values, phi_func, k=10)







   






