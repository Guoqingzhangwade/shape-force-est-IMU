import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Forward kinematics to compute the transformation matrix given the state vector m
def forward_kinematics_single(m, s, phi_func):
    T = np.eye(4)  # Start with the identity matrix
    e3 = np.array([0, 0, 1])  # Local tangent unit vector

    # Calculate the twist based on normalized arc length s
    u_s = phi_func(s) @ m  # Calculate u(s) from m using Phi(s)
    eta_s_hat = twist_matrix(u_s, e3)  # Construct the twist matrix
    T = T @ expm(eta_s_hat * s)  # Update transformation using matrix exponential

    # Scale the position component of T by the total arc length
    T[:3, 3] *= total_arc_length

    return T

# Example modal basis function, using normalized arc length s
def phi_func(s):
    return np.array([[1, s, s**2, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, s, s**2, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, s, s**2]])

# Function to compute rotation matrices and positions for a list of s_values
def forward_kinematics_rotations(m, s_values, phi_func):
    rotations = []
    positions = []
    for s in s_values:
        T = forward_kinematics_single(m, s, phi_func)
        rotations.append(T[:3, :3])  # Extract the rotation matrix
        positions.append(T[:3, 3])   # Extract the position vector
    return rotations, positions

# Set a random true shape state m_true
np.random.seed(42)  # For reproducibility
m_true = np.random.uniform(-10, 10, 9)

# Compute the true transformations using m_true
s_values = np.linspace(0.1, 1.0, 9)  # 9 evenly spaced points along the normalized arc length
true_rotations, true_positions = forward_kinematics_rotations(m_true, s_values, phi_func)

# Add small random noise to the rotation matrices to simulate measurements
noise_level = 0.01
observed_rotations = [R + noise_level * np.random.randn(3, 3) for R in true_rotations]

# Use these noisy rotations as input to inverse kinematics
def objective_function(m, s_values, observed_rotations, phi_func):
    predicted_rotations, _ = forward_kinematics_rotations(m, s_values, phi_func)
    error = 0

    for observed, predicted in zip(observed_rotations, predicted_rotations):
        error += np.linalg.norm(observed - predicted, 'fro')  # Frobenius norm of the difference
    
    return error  # No regularization

def inverse_kinematics(s_values, observed_rotations, phi_func):
    # A small non-zero initial guess
    initial_guess = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    # Optimize the state vector m
    result = minimize(objective_function, initial_guess, args=(s_values, observed_rotations, phi_func), method='BFGS', options={'maxiter': 50000})

    # Extract the optimized state vector m
    return result.x

# Estimate the shape state vector m using the noisy observations
m_estimated = inverse_kinematics(s_values, observed_rotations, phi_func)

# Compare the true and estimated shape state vectors
print("True shape state vector m_true:")
print(m_true)

print("\nEstimated shape state vector m_estimated:")
print(m_estimated)

# Plot the true and estimated 3D curve with frames
def plot_3d_true_estimated_curves(m_true, m_estimated, s_values, phi_func):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Calculate the full 3D curve for true and estimated shapes
    s_full = np.linspace(0, 1, 100)
    _, true_positions = forward_kinematics_rotations(m_true, s_full, phi_func)
    _, estimated_positions = forward_kinematics_rotations(m_estimated, s_full, phi_func)

    # Extract positions for plotting
    true_positions = np.array(true_positions)
    estimated_positions = np.array(estimated_positions)

    ax.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], label='True 3D Curve', color='blue')
    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label='Estimated 3D Curve', color='green', linestyle='--')

    # Plot the frames at measurement points for true and estimated shapes
    true_rotations, true_positions_measure = forward_kinematics_rotations(m_true, s_values, phi_func)
    estimated_rotations, estimated_positions_measure = forward_kinematics_rotations(m_estimated, s_values, phi_func)

    for i, (s, true_rot, est_rot, true_pos, est_pos) in enumerate(zip(s_values, true_rotations, estimated_rotations, true_positions_measure, estimated_positions_measure)):
        # Plot true frame
        plot_frame(ax, true_pos, true_rot, label=f'True Frame at s={s:.1f}', color='blue', alpha=0.6)
        # Plot estimated frame
        plot_frame(ax, est_pos, est_rot, label=f'Estimated Frame at s={s:.1f}', color='green', alpha=0.6)

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
        ax.text(*origin, label, color=color)

# Helper function to set equal scaling on all axes
def set_equal_axis_scale(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    span = limits[:, 1] - limits[:, 0]
    center = np.mean(limits, axis=1)
    max_span = max(span) / 2

    ax.set_xlim3d([center[0] - max_span, center[0] + max_span])
    ax.set_ylim3d([center[1] - max_span, center[1] + max_span])
    ax.set_zlim3d([center[2] - max_span, center[2] + max_span])

# Run the plotting function to visualize true vs. estimated curves
plot_3d_true_estimated_curves(m_true, m_estimated, s_values, phi_func)



