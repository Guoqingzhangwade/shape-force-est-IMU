import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipdb

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
    step = s / k
    T = np.eye(4)  # Start with the identity matrix
    e3 = np.array([0, 0, 1])  # Local tangent unit vector
    
    for i in range(k):
        s_i = i * step
        u_s = phi_func(s_i) @ m
        psi_i = twist_matrix(u_s, e3) * step
        T = T @ expm(psi_i)  # Multiply each small transformation

    return T

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





# Plot the true and estimated 3D curve with frames
def plot_3d_true_estimated_curves(m_true, s_values, phi_func, k=10):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Calculate the full 3D curve for true and estimated shapes
    s_full = np.linspace(0, 1, 100)
    _, true_positions = forward_kinematics_rotations(m_true, s_full, phi_func, k)


    # Extract positions for plotting
    true_positions = np.array(true_positions)


    ax.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], label='True 3D Curve', color='blue')


    # Plot the frames at measurement points for true and estimated shapes
    true_rotations, true_positions_measure = forward_kinematics_rotations(m_true, s_values, phi_func, k)


    for i, (s, true_rot, true_pos) in enumerate(zip(s_values, true_rotations, true_positions_measure)):
        # Plot true frame
        plot_frame(ax, true_pos, true_rot, label=f'True Frame at s={s}', color='blue', alpha=0.6)
        # Plot estimated frame


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
    np.random.seed(66)  # For reproducibility
    m_true = np.random.uniform(-4, 4, 9)
    # ipdb.set_trace()
    m_true = np.array([0.5, 0.1, -0.2, 0.2, 0.3, 0.4, -0.1, -0.3, 0.1])


    # Compute the true transformations using m_true
    s_values = np.array([0.2, 0.5, 1.0])  # 3 measurement points along the normalized arc length
    true_rotations, true_positions = forward_kinematics_rotations(m_true, s_values, phi_func, k=10)

    # Add small random noise to the rotation matrices to simulate measurements
    noise_level = 0.001
    observed_rotations = [R + noise_level * np.random.randn(3, 3) for R in true_rotations]

    # Define weights, giving more dominance to lower-order coefficients (first few elements of m)
    weights = np.array([10, 5, 1, 10, 5, 1, 10, 5, 1])  # Example weights



    # Compare the true and estimated shape state vectors
    print("True shape state vector m_true:")
    print(m_true)


    # Plot the true vs. estimated 3D curves and frames
    plot_3d_true_estimated_curves(m_true, s_values, phi_func, k=10)







   






