import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the skew-symmetric matrix for u(s)
def skew_symmetric(u):
    return np.array([[0, -u[2], u[1]],
                     [u[2], 0, -u[0]],
                     [-u[1], u[0], 0]])

# Define the twist matrix for eta(s)
def twist_matrix(u, e3):
    return np.block([[skew_symmetric(u), e3.reshape(3, 1)],
                     [np.zeros((1, 3)), 0]])

# Calculate the transformation matrix using the Lie group method and extract positions
def calculate_positions_lie_group(m, s_values, phi_func):
    T = np.eye(4)  # Start with the identity matrix
    e3 = np.array([0, 0, 1])  # Local tangent unit vector
    positions = []

    for i in range(len(s_values) - 1):
        s = s_values[i]
        delta_s = s_values[i+1] - s
        u_s = phi_func(s) @ m  # Calculate u(s) from m using Phi(s)
        eta_s_hat = twist_matrix(u_s, e3)  # Construct the twist matrix
        T = T @ expm(eta_s_hat * delta_s)  # Update transformation using matrix exponential
        positions.append(T[:3, 3])  # Extract the position component

    return np.array(positions)

# Example modal basis function, phi_func(s)
def phi_func(s):
    return np.array([[1, s, s**2, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, s, s**2, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, s, s**2]])

# Example shape state vector m
m = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3])

# Discretize the arc length s
s_values = np.linspace(0, 1, 100)

# Calculate the positions along the curve
positions = calculate_positions_lie_group(m, s_values, phi_func)

# Plotting the 3D curve
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='3D Curve')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Curve of the Shape')
ax.legend()
plt.show()
