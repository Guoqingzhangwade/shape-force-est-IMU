import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipdb
# import pi
from math import pi

# Define the total arc length (in mm)
total_arc_length = 70.0  # Changed from 100 mm to 40 mm

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
    T[:3, 3] *= total_arc_length  # Scale the position by the total arc length
    return T

# Example modal basis function, using normalized arc length s
def phi_func(s):
    return np.array([[1, s, s**2, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, s, s**2, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, s, s**2]])

# Compute positions and tangents along the backbone
def compute_positions_and_tangents(m, s_values, phi_func, k=10):
    positions = []
    tangents = []
    for s in s_values:
        T = forward_kinematics_single(m, s, phi_func, k)
        positions.append(T[:3, 3])   # Extract the position vector
        tangents.append(T[:3, 2])  # Extract the tangent vector
    positions = np.array(positions)
    # Compute tangents using finite differences
    # tangents = np.gradient(positions, s_values, axis=0)
    # Compute tanges directly from the 3rd column of the rotation matrix
    tangents = np.array(tangents)
    # ipdb.set_trace()
    # Normalize tangents
    # tangents = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]
    return positions, tangents

# Compute positions and rotations based on tangent vectors
def compute_positions_and_rotations(m, s_values, phi_func, k=10):
    positions, tangents = compute_positions_and_tangents(m, s_values, phi_func, k)
    rotation_matrices = []
    # for t in tangents:
    #     R = compute_frame_from_tangent(t)
    #     rotation_matrices.append(R)

    for s in s_values:
        T = forward_kinematics_single(m, s, phi_func, k)
        rotation_matrices.append(T[:3, :3])  # Extract the rotation matrix
    return positions, rotation_matrices

# Function to compute rotation matrix from tangent vector
def compute_frame_from_tangent(tangent):
    # Ensure the tangent is normalized
    t = tangent / np.linalg.norm(tangent)
    # Reference vector
    ref_vec = np.array([0, 0, 1])
    if np.allclose(t, ref_vec):
        ref_vec = np.array([0, 1, 0])
    x_axis = np.cross(ref_vec, t)
    if np.linalg.norm(x_axis) < 1e-6:
        # t is parallel to ref_vec, choose another ref_vec
        ref_vec = np.array([1, 0, 0])
        x_axis = np.cross(ref_vec, t)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(t, x_axis)
    return np.column_stack((x_axis, y_axis, t))

# # Function to plot a cylinder (disk with thickness)
# def plot_cylinder(ax, origin, rotation_matrix, radius=1.0, height=1.0, color='silver', alpha=0.4, num_points=20):
#     # Create the grid in the local frame
#     theta = np.linspace(0, 2*np.pi, num_points)
#     z = np.linspace(-height/2, height/2, 2)  # Thickness along z
#     theta_grid, z_grid = np.meshgrid(theta, z)
#     x_grid = radius * np.cos(theta_grid)
#     y_grid = radius * np.sin(theta_grid)
#     # Stack into an array
#     points = np.vstack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten()))
#     # Apply rotation
#     points_rotated = rotation_matrix @ points
#     # Translate to the origin
#     points_translated = points_rotated + origin.reshape(3,1)
#     # Reshape back to grid shape
#     X = points_translated[0].reshape(2, num_points)
#     Y = points_translated[1].reshape(2, num_points)
#     Z = points_translated[2].reshape(2, num_points)
#     # Plot the cylinder sides
#     ax.plot_surface(X, Y, Z, color=color, alpha=alpha)
#     # Plot the top and bottom faces
#     for z_offset in [-height/2, height/2]:
#         x = radius * np.cos(theta)
#         y = radius * np.sin(theta)
#         z = np.full_like(x, z_offset)
#         points = np.vstack((x, y, z))
#         points_rotated = rotation_matrix @ points
#         points_translated = points_rotated + origin.reshape(3,1)
#         ax.plot_trisurf(points_translated[0], points_translated[1], points_translated[2], color=color, alpha=alpha)


def plot_cylinder(ax, origin, rotation_matrix, radius=1.0, height=1.0, 
                             color='skyblue', alpha=0.8, edge_color='black', 
                             edge_width=1, num_points=40):
    """
    Plots a cylinder with clear top and bottom edges on a 3D Matplotlib axis.

    Parameters:
    - ax: Matplotlib 3D axis object.
    - origin: List or array of 3 elements [x, y, z] for the cylinder center.
    - rotation_matrix: 3x3 NumPy array for cylinder orientation.
    - radius: Radius of the cylinder.
    - height: Height of the cylinder.
    - color: Color of the cylinder.
    - alpha: Transparency level of the cylinder (0 to 1).
    - edge_color: Color of the edge lines.
    - edge_width: Line width of the edge lines.
    - num_points: Number of points around the circular cross-section.
    """
    # Create the grid in the local frame for the cylindrical surface
    theta = np.linspace(0, 2*np.pi, num_points)
    z = np.linspace(-height/2, height/2, 2)  # Bottom and top of the cylinder
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid)
    y_grid = radius * np.sin(theta_grid)
    
    # Stack into an array
    points = np.vstack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten()))
    
    # Apply rotation
    points_rotated = rotation_matrix @ points
    
    # Translate to the origin
    points_translated = points_rotated + np.array(origin).reshape(3,1)
    
    # Reshape back to grid shape
    X = points_translated[0].reshape(2, num_points)
    Y = points_translated[1].reshape(2, num_points)
    Z = points_translated[2].reshape(2, num_points)
    
    # Plot the cylindrical surface
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=True)
        # Plot the top and bottom faces
    for z_offset in [-height/2, height/2]:
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.full_like(x, z_offset)
        points = np.vstack((x, y, z))
        points_rotated = rotation_matrix @ points
        points_translated = points_rotated + origin.reshape(3,1)
        if z_offset == -height/2:
            surface_color = 'black'
        else:
            surface_color = color
        ax.plot_trisurf(points_translated[0], points_translated[1], points_translated[2], color=surface_color, alpha=alpha)
    
    # Function to plot circular edges
    def plot_circle(ax, center, rotation_matrix, radius, num_points=200, 
                   color='black', linewidth=1):
        theta = np.linspace(0, 2*np.pi, num_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros_like(theta)
        circle_points = np.vstack((x, y, z))
        
        # Apply rotation
        circle_rotated = rotation_matrix @ circle_points
        
        # Translate to center
        circle_translated = circle_rotated + np.array(center).reshape(3,1)
        
        # Extract coordinates
        X = circle_translated[0]
        Y = circle_translated[1]
        Z = circle_translated[2]
        
        # Plot the circle
        ax.plot(X, Y, Z, color=color, linewidth=linewidth)
    
    # Define top and bottom centers after rotation and translation
    top_center = np.array(origin) + rotation_matrix[:,2] * (height/2)
    bottom_center = np.array(origin) - rotation_matrix[:,2] * (height/2)
    
    # Plot top and bottom edges
    plot_circle(ax, top_center, rotation_matrix, radius, 
               num_points=200, color=edge_color, linewidth=edge_width)
    plot_circle(ax, bottom_center, rotation_matrix, radius, 
               num_points=200, color=edge_color, linewidth=edge_width)

# Function to compute attachment points along the backbone for each wire
def compute_wire_attachment_points(positions, rotations, attachment_radius, angles_rad):
    num_wires = len(angles_rad)
    num_points = len(positions)
    wire_points = [np.zeros((num_points, 3)) for _ in range(num_wires)]
    for i in range(num_points):
        origin = positions[i]
        rotation_matrix = rotations[i]
        # For each wire
        for w, angle_rad in enumerate(angles_rad):
            # Attachment point in local frame
            x_local = attachment_radius * np.cos(angle_rad)
            y_local = attachment_radius * np.sin(angle_rad)
            z_local = 0
            point_local = np.array([x_local, y_local, z_local])
            # Transform to global frame
            point_global = rotation_matrix @ point_local + origin
            wire_points[w][i, :] = point_global
    return wire_points  # List of arrays, one for each wire

# # Function to plot a coordinate frame at a given position and orientation
# def plot_frame(ax, origin, rotation_matrix, length=10.0, label=None, label_offset=None):
#     # Extract axes from the rotation matrix
#     x_axis = rotation_matrix[:, 0] * length
#     y_axis = rotation_matrix[:, 1] * length
#     z_axis = rotation_matrix[:, 2] * length
#     origin = origin + rotation_matrix[:, 2] * 0.2  # Adjust origin slightly along z-axis
#     # Plot the axes
#     ax.quiver(origin[0], origin[1], origin[2],
#               x_axis[0], x_axis[1], x_axis[2],
#               color='r', arrow_length_ratio=0.5, linewidth=3)
#     ax.quiver(origin[0], origin[1], origin[2],
#               y_axis[0], y_axis[1], y_axis[2],
#               color='g', arrow_length_ratio=0.5, linewidth=3)
#     ax.quiver(origin[0], origin[1], origin[2],
#               z_axis[0], z_axis[1], z_axis[2],
#               color='b', arrow_length_ratio=0.5, linewidth=3)
#     if label:
#         if label_offset is not None:
#             label_pos = origin + label_offset
#         else:
#             # Default offset along z-axis
#             label_pos = origin + z_axis * 0.2  # Adjust as needed
#         ax.text(label_pos[0], label_pos[1], label_pos[2], label, fontsize=12)

def plot_3d_arrow(ax, origin, direction, color='k', linewidth=2, 
                 shaft_radius=0.02, head_radius=0.08, head_length=0.2, resolution=20):
    """
    Plots a 3D arrow with a cylindrical shaft and a conical head.
    
    Parameters:
    - ax: Matplotlib 3D axis object.
    - origin: List or array of 3 elements [x, y, z] for the arrow start.
    - direction: List or array of 3 elements [dx, dy, dz] indicating direction and length.
    - color: Color of the arrow.
    - linewidth: Thickness of the shaft line.
    - shaft_radius: Radius of the cylindrical shaft.
    - head_radius: Radius of the conical arrowhead.
    - head_length: Length of the arrowhead.
    - resolution: Number of points to define the circular cross-section.
    """
    origin = np.array(origin)
    direction = np.array(direction)
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("Direction vector cannot be zero.")
    direction_unit = direction / length
    
    # Define shaft length (total length minus head length)
    shaft_length = length - head_length
    if shaft_length < 0:
        shaft_length = length * 0.8
        head_length = length - shaft_length
    
    # Compute end point of the shaft
    shaft_end = origin + direction_unit * shaft_length
    
    # Plot the shaft as a line
    ax.plot([origin[0], shaft_end[0]],
            [origin[1], shaft_end[1]],
            [origin[2], shaft_end[2]],
            color=color, linewidth=linewidth)
    
    # Create the arrowhead (cone)
    # Generate a cone in local coordinates
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi = np.linspace(0, np.pi / 2, resolution)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    X = head_radius * np.sin(phi_grid) * np.cos(theta_grid)
    Y = head_radius * np.sin(phi_grid) * np.sin(theta_grid)
    Z = head_length * np.cos(phi_grid)
    
    # Stack into a single array
    cone_points = np.vstack((X.flatten(), Y.flatten(), Z.flatten()))
    
    # Compute rotation matrix to align the cone with the direction vector
    # Default cone is along +Z axis
    # Find rotation axis and angle
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, direction_unit)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, direction_unit)
    if s == 0:
        R = np.eye(3)
        if c < 0:
            R = np.array([[-1, 0, 0],
                          [0, -1, 0],
                          [0, 0, 1]])
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    
    # Rotate the cone points
    cone_rotated = R @ cone_points
    
    # Translate the cone to the end of the shaft
    cone_translated = cone_rotated + shaft_end.reshape(3,1)
    
    # Reshape for surface plotting
    X_cone = cone_translated[0].reshape(resolution, resolution)
    Y_cone = cone_translated[1].reshape(resolution, resolution)
    Z_cone = cone_translated[2].reshape(resolution, resolution)
    
    # Plot the arrowhead
    ax.plot_surface(X_cone, Y_cone, Z_cone, color=color, linewidth=0, shade=True)

def plot_frame(ax, origin, rotation_matrix, length=10.0, label=None, label_offset=None,
               arrow_length=8.0, arrow_head_length=3, arrow_head_radius=0.3, linewidth=2):
    """
    Plots a coordinate frame (x, y, z axes) at a given origin and orientation.

    Parameters:
    - ax: Matplotlib 3D axis object.
    - origin: List or array of 3 elements [x, y, z] for the frame origin.
    - rotation_matrix: 3x3 NumPy array for frame orientation.
    - length: Length of the coordinate axes.
    - label: Optional label for the frame.
    - label_offset: Optional offset for the label position (array-like of 3 elements).
    - arrow_length: Length of each axis arrow.
    - arrow_head_length: Length of the arrowhead.
    - arrow_head_radius: Radius of the arrowhead.
    - linewidth: Thickness of the arrow shafts.
    """
    # Extract axes from the rotation matrix and scale by desired length
    x_axis = rotation_matrix[:, 0] * arrow_length
    y_axis = rotation_matrix[:, 1] * arrow_length
    z_axis = rotation_matrix[:, 2] * arrow_length

    # Optionally adjust the origin slightly along the z-axis for better visualization
    adjusted_origin = np.array(origin) + rotation_matrix[:, 2] * 0.2

    # Plot the axes using custom 3D arrows
    plot_3d_arrow(ax, adjusted_origin, x_axis, color='r', linewidth=linewidth, 
                 head_length=arrow_head_length, head_radius=arrow_head_radius, resolution=20)
    plot_3d_arrow(ax, adjusted_origin, y_axis, color='g', linewidth=linewidth, 
                 head_length=arrow_head_length, head_radius=arrow_head_radius, resolution=20)
    plot_3d_arrow(ax, adjusted_origin, z_axis, color='b', linewidth=linewidth, 
                 head_length=arrow_head_length, head_radius=arrow_head_radius, resolution=20)

    # Add label if provided
    if label:
        if label_offset is not None:
            label_pos = adjusted_origin + label_offset
        else:
            # Default offset: slightly beyond the z-axis arrow
            label_pos = adjusted_origin + z_axis * 1.1
        ax.text(label_pos[0], label_pos[1], label_pos[2], label, 
                fontsize=12, color='k', weight='bold')

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

# Plot the 3D curve with disks and pulling wires
def plot_3d_curve_with_disks_and_wires(m_true, phi_func, k=10):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Disk and attachment parameters
    disk_radius = 5.0  # Radius of disks
    disk_thickness = 1.0  # Thickness of disks
    attachment_radius = 3.0  # Radius for attachment points (between center and edge)
    angles = [0, 90, 180, 270]
    angles_rad = np.deg2rad(angles)

    # Compute positions and rotations at disk locations
    disk_s_values = np.linspace(0, 1, 9)  # 7 disks along the backbone
    positions_disks, rotations_disks = compute_positions_and_rotations(m_true, disk_s_values, phi_func, k)

    # Plot disks
    for origin, rotation_matrix in zip(positions_disks, rotations_disks):
        # Plot disk (as a thick cylinder)
        offset = rotation_matrix[:, 2] * (disk_thickness / 2)
        origin = origin - offset
        plot_cylinder(ax, origin, rotation_matrix, radius=disk_radius, height=disk_thickness, color='lightgrey', alpha=0.2)

    # Compute positions and rotations along the backbone for fine s values
    s_full = np.linspace(0, 1, 200)
    positions_full, rotations_full = compute_positions_and_rotations(m_true, s_full, phi_func, k)
    ax.plot(positions_full[:, 0], positions_full[:, 1], positions_full[:, 2], label='Backbone', color='k', linewidth=3)



    # Plot frames at the base, middle, and tip disks
    frame_length = 7.0  # Length of the frame axes
    label_distance = 1.1  # Multiplier for label offset along the z-axis

    # Base disk
    base_origin = positions_disks[0]
    base_rotation = rotations_disks[0]
    # Adjust the origin to the top surface of the disk
    offset = base_rotation[:, 2] * (disk_thickness / 2)
    base_origin_top = base_origin 
    # Compute label offset
    label_offset = base_rotation[:, 0] * frame_length * label_distance
    # plot_frame(ax, base_origin_top, base_rotation, length=frame_length, label='T(0)', label_offset=label_offset)
    plot_frame(ax, base_origin_top, base_rotation, length=frame_length)

    # Middle disk
    middle_index = len(positions_disks) // 2
    middle_origin = positions_disks[middle_index]
    middle_rotation = rotations_disks[middle_index]
    # Adjust the origin to the top surface of the disk
    offset = middle_rotation[:, 2] * (disk_thickness / 2)
    middle_origin_top = middle_origin 
    # Compute label offset
    label_offset = middle_rotation[:, 0] * frame_length * label_distance
    # plot_frame(ax, middle_origin_top, middle_rotation, length=frame_length, label='T(s)', label_offset=label_offset)
    plot_frame(ax, middle_origin_top, middle_rotation, length=frame_length)

    # Tip disk
    tip_origin = positions_disks[-1]
    tip_rotation = rotations_disks[-1]
    # Adjust the origin to the top surface of the disk
    offset = tip_rotation[:, 2] * (disk_thickness / 2)
    tip_origin_top = tip_origin 
    # Compute label offset
    label_offset = tip_rotation[:, 0] * frame_length * label_distance
    # plot_frame(ax, tip_origin_top, tip_rotation, length=frame_length, label='T(1)', label_offset=label_offset)
    plot_frame(ax, tip_origin_top, tip_rotation, length=frame_length)

    # World frame at the origin
    world_origin = np.array([30, -10, 0])  # Set to (0, 0, 0)
    world_rotation = np.eye(3)
    # Compute label offset
    label_offset = world_origin + np.array([frame_length * label_distance, 0, 0])
    # plot_frame(ax, world_origin, world_rotation, length=frame_length, label='World Frame{w}', label_offset=label_offset)
    plot_frame(ax, world_origin, world_rotation, length=frame_length)

    # Compute the attachment points for each wire along the backbone
    wire_points = compute_wire_attachment_points(positions_full, rotations_full, attachment_radius, angles_rad)
    # ipdb.set_trace()

    # Plot the wires as smooth curves
    for w, points in enumerate(wire_points):
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color='dodgerblue', alpha = 1, linestyle='-', linewidth=1.5, label='Wires' if w == 0 else "")

    # Set equal scaling for all axes
    set_equal_axis_scale(ax)

    # Remove grid lines and axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    # Hide axes panes
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)

    # Hide axes lines
    ax.xaxis._axinfo["axisline"]["linewidth"] = 0
    ax.yaxis._axinfo["axisline"]["linewidth"] = 0
    ax.zaxis._axinfo["axisline"]["linewidth"] = 0

    # Hide grid lines
    ax.xaxis._axinfo["grid"]["linewidth"] = 0
    ax.yaxis._axinfo["grid"]["linewidth"] = 0
    ax.zaxis._axinfo["grid"]["linewidth"] = 0

    # Hide tick labels
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    # Hide axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # Remove title if any
    ax.set_title('')

        # set limits to the axis
    ax.set_xlim(0, 30)
    ax.set_ylim(-30, 30)
    ax.set_zlim(0, 60)
    ax.set_aspect('equal')

    # off the axes
    ax.view_init(elev=20, azim=90)
    # turn off the axis
    ax.axis('off')
    plt.show()

# Main execution to plot the curve with disks and pulling wires
if __name__ == "__main__":
    # Set a random true shape state m_true
    np.random.seed(35)  # For reproducibility
    # m_true = np.random.uniform(-4, 4, 9)
    m_true = np.zeros(9)
    m_true[:6] = np.random.uniform(-4, 4, 6)

    m_true = np.array([-1, 6, -4, 1, 3, -5, 0, 0, 0])
    # Plot the 3D curve with disks and pulling wires
    plot_3d_curve_with_disks_and_wires(m_true, phi_func, k=20)













   






