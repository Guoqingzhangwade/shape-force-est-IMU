import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# Quaternion utility functions
def quaternion_multiply(q1, q2):
    """Hamilton product of quaternions q1 * q2 (both as [w,x,y,z])."""
    w1,x1,y1,z1 = q1;    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_conjugate(q):
    """Conjugate (inverse for unit quaternion) of quaternion [w,x,y,z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_from_matrix(R):
    """Convert 3x3 rotation matrix to unit quaternion [w,x,y,z] with w>=0."""
    # Using formula that avoids numerical issues
    w = np.sqrt(max(0, 1 + R[0,0] + R[1,1] + R[2,2])) / 2.0
    x = np.sqrt(max(0, 1 + R[0,0] - R[1,1] - R[2,2])) / 2.0
    y = np.sqrt(max(0, 1 - R[0,0] + R[1,1] - R[2,2])) / 2.0
    z = np.sqrt(max(0, 1 - R[0,0] - R[1,1] + R[2,2])) / 2.0
    # Determine correct sign for x, y, z
    if R[2,1] - R[1,2] < 0: x = -x
    if R[0,2] - R[2,0] < 0: y = -y
    if R[1,0] - R[0,1] < 0: z = -z
    if w < 0:  # enforce non-negative scalar part
        w, x, y, z = -w, -x, -y, -z
    q = np.array([w, x, y, z])
    return q / la.norm(q)

def quaternion_to_matrix(q):
    """Convert quaternion [w,x,y,z] to a 3x3 rotation matrix."""
    w,x,y,z = q
    # Rotation matrix elements
    return np.array([
        [1-2*(y**2+z**2),  2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),    1-2*(x**2+z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),    2*(y*z + x*w),   1-2*(x**2+y**2)]
    ])

def hat_matrix(v):
    """Return the skew-symmetric matrix hat(v) for v = [v1,v2,v3]."""
    vx, vy, vz = v
    return np.array([[0,   -vz,  vy],
                     [vz,   0,  -vx],
                     [-vy, vx,   0]])

# Forward kinematics with Jacobian
def forward_kinematics_with_derivatives(x, L=1.0, num_segments=60):
    """
    Compute the rotation at 3 sensor positions (1/3, 2/3, 1 of length)
    and the derivatives dR/da of those rotations w.rt state parameters.
    Returns:
      sensor_Rs: list of 3 rotation matrices at sensor positions.
      sensor_quats: list of 3 quaternions (w,x,y,z) at sensor positions.
      sensor_S_list: list of 3, each is a list of 5 matrices (3x3) = ∂R_i/∂a_j.
      positions: array of shape (num_segments+1, 3) of backbone points.
    """
    a1,a2,a3,a4,a5 = x
    # Initialize base frame
    R = np.eye(3)
    p = np.zeros(3)
    # Initialize Jacobian matrices ∂R/∂a_j at base as zero
    S_list = [np.zeros((3,3)) for _ in range(5)]
    # Storage for results
    positions = [p.copy()]
    sensor_Rs = []; sensor_quats = []; sensor_S_list = []
    sensor_positions = [L/3, 2*L/3, L]  # arc lengths of sensors
    sensor_index = 0

    def curvature(s):
        """Return u(s) = [kappa_x, kappa_y, tau] at arc-length s."""
        t = s / L
        return np.array([a1 + a3*t,  a2 + a4*t,  a5])
    def curvature_derivatives(s):
        """Return list of ∂u/∂a for each state parameter at arc-length s."""
        t = s / L
        return [
            np.array([1.0, 0.0, 0.0]),    # ∂/∂a1 of [kx,ky,tau] = [1, 0, 0]
            np.array([0.0, 1.0, 0.0]),    # ∂/∂a2 = [0, 1, 0]
            np.array([t,   0.0, 0.0]),    # ∂/∂a3 = [t, 0, 0]
            np.array([0.0, t,   0.0]),    # ∂/∂a4 = [0, t, 0]
            np.array([0.0, 0.0, 1.0])     # ∂/∂a5 = [0, 0, 1]
        ]

    s = 0.0
    ds = L / num_segments
    for i in range(1, num_segments+1):
        # Current curvature and half-step curvature
        u_prev = curvature(s)
        u_mid = curvature(s + ds/2)
        # Compute rotation for half-step and full-step using Rodrigues' formula
        w_mid = u_mid * ds
        theta = la.norm(w_mid)
        if theta < 1e-8:
            exp_mid = np.eye(3) + hat_matrix(w_mid)  # ~I + hat(w)
        else:
            k = w_mid / theta
            K = hat_matrix(k)
            exp_mid = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)
        # Update orientation
        R_new = R @ exp_mid
        # Update Jacobians via midpoint method
        # First, compute half-step orientation for use in Jacobian
        w_half = u_prev * (ds/2)
        th_half = la.norm(w_half)
        if th_half < 1e-8:
            exp_half = np.eye(3) + hat_matrix(w_half)
        else:
            k_half = w_half / th_half
            K_half = hat_matrix(k_half)
            exp_half = np.eye(3) + np.sin(th_half)*K_half + (1 - np.cos(th_half))*(K_half @ K_half)
        R_half = R @ exp_half
        # Compute curvature derivatives at s and s+ds/2
        du_prev = curvature_derivatives(s)
        du_mid  = curvature_derivatives(s + ds/2)
        # Update each Jacobian matrix S_j
        for j in range(5):
            # Euler prediction to midpoint:
            S_half = S_list[j] + (S_list[j] @ hat_matrix(u_prev) + R @ hat_matrix(du_prev[j])) * (ds/2)
            # Midpoint update:
            S_list[j] = S_list[j] + (S_half @ hat_matrix(u_mid) + R_half @ hat_matrix(du_mid[j])) * ds
        # Update position
        p = p + R @ np.array([0, 0, ds])
        R = R_new;  s += ds
        positions.append(p.copy())
        # Check if we reached a sensor location (within a tiny tolerance)
        if sensor_index < len(sensor_positions) and abs(s - sensor_positions[sensor_index]) < 1e-6:
            sensor_Rs.append(R.copy())
            sensor_quats.append(quaternion_from_matrix(R))
            # Store a copy of the current Jacobian matrices for this sensor
            sensor_S_list.append([Sj.copy() for Sj in S_list])
            sensor_index += 1

    return sensor_Rs, sensor_quats, sensor_S_list, np.array(positions)

# Simulate EKF shape estimation
np.random.seed(0)
L = 1.0  # rod length
# Ground truth initial modal coefficients
x_true = np.array([0.1, 0.05, -0.05, 0.0, 0.1])
# Initial EKF estimate (could be off)
x_est = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
P = np.eye(5) * 0.1            # initial state covariance
Q = np.eye(5) * 1e-6           # process noise covariance (small)
R_cov = np.eye(3*3) * (0.01**2)  # measurement noise cov (0.01 rad std per axis)

# Lists to log trajectories
est_history = [x_est.copy()]
true_history = [x_true.copy()]

# Time-step loop (simulate a few steps for demonstration)
for k in range(10):
    # --- Measurement generation (synthetic) ---
    # Get true orientations at sensors
    sensor_R_true, sensor_q_true, _, _ = forward_kinematics_with_derivatives(x_true, L=L)
    # Add measurement noise (small random rotation)
    q_meas_list = []
    for q in sensor_q_true:
        # small-angle noise
        noise_axis = np.random.normal(0, 0.01, 3)
        angle = la.norm(noise_axis)
        if angle > 1e-8:
            noise_axis /= angle
        dq = np.hstack(([np.cos(angle/2)], np.sin(angle/2)*noise_axis))
        dq = dq / la.norm(dq)
        # apply noise: q_meas = dq * q_true  (apply small rotation after true orientation)
        q_meas = quaternion_multiply(dq, q)
        if q_meas[0] < 0:  # keep scalar part positive for consistency
            q_meas = -q_meas
        q_meas_list.append(q_meas)
    # --- EKF Prediction ---
    x_pred = x_est.copy()            # x doesn't change (random walk)
    P_pred = P + Q                  # add process covariance
    # --- EKF Measurement Update ---
    # Get predicted orientations and Jacobians for current estimate
    sensor_R_pred, sensor_q_pred, sensor_S_pred, _ = forward_kinematics_with_derivatives(x_pred, L=L)
    # Assemble residual and Jacobian
    z_residual = np.zeros(9)
    H = np.zeros((9, 5))
    for i in range(3):  # for each sensor
        # Predicted and measured quaternion
        q_pred = sensor_q_pred[i];   q_meas = q_meas_list[i]
        # Orientation error quaternion
        q_err = quaternion_multiply(q_meas, quaternion_conjugate(q_pred))
        if q_err[0] < 0:
            q_err = -q_err
        # Residual: small-angle approx (2 * vector part of q_err)
        z_residual[3*i : 3*i+3] = 2.0 * q_err[1:4]
        # Jacobian block: compute vee( R_meas * (∂R_pred/∂a_j)^T )
        R_meas = quaternion_to_matrix(q_meas)
        for j in range(5):
            dR = sensor_S_pred[i][j]      # ∂R/∂a_j at sensor i
            A = R_meas.dot(dR.T)          # R_meas * dR^T
            # Extract axis-angle from skew-symmetric part of A
            skew = 0.5 * (A - A.T)
            H[3*i : 3*i+3, j] = np.array([skew[2,1], skew[0,2], skew[1,0]])
    # Kalman gain
    S = H.dot(P_pred).dot(H.T) + R_cov
    K = P_pred.dot(H.T).dot(la.inv(S))
    # State update
    x_est = x_pred + K.dot(z_residual)
    # Covariance update
    P = (np.eye(5) - K.dot(H)).dot(P_pred)
    # Log states
    est_history.append(x_est.copy())
    true_history.append(x_true.copy())
    # (In a real scenario, we would update x_true via system dynamics; here we keep it constant or add minimal drift)
    x_true = x_true + np.random.normal(0, 1e-5, size=5)  # small drift in truth (for simulation)

# Print final estimated vs true coefficients
print("Final true coefficients:", x_true)
print("Final estimated coefficients:", x_est)

# Visualize the backbone curve for ground truth vs. estimated
_, _, _, backbone_true = forward_kinematics_with_derivatives(x_true, L=L, num_segments=100)
_, _, _, backbone_est = forward_kinematics_with_derivatives(x_est, L=L, num_segments=100)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(backbone_true[:,0], backbone_true[:,1], backbone_true[:,2], label='Ground Truth')
ax.plot(backbone_est[:,0], backbone_est[:,1], backbone_est[:,2], label='EKF Estimate')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()
plt.show()


