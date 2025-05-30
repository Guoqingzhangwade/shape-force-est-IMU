#!/usr/bin/env python3
"""
Generate synthetic sensor measurements for a single-segment TDCR.

Inputs
------
tdcr_gt_samples.npz   (the file produced by generate_tdcr_dataset.py)

Outputs
-------
tdcr_meas_samples.npz  containing
    tau_meas : (N, 4)          cable-tension measurements   [N]
    q_meas   : (N, 3, 4)       orientation quaternions at s = {0.30, 0.65, 1.0}
                                (scalar-first, unit length)
    meta     : dict            JSON-encoded meta-data

Noise model
-----------
τᵢ  ~ N(τᵢ_GT,  σ_τ²)           with σ_τ = 0.02 N
q   =  q_GT  ⊗  δq              with δq a random rotation, axis ∼ U(S²),
                                 angle ∼ N(0, σ_θ²), σ_θ = 1° (1 σ)

Author: Guoqing Zhang, 2025-05-20
"""
from __future__ import annotations
import numpy as np, argparse, json, datetime
from numpy.random import default_rng
import ipdb

# --------------------------------------------------------------------------- #
# Quaternion helpers                                                          #
# --------------------------------------------------------------------------- #
def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """axis: (3,), unit; angle: rad.  Return w,x,y,z (scalar-first)."""
    half = 0.5 * angle
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.hstack([w, xyz])

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q = q1 ⊗ q2, both scalar-first."""
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2 ])

# def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
#     """Convert 3×3 rotation matrix to scalar-first quaternion."""
#     w = np.sqrt(1.0 + np.trace(R)) / 2.0
#     if w < 1e-8:          # numerical edge
#         return np.array([1.,0.,0.,0.])
#     x = (R[2,1] - R[1,2]) / (4*w)
#     y = (R[0,2] - R[2,0]) / (4*w)
#     z = (R[1,0] - R[0,1]) / (4*w)
#     return np.array([w,x,y,z])

def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to scalar-first quaternion [w, x, y, z]."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]

    trace = m00 + m11 + m22

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S=4*w
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # S=4*x
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # S=4*y
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # S=4*z
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",    default="tdcr_gt_samples.npz")
    parser.add_argument("--out",   default="tdcr_meas_samples.npz")
    parser.add_argument("--tau-noise", type=float, default=0.02, help="σ_τ  [N]")
    parser.add_argument("--ori-noise", type=float, default=np.deg2rad(1.0),
                        help="σ_θ  [rad]")
    parser.add_argument("--seed",  type=int, default=None)
    args = parser.parse_args()

    rng = default_rng(args.seed)

    # ------------------- load ground truth ----------------------------------
    data_gt = np.load(args.gt, allow_pickle=True)
    tau_gt  = data_gt["tau"]                       # (N,4)
    T_gt    = data_gt["T"]                         # (N, n_disks, 4,4)
    meta_gt = json.loads(data_gt["meta"].item())
    N, n_disks = T_gt.shape[:2]

    # ------------------- sensor locations -----------------------------------
    s_locations = np.array([12.0/39, 25.0/39, 1.00])     # normalised arclength
    idx_sensor = np.round(s_locations * (n_disks-1)).astype(int)  # indices

    # ipdb.set_trace()

    # ------------------- allocate measurement arrays ------------------------
    tau_meas = np.empty_like(tau_gt)
    q_meas   = np.empty((N, len(s_locations), 4))   # w,x,y,z

    # ------------------- noise parameters -----------------------------------
    sigma_tau = args.tau_noise
    sigma_theta = args.ori_noise   # [rad]

    # ------------------- iterate over samples -------------------------------
    for k in range(N):
        # ---- cable tensions ------------------------------------------------
        tau_meas[k] = tau_gt[k] + rng.normal(0.0, sigma_tau, size=4)

        # ---- orientations --------------------------------------------------
        for j, idx in enumerate(idx_sensor):
            R_gt = T_gt[k, idx, 0:3, 0:3]
            q_gt = rotmat_to_quat(R_gt)

            # random small-angle rotation
            axis  = rng.normal(size=3); axis /= np.linalg.norm(axis)
            angle = rng.normal(0.0, sigma_theta)
            q_noise = quat_from_axis_angle(axis, angle)

            q_meas[k, j] = quat_mul(q_noise, q_gt)   # measurement = noise ⊗ GT
            q_meas[k, j] /= np.linalg.norm(q_meas[k, j])

    # ------------------- save ------------------------------------------------
    meta_meas = {
        "tau_noise_sigma_N"  : sigma_tau,
        "ori_noise_sigma_deg": np.rad2deg(sigma_theta),
        "s_locations"        : s_locations.tolist(),
        "n_samples"          : int(N),
        "date"               : datetime.datetime.now().isoformat(timespec="seconds"),
        "gt_file"            : args.gt,
        "rng_seed"           : args.seed,
    }
    np.savez_compressed(args.out,
                        tau_meas=tau_meas,
                        q_meas=q_meas,
                        meta=json.dumps(meta_meas))
    print(f"Measurement file saved ➜ {args.out}")
    print(f"  tau_meas shape : {tau_meas.shape}")
    print(f"  q_meas   shape : {q_meas.shape}")

if __name__ == "__main__":
    main()
