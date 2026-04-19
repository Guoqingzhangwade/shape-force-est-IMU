#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensor_placement_study_final.py
================================
Finalized matched-model sensor number / placement study for continuum-robot
shape estimation using the SO(3)-analytic EKF selected in the formulation
comparison study.

Estimator  : SO(3) log-residual EKF, analytic Jacobian
State      : m = [k0_x, k1_x, k0_y, k1_y, k0_z]  (5-D modal coefficient)
R          : alpha * sigma^2 * I,  alpha = 1.0  (fixed from comparison study)
P0         : diag([1.0, 0.5, 1.0, 0.5, 0.25])
Q          : diag([1e-10, 1e-12, 1e-10, 1e-12, 1e-10])

Runtime note: the analytic Jacobian is ~2x faster than the numeric (finite-
difference) alternative for this 5-D state, as confirmed by the formulation
comparison study (~28 ms/step vs ~56 ms/step on a typical CPU).

Fairness guarantee
------------------
All sensor configurations use the SAME fixed shape bank and the SAME noise
seed bank.  Measurement sequences are re-generated from the fixed seed for
each configuration so that every config sees identical underlying truth shapes
and the same noise process starting point.

Default matched-model settings (manuscript defaults)
-----------------------------------------------------
  --num-shapes              10
  --num-noise-realizations  3
  --shape-bank-seed         123
  --noise-seed-base         1000
  --steps                   50
  --meas-std-deg            0.5
  --gamma                   10
  --alpha                   1.0

Usage
-----
  python sensor_placement_study_final.py
  python sensor_placement_study_final.py --plot-best-vs-count
  python sensor_placement_study_final.py --plot-best-vs-count --save-plots
  python sensor_placement_study_final.py --heatmap-grid 8
  python sensor_placement_study_final.py --no-heatmap
  python sensor_placement_study_final.py --plot-best-convergence
  python sensor_placement_study_final.py --plot-all-curves
"""

import argparse
import csv
import json
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as Rot


# ============================================================
#  CONSTANTS
# ============================================================

STATE_DIM   = 5
L_PHYS      = 100.0    # mm (enters translation, not rotation)
PROC_STD0   = 1e-5     # process noise sigma for k0 terms
PROC_STD1   = 1e-6     # process noise sigma for k1 terms
FD_EPS      = 1e-4     # finite-difference step (kept for numerical reference, not used here)

ALPHA_DEFAULT   = 1.0
MEAS_STD_DEG    = 0.5
STEPS_DEFAULT   = 50
GAMMA_DEFAULT   = 10

P0_DIAG = np.array([1.0, 0.5, 1.0, 0.5, 0.25])

# Gauss-Legendre quadrature points for 2-point rule on [0, 1]
_GL_XI = np.array([0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6])

# Plot font sizes (manuscript-ready)
FS_LABEL  = 16
FS_TICK   = 14
FS_LEGEND = 13
FS_TITLE  = 15

# Color scheme: one per sensor count
_COUNT_COLOR = {1: "#4878d0", 2: "#6acc65", 3: "#d65f5f",
                4: "#b47cc7", 5: "#c4ad66"}
_COUNT_LABEL = {1: "1 IMU", 2: "2 IMUs", 3: "3 IMUs",
                4: "4 IMUs", 5: "5 IMUs"}


# ============================================================
#  LIE ALGEBRA HELPERS
# ============================================================

def skew(u: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -u[2],  u[1]],
                     [u[2],  0.0, -u[0]],
                     [-u[1], u[0],  0.0]], dtype=float)


def vee(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def so3_log(Rm: np.ndarray) -> np.ndarray:
    return Rot.from_matrix(Rm).as_rotvec()


def jr_inv_so3(r: np.ndarray) -> np.ndarray:
    """Right Jacobian inverse Jr^{-1}(r), numerically stable via cot(theta/2)."""
    theta = float(np.linalg.norm(r))
    A = skew(r)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * A + (1.0 / 12.0) * (A @ A)
    half = 0.5 * theta
    sin_h = np.sin(half)
    if abs(sin_h) < 1e-12:
        return np.eye(3) + 0.5 * A + (1.0 / 12.0) * (A @ A)
    cot_h = np.cos(half) / sin_h
    a = (1.0 - 0.5 * theta * cot_h) / (theta ** 2)
    return np.eye(3) + 0.5 * A + a * (A @ A)


# ============================================================
#  KINEMATICS (Magnus PoE)
# ============================================================

def phi_mat(s: float) -> np.ndarray:
    """Shape function matrix Phi(s): kappa(s) = Phi(s) @ m."""
    return np.array([[1, s, 0, 0, 0],
                     [0, 0, 1, s, 0],
                     [0, 0, 0, 0, 1]], dtype=float)


def twist_mat(k: np.ndarray, e3: np.ndarray) -> np.ndarray:
    return np.block([[skew(k), e3[:, None]],
                     [np.zeros((1, 3)), 0.0]])


def magnus_psi(s_i: float, h: float, m: np.ndarray, e3: np.ndarray) -> np.ndarray:
    c1, c2 = s_i - h + _GL_XI * h
    k1, k2 = phi_mat(c1) @ m, phi_mat(c2) @ m
    e1, e2 = twist_mat(k1, e3), twist_mat(k2, e3)
    return (h / 2) * (e1 + e2) + (np.sqrt(3) / 12) * h ** 2 * (e1 @ e2 - e2 @ e1)


def fwd_rotation(m: np.ndarray, s: float, e3: np.ndarray, gamma: int) -> np.ndarray:
    """Rotation matrix at arc-length s via piece-wise Magnus PoE."""
    T = np.eye(4)
    h = s / gamma
    for k in range(1, gamma + 1):
        T = T @ expm(magnus_psi(k * h, h, m, e3))
    return T[:3, :3]


def compute_dR_dm(m: np.ndarray, s: float, gamma: int,
                  e3: np.ndarray) -> np.ndarray:
    """
    Analytic derivative dR/dm (3 x 3 x Nm) via Magnus expansion + 1st-order dexp.
    """
    Nm = len(m)
    h  = s / gamma

    Psi_list   = []
    ePsi_list  = []
    T_before   = [np.eye(4)]
    T_after    = [np.eye(4)] * (gamma + 1)
    dPsi_dm    = [np.zeros((4, 4, Nm)) for _ in range(gamma)]
    dePsi_dm   = [np.zeros((4, 4, Nm)) for _ in range(gamma)]

    for k in range(1, gamma + 1):
        s_i   = k * h
        Psi_k = magnus_psi(s_i, h, m, e3)
        Psi_list.append(Psi_k)
        ePsi_k = expm(Psi_k)
        ePsi_list.append(ePsi_k)
        T_before.append(T_before[-1] @ ePsi_k)

    T_after_arr = [np.eye(4) for _ in range(gamma + 1)]
    for k in range(gamma - 1, 0, -1):
        T_after_arr[k] = ePsi_list[k] @ T_after_arr[k + 1]

    for k in range(1, gamma + 1):
        s_i = k * h
        c1  = s_i - h + _GL_XI[0] * h
        c2  = s_i - h + _GL_XI[1] * h
        ph1, ph2   = phi_mat(c1), phi_mat(c2)
        kap1, kap2 = ph1 @ m, ph2 @ m
        eta1, eta2 = twist_mat(kap1, e3), twist_mat(kap2, e3)
        for i in range(Nm):
            dk1 = ph1[:, i]
            dk2 = ph2[:, i]
            de1 = twist_mat(dk1, np.zeros(3))
            de2 = twist_mat(dk2, np.zeros(3))
            comm = (de1 @ eta2 - eta2 @ de1) + (eta1 @ de2 - de2 @ eta1)
            dPsi_dm[k - 1][:, :, i] = (h / 2) * (de1 + de2) + (np.sqrt(3) / 12) * h ** 2 * comm

    for k in range(gamma):
        Psi_k  = Psi_list[k]
        ePsi_k = ePsi_list[k]
        for i in range(Nm):
            dP = dPsi_dm[k][:, :, i]
            dexpinv = dP + 0.5 * (Psi_k @ dP - dP @ Psi_k)
            dePsi_dm[k][:, :, i] = ePsi_k @ dexpinv

    dT_dm = np.zeros((4, 4, Nm))
    for i in range(Nm):
        for k in range(gamma):
            dT_dm[:, :, i] += T_before[k] @ dePsi_dm[k][:, :, i] @ T_after_arr[k + 1]

    return dT_dm[:3, :3, :]


# ============================================================
#  SO(3)-ANALYTIC MEASUREMENT FUNCTION
# ============================================================

def so3_analytic_H_and_r(m: np.ndarray,
                          imu_pos: np.ndarray,
                          meas_frame: List[np.ndarray],
                          e3: np.ndarray,
                          gamma: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stacked SO(3) residuals and analytic Jacobian for all sensors.

    Residual  : r_i = log(R_meas_i @ R_pred_i^T)
    Jacobian  : H_i = -Jr^{-1}(r_i) @ vee( skew_part(dR_i/dm @ R_pred_i^T) )

    Returns
    -------
    H : (3*n_sensors, 5) stacked Jacobian
    r : (3*n_sensors,)  stacked residual
    """
    H_list, r_list = [], []
    for s_i, R_meas in zip(imu_pos, meas_frame):
        R_pred = fwd_rotation(m, s_i, e3, gamma)
        R_e    = R_meas @ R_pred.T
        r      = so3_log(R_e)
        Jr_inv = jr_inv_so3(r)
        dR_dm  = compute_dR_dm(m, s_i, gamma, e3)

        H_i = np.zeros((3, STATE_DIM))
        for p in range(STATE_DIM):
            A = dR_dm[:, :, p] @ R_pred.T
            H_i[:, p] = -Jr_inv @ vee(0.5 * (A - A.T))

        H_list.append(H_i)
        r_list.append(r)

    return np.vstack(H_list), np.hstack(r_list)


# ============================================================
#  EKF  (SO(3)-analytic, matched to finalized comparison study)
# ============================================================

def run_ekf_so3_analytic(meas_R_seq: List[List[np.ndarray]],
                         m_true: np.ndarray,
                         imu_pos: np.ndarray,
                         e3: np.ndarray,
                         gamma: int,
                         meas_std_deg: float,
                         alpha: float,
                         P0: np.ndarray,
                         nis_burnin: int = 10) -> Dict:
    """
    One EKF trial: SO(3)-analytic measurement model.

    Returns dict with rmse_final, rmse_mean, rmse_t, nis_mean, nis_mean_postburnin,
    time_per_step.
    """
    sigma = np.deg2rad(meas_std_deg)
    R_single = alpha * (sigma ** 2) * np.eye(3)
    Q = np.diag([PROC_STD0**2, PROC_STD1**2,
                 PROC_STD0**2, PROC_STD1**2,
                 PROC_STD0**2])

    m_est = np.zeros(STATE_DIM)
    P_est = P0.copy()
    hist  = []
    nis_hist = []

    t0 = time.perf_counter()
    n_sensors = len(imu_pos)

    for frame in meas_R_seq:
        m_pred = m_est.copy()
        P_pred = P_est + Q

        H, r = so3_analytic_H_and_r(m_pred, imu_pos, frame, e3, gamma)
        innov  = -r  # innovation = 0 - h(m)
        R_big  = np.kron(np.eye(n_sensors), R_single)

        S     = H @ P_pred @ H.T + R_big
        S_inv = inv(S)
        K     = P_pred @ H.T @ S_inv

        m_est = m_pred + K @ innov

        IKH   = np.eye(STATE_DIM) - K @ H
        P_est = IKH @ P_pred @ IKH.T + K @ R_big @ K.T
        P_est = 0.5 * (P_est + P_est.T)

        hist.append(m_est.copy())
        nis_hist.append(float(innov @ S_inv @ innov))

    t1 = time.perf_counter()

    hist    = np.array(hist)           # (steps, 5)
    err     = hist - m_true
    rmse_t  = np.sqrt(np.mean(err ** 2, axis=1))

    nis_arr    = np.array(nis_hist)
    bi         = min(nis_burnin, len(nis_arr) - 1)
    nis_post   = nis_arr[bi:]

    return {
        "rmse_final":           float(np.sqrt(np.mean((m_est - m_true) ** 2))),
        "rmse_mean":            float(np.mean(rmse_t)),
        "rmse_t":               rmse_t,
        "nis_mean":             float(np.mean(nis_arr)),
        "nis_mean_postburnin":  float(np.mean(nis_post)),
        "time_per_step":        float((t1 - t0) / len(meas_R_seq)),
    }


# ============================================================
#  DATA GENERATION
# ============================================================

def generate_meas_R(m_true: np.ndarray,
                    imu_pos: np.ndarray,
                    e3: np.ndarray,
                    gamma: int,
                    steps: int,
                    meas_std_deg: float,
                    rng: np.random.RandomState) -> List[List[np.ndarray]]:
    """
    Generate noisy rotation-matrix measurement sequence.
    seq[step][sensor] = R_meas  (3x3).
    """
    sigma = np.deg2rad(meas_std_deg)
    seq = []
    for _ in range(steps):
        frame = []
        for s in imu_pos:
            R_clean = fwd_rotation(m_true, s, e3, gamma)
            R_noise = Rot.from_rotvec(rng.randn(3) * sigma).as_matrix()
            frame.append(R_noise @ R_clean)
        seq.append(frame)
    return seq


def build_shape_bank(num_shapes: int,
                     k0_range: Tuple[float, float],
                     k1_range: Tuple[float, float],
                     kz_range: Tuple[float, float],
                     seed: int) -> List[np.ndarray]:
    """Generate a fixed list of 5D modal coefficient vectors."""
    rng = np.random.RandomState(seed)
    bank = []
    for _ in range(num_shapes):
        k0x = rng.uniform(*k0_range)
        k1x = rng.uniform(*k1_range)
        k0y = rng.uniform(*k0_range)
        k1y = rng.uniform(*k1_range)
        k0z = rng.uniform(*kz_range)
        bank.append(np.array([k0x, k1x, k0y, k1y, k0z]))
    return bank


def build_noise_seed_bank(num_shapes: int,
                          num_noise_realizations: int,
                          seed_base: int) -> np.ndarray:
    """Return (num_shapes x num_noise_realizations) integer seed table."""
    seeds = np.arange(num_shapes * num_noise_realizations, dtype=int)
    seeds = seeds.reshape(num_shapes, num_noise_realizations) + seed_base
    return seeds


# ============================================================
#  SENSOR CONFIGURATIONS
# ============================================================

def build_sensor_configs() -> Dict[str, np.ndarray]:
    """
    Returns an ordered dict of named sensor configurations.
    Keys are manuscript-friendly: e.g. '1-IMU [0.25]', '2-IMU [0.25,0.50]'.
    """
    cfgs = {}

    # 1 sensor
    for s in [0.25, 0.50, 0.75, 1.00]:
        cfgs[f"1-IMU [{s:.2f}]"] = np.array([s])

    # 2 sensors
    pairs = [(0.25, 0.50), (0.25, 0.75), (0.25, 1.00),
             (0.50, 0.75), (0.50, 1.00), (0.75, 1.00)]
    for a, b in pairs:
        cfgs[f"2-IMU [{a:.2f},{b:.2f}]"] = np.array([a, b])

    # 3 sensors
    triples = [
        (0.25, 0.50, 0.75),
        (0.25, 0.50, 1.00),
        (0.25, 0.75, 1.00),
        (0.33, 0.67, 1.00),
    ]
    for t in triples:
        key = "3-IMU [" + ",".join(f"{v:.2f}" for v in t) + "]"
        cfgs[key] = np.array(t)

    # 4 sensors
    quads = [
        (0.25, 0.50, 0.75, 1.00),
        (0.20, 0.40, 0.60, 0.80),
    ]
    for q in quads:
        key = "4-IMU [" + ",".join(f"{v:.2f}" for v in q) + "]"
        cfgs[key] = np.array(q)

    # 5 sensors
    cfgs["5-IMU [0.20,0.40,0.60,0.80,1.00]"] = np.array([0.20, 0.40, 0.60, 0.80, 1.00])

    return cfgs


# ============================================================
#  STUDY RUNNER
# ============================================================

def run_sensor_config(imu_pos: np.ndarray,
                      shape_bank: List[np.ndarray],
                      noise_seeds: np.ndarray,
                      e3: np.ndarray,
                      gamma: int,
                      steps: int,
                      meas_std_deg: float,
                      alpha: float,
                      P0: np.ndarray,
                      nis_burnin: int) -> Dict:
    """
    Run EKF for one sensor configuration across the full matched-model bank.

    Measurements are re-generated for each (shape, noise_seed) pair using
    the stored seed.  This guarantees identical truth shapes and the same noise
    starting point across all configurations.

    Returns aggregated stats dict.
    """
    num_shapes            = len(shape_bank)
    num_noise_realizations = noise_seeds.shape[1]

    rmse_finals  = []
    rmse_means   = []
    nis_means    = []
    nis_post_means = []
    time_steps   = []
    rmse_trajs   = []

    for si, m_true in enumerate(shape_bank):
        for ni in range(num_noise_realizations):
            seed = int(noise_seeds[si, ni])
            rng  = np.random.RandomState(seed)
            meas_R = generate_meas_R(m_true, imu_pos, e3, gamma, steps,
                                     meas_std_deg, rng)
            res = run_ekf_so3_analytic(meas_R, m_true, imu_pos, e3, gamma,
                                       meas_std_deg, alpha, P0, nis_burnin)
            rmse_finals.append(res["rmse_final"])
            rmse_means.append(res["rmse_mean"])
            nis_means.append(res["nis_mean"])
            nis_post_means.append(res["nis_mean_postburnin"])
            time_steps.append(res["time_per_step"])
            rmse_trajs.append(res["rmse_t"])

    trajs = np.vstack(rmse_trajs)   # (total_runs, steps)
    nis_dof = 3 * len(imu_pos)

    return {
        "imu_pos":              imu_pos,
        "n_sensors":            len(imu_pos),
        "nis_dof":              nis_dof,
        "total_runs":           num_shapes * num_noise_realizations,
        "rmse_final_mean":      float(np.mean(rmse_finals)),
        "rmse_final_std":       float(np.std(rmse_finals)),
        "rmse_mean_mean":       float(np.mean(rmse_means)),
        "rmse_mean_std":        float(np.std(rmse_means)),
        "nis_mean_full":        float(np.mean(nis_means)),
        "nis_mean_postburnin":  float(np.mean(nis_post_means)),
        "time_per_step_mean":   float(np.mean(time_steps)),
        "rmse_traj_mean":       trajs.mean(axis=0),
        "rmse_traj_std":        trajs.std(axis=0),
    }


def run_sensor_study(configs: Dict[str, np.ndarray],
                     shape_bank: List[np.ndarray],
                     noise_seeds: np.ndarray,
                     e3: np.ndarray,
                     gamma: int,
                     steps: int,
                     meas_std_deg: float,
                     alpha: float,
                     P0: np.ndarray,
                     nis_burnin: int) -> Dict[str, Dict]:
    """
    Run run_sensor_config for every entry in configs.  Prints progress.
    Returns stats_dict: config_name -> stats.
    """
    stats_dict = {}
    total = len(configs)
    for idx, (name, imu_pos) in enumerate(configs.items(), 1):
        n = len(imu_pos)
        print(f"  [{idx:2d}/{total}]  {name:42s}  n={n} ...", end=" ", flush=True)
        stat = run_sensor_config(imu_pos, shape_bank, noise_seeds,
                                 e3, gamma, steps, meas_std_deg,
                                 alpha, P0, nis_burnin)
        stats_dict[name] = stat
        print(f"RMSE_final = {stat['rmse_final_mean']:.4e} +/- {stat['rmse_final_std']:.2e}"
              f"  NIS_post = {stat['nis_mean_postburnin']:.2f} (DOF={stat['nis_dof']})")
    return stats_dict


def summarize_best_by_count(stats_dict: Dict[str, Dict]) -> Dict[int, Tuple[str, Dict]]:
    """
    For each sensor count, find the config with the lowest rmse_final_mean.
    Returns {count: (name, stat)}.
    """
    by_count: Dict[int, List[Tuple[str, float, Dict]]] = {}
    for name, stat in stats_dict.items():
        n = stat["n_sensors"]
        by_count.setdefault(n, []).append((name, stat["rmse_final_mean"], stat))

    best = {}
    for n, entries in sorted(by_count.items()):
        entries.sort(key=lambda x: x[1])
        best_name, _, best_stat = entries[0]
        best[n] = (best_name, best_stat)
    return best


# ============================================================
#  CONSOLE REPORTING
# ============================================================

def print_summary_table(stats_dict: Dict[str, Dict]) -> None:
    w = 108
    print("\n" + "=" * w)
    print("SENSOR PLACEMENT STUDY  --  SO(3)-Analytic EKF, Matched-Model Bank")
    print("=" * w)
    hdr = (f"{'Config':44s}  {'n':>2}  {'RMSE_final_mean':>15}  "
           f"{'std':>10}  {'RMSE_mean_mean':>14}  {'NIS_post':>9}  {'ms/step':>7}")
    print(hdr)
    print("-" * w)
    prev_n = -1
    for name, stat in stats_dict.items():
        n = stat["n_sensors"]
        if n != prev_n and prev_n != -1:
            print()
        prev_n = n
        print(f"{name:44s}  {n:>2}  "
              f"{stat['rmse_final_mean']:>15.4e}  "
              f"{stat['rmse_final_std']:>10.2e}  "
              f"{stat['rmse_mean_mean']:>14.4e}  "
              f"{stat['nis_mean_postburnin']:>9.3f}  "
              f"{stat['time_per_step_mean']*1e3:>7.2f}")
    print("=" * w)


def print_best_by_count(best: Dict[int, Tuple[str, Dict]]) -> None:
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION PER SENSOR COUNT")
    print("=" * 80)
    for n, (name, stat) in sorted(best.items()):
        print(f"  {n} IMU(s)  ->  {name}")
        print(f"            RMSE_final = {stat['rmse_final_mean']:.4e}"
              f" +/- {stat['rmse_final_std']:.2e}"
              f"   NIS_post = {stat['nis_mean_postburnin']:.2f}"
              f" (DOF={stat['nis_dof']})")
    print("=" * 80)


# ============================================================
#  CSV / JSON OUTPUT
# ============================================================

def save_csv(stats_dict: Dict[str, Dict],
             best: Dict[int, Tuple[str, Dict]],
             alpha: float,
             nis_burnin: int) -> None:
    # Full results
    fieldnames = [
        "config", "n_sensors", "imu_positions",
        "rmse_final_mean", "rmse_final_std",
        "rmse_mean_mean", "rmse_mean_std",
        "nis_mean_full", "nis_mean_postburnin",
        "nis_dof", "total_runs", "time_per_step_mean_ms",
    ]
    rows = []
    for name, stat in stats_dict.items():
        rows.append({
            "config":                name,
            "n_sensors":             stat["n_sensors"],
            "imu_positions":         str(list(stat["imu_pos"])),
            "rmse_final_mean":       stat["rmse_final_mean"],
            "rmse_final_std":        stat["rmse_final_std"],
            "rmse_mean_mean":        stat["rmse_mean_mean"],
            "rmse_mean_std":         stat["rmse_mean_std"],
            "nis_mean_full":         stat["nis_mean_full"],
            "nis_mean_postburnin":   stat["nis_mean_postburnin"],
            "nis_dof":               stat["nis_dof"],
            "total_runs":            stat["total_runs"],
            "time_per_step_mean_ms": stat["time_per_step_mean"] * 1e3,
        })
    with open("sensor_placement_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("  Saved -> sensor_placement_results.csv")

    # Best-per-count
    best_fieldnames = [
        "n_sensors", "best_config", "imu_positions",
        "rmse_final_mean", "rmse_final_std", "nis_mean_postburnin", "nis_dof",
    ]
    best_rows = []
    for n, (name, stat) in sorted(best.items()):
        best_rows.append({
            "n_sensors":           n,
            "best_config":         name,
            "imu_positions":       str(list(stat["imu_pos"])),
            "rmse_final_mean":     stat["rmse_final_mean"],
            "rmse_final_std":      stat["rmse_final_std"],
            "nis_mean_postburnin": stat["nis_mean_postburnin"],
            "nis_dof":             stat["nis_dof"],
        })
    with open("sensor_best_by_count.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=best_fieldnames)
        writer.writeheader()
        writer.writerows(best_rows)
    print("  Saved -> sensor_best_by_count.csv")


def save_json(stats_dict: Dict[str, Dict],
              best: Dict[int, Tuple[str, Dict]],
              cfg_meta: Dict) -> None:
    def _stat_to_serializable(s: Dict) -> Dict:
        out = {k: v for k, v in s.items()
               if k not in ("rmse_traj_mean", "rmse_traj_std", "imu_pos")}
        out["imu_pos"] = list(s["imu_pos"])
        return out

    summary = {
        "config": cfg_meta,
        "all_configs": {name: _stat_to_serializable(stat)
                        for name, stat in stats_dict.items()},
        "best_by_count": {str(n): {"config": name,
                                   "rmse_final_mean": stat["rmse_final_mean"],
                                   "rmse_final_std":  stat["rmse_final_std"],
                                   "nis_mean_postburnin": stat["nis_mean_postburnin"]}
                          for n, (name, stat) in sorted(best.items())},
    }
    with open("sensor_placement_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  Saved -> sensor_placement_results.json")


# ============================================================
#  PLOTTING
# ============================================================

def plot_best_vs_count(best: Dict[int, Tuple[str, Dict]],
                       save: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot skipped] pip install matplotlib")
        return

    counts = sorted(best.keys())
    means  = [best[n][1]["rmse_final_mean"] for n in counts]
    stds   = [best[n][1]["rmse_final_std"]  for n in counts]
    colors = [_COUNT_COLOR.get(n, "gray") for n in counts]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(counts, means, yerr=stds,
                fmt="o-", linewidth=2.2, markersize=8,
                color="#2166ac", capsize=5, capthick=1.5,
                ecolor="#555555", zorder=3)
    for n, m, c in zip(counts, means, colors):
        ax.scatter([n], [m], color=c, s=80, zorder=4)

    ax.set_xticks(counts)
    ax.set_xticklabels([f"{n}" for n in counts], fontsize=FS_TICK)
    ax.set_xlabel("Number of IMUs", fontsize=FS_LABEL)
    ax.set_ylabel("Best final RMSE (modal coefficients)", fontsize=FS_LABEL)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.grid(True, alpha=0.35, which="both")
    fig.tight_layout()

    if save:
        fig.savefig("sensor_best_vs_count.pdf", bbox_inches="tight")
        print("  Saved -> sensor_best_vs_count.pdf")
    plt.show()


def plot_2sensor_heatmap_final(shape_bank: List[np.ndarray],
                               noise_seeds: np.ndarray,
                               e3: np.ndarray,
                               gamma: int,
                               steps: int,
                               meas_std_deg: float,
                               alpha: float,
                               P0: np.ndarray,
                               nis_burnin: int,
                               grid_n: int = 8,
                               save: bool = False) -> None:
    """
    Sweep all ordered pairs from a uniform grid and plot mean final RMSE
    as a heatmap.  Uses the same matched-model bank for fairness.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot skipped] pip install matplotlib")
        return

    grid   = np.linspace(0.10, 1.00, grid_n)
    hmap   = np.full((grid_n, grid_n), np.nan)
    n_pair = grid_n * (grid_n - 1) // 2

    print(f"\n  [2-IMU heatmap]  grid={grid_n}x{grid_n}  ->  {n_pair} unique pairs")

    done = 0
    for i, s1 in enumerate(grid):
        for j, s2 in enumerate(grid):
            if j <= i:
                continue
            stat = run_sensor_config(
                np.array([s1, s2]), shape_bank, noise_seeds,
                e3, gamma, steps, meas_std_deg, alpha, P0, nis_burnin
            )
            val          = stat["rmse_final_mean"]
            hmap[i, j]   = val
            hmap[j, i]   = val
            done += 1
            if done % max(1, n_pair // 10) == 0 or done == n_pair:
                print(f"    {done}/{n_pair} pairs done ...", flush=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ext = [grid[0], grid[-1], grid[0], grid[-1]]
    im  = ax.imshow(hmap, origin="lower", aspect="auto", extent=ext,
                    cmap="RdYlGn_r", interpolation="nearest")
    cb  = plt.colorbar(im, ax=ax)
    cb.set_label("Mean final RMSE", fontsize=FS_LEGEND)
    cb.ax.tick_params(labelsize=FS_TICK - 2)

    ax.set_xlabel("Sensor 2 position $s_2$", fontsize=FS_LABEL)
    ax.set_ylabel("Sensor 1 position $s_1$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_title("2-IMU placement sweep  (lower = better)",
                 fontsize=FS_TITLE)
    fig.tight_layout()

    if save:
        fig.savefig("sensor_2imu_heatmap.pdf", bbox_inches="tight")
        print("  Saved -> sensor_2imu_heatmap.pdf")
    plt.show()


def plot_best_convergence(best: Dict[int, Tuple[str, Dict]],
                          steps: int,
                          save: bool = False) -> None:
    """RMSE convergence curves for the best config per sensor count."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot skipped] pip install matplotlib")
        return

    t_axis = np.arange(1, steps + 1)
    fig, ax = plt.subplots(figsize=(9, 5))

    for n, (name, stat) in sorted(best.items()):
        color = _COUNT_COLOR.get(n, "gray")
        mean  = stat["rmse_traj_mean"]
        std   = stat["rmse_traj_std"]
        ax.plot(t_axis, mean, color=color, lw=2.2,
                label=f"{_COUNT_LABEL[n]}: {name}")
        ax.fill_between(t_axis, np.maximum(mean - std, 1e-10), mean + std,
                        alpha=0.15, color=color)

    ax.set_yscale("log")
    ax.set_xlabel("EKF step", fontsize=FS_LABEL)
    ax.set_ylabel("RMSE (modal coefficients)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND - 1, framealpha=0.9, edgecolor="0.7")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    if save:
        fig.savefig("sensor_best_convergence.pdf", bbox_inches="tight")
        print("  Saved -> sensor_best_convergence.pdf")
    plt.show()


def plot_all_curves(stats_dict: Dict[str, Dict],
                    steps: int,
                    save: bool = False) -> None:
    """All-config convergence panel, one subplot per sensor count."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot skipped] pip install matplotlib")
        return

    by_count: Dict[int, List[str]] = {}
    for name, stat in stats_dict.items():
        by_count.setdefault(stat["n_sensors"], []).append(name)

    counts = sorted(by_count.keys())
    ncols  = min(3, len(counts))
    nrows  = (len(counts) + ncols - 1) // ncols
    t_axis = np.arange(1, steps + 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False, sharey=True)

    for idx, n in enumerate(counts):
        ax    = axes[idx // ncols][idx % ncols]
        names = by_count[n]
        cmap  = plt.cm.get_cmap("tab10", len(names))
        for j, name in enumerate(names):
            stat  = stats_dict[name]
            mean  = stat["rmse_traj_mean"]
            std   = stat["rmse_traj_std"]
            color = cmap(j)
            lbl   = name.split("[", 1)[-1].rstrip("]")
            ax.plot(t_axis, mean, color=color, lw=1.6, label=lbl)
            ax.fill_between(t_axis,
                            np.maximum(mean - std, 1e-10), mean + std,
                            alpha=0.12, color=color)
        ax.set_yscale("log")
        ax.set_title(f"{n} IMU(s)", fontsize=FS_TITLE - 1)
        ax.set_xlabel("EKF step", fontsize=FS_LABEL - 2)
        ax.set_ylabel("RMSE", fontsize=FS_LABEL - 2)
        ax.tick_params(labelsize=FS_TICK - 2)
        ax.legend(fontsize=FS_LEGEND - 3, loc="upper right")
        ax.grid(True, alpha=0.3, which="both")

    for idx in range(len(counts), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("RMSE convergence by sensor configuration  (mean +/- 1 std)",
                 fontsize=FS_TITLE, fontweight="bold")
    fig.tight_layout()

    if save:
        fig.savefig("sensor_all_convergence.pdf", bbox_inches="tight")
        print("  Saved -> sensor_all_convergence.pdf")
    plt.show()


# ============================================================
#  MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalized matched-model sensor placement study "
                    "(SO(3)-analytic EKF).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Matched-model bank
    parser.add_argument("--num-shapes", type=int, default=10,
                        help="Number of distinct true shapes in the bank (default: 10)")
    parser.add_argument("--num-noise-realizations", type=int, default=3,
                        help="Noise realizations per shape (default: 3)")
    parser.add_argument("--shape-bank-seed", type=int, default=123,
                        help="Seed for shape bank generation (default: 123)")
    parser.add_argument("--noise-seed-base", type=int, default=1000,
                        help="Base offset for noise seeds (default: 1000)")

    # EKF / physics
    parser.add_argument("--steps", type=int, default=50,
                        help="EKF steps per run (default: 50)")
    parser.add_argument("--gamma", type=int, default=10,
                        help="Magnus PoE segments (default: 10)")
    parser.add_argument("--meas-std-deg", type=float, default=0.5,
                        help="Measurement noise std dev [deg] (default: 0.5)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="R-scale alpha applied to all configs (default: 1.0)")
    parser.add_argument("--nis-burnin", type=int, default=10,
                        help="Steps to skip for post-burn-in NIS (default: 10)")

    # Heatmap
    parser.add_argument("--heatmap-grid", type=int, default=8,
                        help="Grid size for 2-IMU heatmap sweep (default: 8)")
    parser.add_argument("--no-heatmap", action="store_true",
                        help="Skip 2-IMU placement heatmap")

    # Optional plots
    parser.add_argument("--plot-best-vs-count", action="store_true",
                        help="Plot best RMSE vs sensor count")
    parser.add_argument("--plot-best-convergence", action="store_true",
                        help="Plot RMSE convergence for best config per count")
    parser.add_argument("--plot-all-curves", action="store_true",
                        help="Plot full convergence panel (all configs)")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save all plots as PDF files")

    args = parser.parse_args()

    # Physical setup
    e3    = np.array([0.0, 0.0, L_PHYS])
    P0    = np.diag(P0_DIAG)
    total_runs = args.num_shapes * args.num_noise_realizations

    print("\n" + "=" * 72)
    print("SENSOR PLACEMENT STUDY  --  Finalized SO(3)-Analytic EKF")
    print("=" * 72)
    print(f"  Estimator   : SO(3) log-residual EKF, analytic Jacobian")
    print(f"  Alpha        : {args.alpha}")
    print(f"  P0 (diag)    : {[float(x) for x in P0_DIAG]}")
    print(f"  Q (diag)     : [{PROC_STD0**2:.0e}, {PROC_STD1**2:.0e}, "
          f"{PROC_STD0**2:.0e}, {PROC_STD1**2:.0e}, {PROC_STD0**2:.0e}]")
    print(f"  Shapes       : {args.num_shapes}  (seed={args.shape_bank_seed})")
    print(f"  Noise real.  : {args.num_noise_realizations}  (base={args.noise_seed_base})")
    print(f"  Total runs   : {total_runs} per configuration")
    print(f"  Steps / run  : {args.steps}")
    print(f"  Meas noise   : {args.meas_std_deg} deg")
    print(f"  NIS burn-in  : {args.nis_burnin} steps")
    print("=" * 72)

    # Build matched-model bank (shared across ALL configs)
    shape_bank  = build_shape_bank(
        args.num_shapes,
        (-2.0, 2.0), (-1.5, 1.5), (-1.0, 1.0),
        args.shape_bank_seed
    )
    noise_seeds = build_noise_seed_bank(
        args.num_shapes, args.num_noise_realizations, args.noise_seed_base
    )

    print(f"\n  Shape bank [{args.num_shapes} shapes]:")
    for i, m in enumerate(shape_bank):
        print(f"    {i+1:2d}: [{m[0]:+6.3f}, {m[1]:+6.3f}, "
              f"{m[2]:+6.3f}, {m[3]:+6.3f}, {m[4]:+6.3f}]")

    # Build sensor configurations
    configs = build_sensor_configs()
    print(f"\n  Running {len(configs)} sensor configurations ...\n")

    # Run study
    stats_dict = run_sensor_study(
        configs, shape_bank, noise_seeds, e3,
        args.gamma, args.steps, args.meas_std_deg,
        args.alpha, P0, args.nis_burnin
    )

    # Console summary
    print_summary_table(stats_dict)

    best = summarize_best_by_count(stats_dict)
    print_best_by_count(best)

    # Save outputs
    print()
    cfg_meta = {
        "estimator":            "SO(3)-analytic EKF",
        "alpha":                args.alpha,
        "P0_diag":              list(P0_DIAG),
        "num_shapes":           args.num_shapes,
        "num_noise_realizations": args.num_noise_realizations,
        "shape_bank_seed":      args.shape_bank_seed,
        "noise_seed_base":      args.noise_seed_base,
        "steps":                args.steps,
        "meas_std_deg":         args.meas_std_deg,
        "gamma":                args.gamma,
        "nis_burnin":           args.nis_burnin,
    }
    save_csv(stats_dict, best, args.alpha, args.nis_burnin)
    save_json(stats_dict, best, cfg_meta)

    # Optional plots (all configs)
    if args.plot_best_vs_count:
        plot_best_vs_count(best, save=args.save_plots)

    if args.plot_best_convergence:
        plot_best_convergence(best, args.steps, save=args.save_plots)

    if args.plot_all_curves:
        plot_all_curves(stats_dict, args.steps, save=args.save_plots)

    # 2-IMU heatmap
    if not args.no_heatmap:
        plot_2sensor_heatmap_final(
            shape_bank, noise_seeds, e3,
            args.gamma, args.steps, args.meas_std_deg,
            args.alpha, P0, args.nis_burnin,
            grid_n=args.heatmap_grid,
            save=args.save_plots,
        )


if __name__ == "__main__":
    main()
