#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensor_placement_study.py

Compare EKF shape estimation performance across different IMU sensor
configurations (number and arc-length placement along s ∈ (0, 1]).

Residual:  SO(3) log map  r = log(R_meas @ R_pred^T)
Jacobian:  central finite differences
State:     m = [mx0, mx1, my0, my1, mz0]  (5-D curvature model)

Usage examples:
  python sensor_placement_study.py
  python sensor_placement_study.py --trials 20 --steps 100
  python sensor_placement_study.py --fast            # quicker full sweep
  python sensor_placement_study.py --smoke           # very fast sanity check
  python sensor_placement_study.py --no-heatmap      # skip 2-sensor sweep
"""

import argparse
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as Rot
from scipy.linalg import expm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ========================== CONSTANTS ==========================

STATE_DIM  = 5
L_PHYS     = 100.0          # mm (enters translation, not rotation)
GAMMA      = 10             # Magnus PoE subdivisions
FD_EPS     = 1e-4           # finite-difference step
PROC_STD0  = 1e-5           # process noise σ for k0 terms
PROC_STD1  = 1e-6           # process noise σ for k1 terms
INIT_COV   = 1e-2           # initial covariance scale

e3 = np.array([0.0, 0.0, L_PHYS])


# ========================== SENSOR CONFIGURATIONS ==========================

def build_configs() -> Dict[str, np.ndarray]:
    """
    Returns an ordered dict of named sensor configurations.
    Keys encode the number of sensors and their positions.
    """
    cfgs = {}

    # ---------- 1 sensor ----------
    for s in [0.25, 0.50, 0.75, 1.00]:
        cfgs[f"1·s={s:.2f}"] = np.array([s])

    # ---------- 2 sensors ----------
    pairs = [(0.25, 0.50), (0.25, 0.75), (0.25, 1.00),
             (0.50, 0.75), (0.50, 1.00), (0.75, 1.00)]
    for a, b in pairs:
        cfgs[f"2·[{a},{b}]"] = np.array([a, b])

    # ---------- 3 sensors ----------
    cfgs["3·[.25,.50,.75]"] = np.array([0.25, 0.50, 0.75])
    cfgs["3·[.25,.50,1.0]"] = np.array([0.25, 0.50, 1.00])
    cfgs["3·[.25,.75,1.0]"] = np.array([0.25, 0.75, 1.00])
    cfgs["3·[.33,.67,1.0]"] = np.array([0.33, 0.67, 1.00])

    # ---------- 4 sensors ----------
    cfgs["4·[.25,.50,.75,1.0]"]    = np.array([0.25, 0.50, 0.75, 1.00])
    cfgs["4·[.20,.40,.60,.80]"]    = np.array([0.20, 0.40, 0.60, 0.80])

    # ---------- 5 sensors ----------
    cfgs["5·[.20,.40,.60,.80,1.0]"] = np.array([0.20, 0.40, 0.60, 0.80, 1.00])

    return cfgs


# ========================== KINEMATICS ==========================

def skew(u: np.ndarray) -> np.ndarray:
    return np.array([[0,    -u[2],  u[1]],
                     [u[2],  0,    -u[0]],
                     [-u[1], u[0],  0   ]], dtype=float)


def phi(s: float) -> np.ndarray:
    """Shape basis Φ(s) ∈ ℝ^{3×5}: κ(s) = Φ(s) @ m."""
    return np.array([[1, s, 0, 0, 0],
                     [0, 0, 1, s, 0],
                     [0, 0, 0, 0, 1]], dtype=float)


def twist(k: np.ndarray, e3_vec: np.ndarray) -> np.ndarray:
    return np.block([[skew(k), e3_vec[:, None]],
                     [np.zeros((1, 3)), 0.0]])


def magnus_psi(s_i: float, h: float, m: np.ndarray,
               e3_vec: np.ndarray) -> np.ndarray:
    xi = np.array([0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6])
    c1, c2 = s_i - h + xi * h
    k1, k2 = phi(c1) @ m, phi(c2) @ m
    e1, e2 = twist(k1, e3_vec), twist(k2, e3_vec)
    return (h / 2) * (e1 + e2) + (np.sqrt(3) / 12) * h ** 2 * (e1 @ e2 - e2 @ e1)


def fwd_rotation(m: np.ndarray, s: float,
                 e3_vec: np.ndarray, gamma: int) -> np.ndarray:
    """Rotation matrix at arc-length s via piece-wise Magnus PoE."""
    T = np.eye(4)
    h = s / gamma
    for k in range(1, gamma + 1):
        T = T @ expm(magnus_psi(k * h, h, m, e3_vec))
    return T[:3, :3]


# ========================== SO(3) RESIDUAL AND JACOBIAN ==========================

def residual_so3(R_meas: np.ndarray, m: np.ndarray, s: float,
                 e3_vec: np.ndarray, gamma: int) -> np.ndarray:
    """r = log(R_meas @ R_pred^T)  –  zero at true state."""
    R_pred = fwd_rotation(m, s, e3_vec, gamma)
    return Rot.from_matrix(R_meas @ R_pred.T).as_rotvec()


def jacobian_numeric(R_meas: np.ndarray, m: np.ndarray, s: float,
                     e3_vec: np.ndarray, gamma: int) -> np.ndarray:
    """Central-difference Jacobian ∂r/∂m  (3 × STATE_DIM)."""
    J = np.zeros((3, STATE_DIM))
    for i in range(STATE_DIM):
        mp, mm = m.copy(), m.copy()
        mp[i] += FD_EPS
        mm[i] -= FD_EPS
        J[:, i] = (residual_so3(R_meas, mp, s, e3_vec, gamma) -
                   residual_so3(R_meas, mm, s, e3_vec, gamma)) / (2 * FD_EPS)
    return J


# ========================== SYNTHETIC DATA ==========================

def generate_meas_seq(m_true: np.ndarray, imu_pos: np.ndarray,
                      e3_vec: np.ndarray, gamma: int,
                      steps: int, meas_std_deg: float,
                      rng: np.random.RandomState) -> List[List[np.ndarray]]:
    """Noisy rotation matrix measurements: seq[step][sensor] = R_meas."""
    sigma = np.deg2rad(meas_std_deg)
    seq = []
    for _ in range(steps):
        frame = []
        for s in imu_pos:
            R_clean = fwd_rotation(m_true, s, e3_vec, gamma)
            R_noise = Rot.from_rotvec(rng.randn(3) * sigma).as_matrix()
            frame.append(R_noise @ R_clean)
        seq.append(frame)
    return seq


# ========================== EKF ==========================

def run_ekf(m_true: np.ndarray,
            meas_seq: List[List[np.ndarray]],
            imu_pos: np.ndarray,
            e3_vec: np.ndarray,
            gamma: int,
            meas_std_deg: float) -> Dict:
    """
    Run one EKF trial.

    Sign convention (same as shape_ekf_5d.py):
        innovation = -residual  (residual is treated as h(m), measurement = 0)
        m_est = m_pred + K @ innovation = m_pred - K @ residual
    """
    sigma = np.deg2rad(meas_std_deg)
    R_single = (sigma ** 2) * np.eye(3)
    Q = np.diag([PROC_STD0 ** 2, PROC_STD1 ** 2,
                 PROC_STD0 ** 2, PROC_STD1 ** 2,
                 PROC_STD0 ** 2])

    m_est = np.zeros(STATE_DIM)
    P_est = INIT_COV * np.eye(STATE_DIM)
    hist  = []

    for frame in meas_seq:
        # --- Prediction ---
        m_pred = m_est.copy()
        P_pred = P_est + Q

        # --- Measurement update ---
        H_list, innov_list = [], []
        for i, s in enumerate(imu_pos):
            r = residual_so3(frame[i], m_pred, s, e3_vec, gamma)
            H = jacobian_numeric(frame[i], m_pred, s, e3_vec, gamma)
            H_list.append(H)
            innov_list.append(-r)           # innovation = 0 - h(m) = -r

        H_full  = np.vstack(H_list)
        innov   = np.hstack(innov_list)
        R_big   = np.kron(np.eye(len(imu_pos)), R_single)

        S = H_full @ P_pred @ H_full.T + R_big
        K = P_pred @ H_full.T @ inv(S)

        m_est = m_pred + K @ innov

        # Joseph-form covariance update (numerically stable)
        IKH   = np.eye(STATE_DIM) - K @ H_full
        P_est = IKH @ P_pred @ IKH.T + K @ R_big @ K.T
        P_est = 0.5 * (P_est + P_est.T)

        hist.append(m_est.copy())

    hist = np.array(hist)                   # (steps, 5)
    err  = hist - m_true
    rmse_t = np.sqrt(np.mean(err ** 2, axis=1))

    return {
        "hist":        hist,
        "rmse_t":      rmse_t,
        "rmse_final":  float(np.sqrt(np.mean((m_est - m_true) ** 2))),
        "rmse_mean":   float(np.mean(rmse_t)),
    }


# ========================== MONTE CARLO ==========================

def run_monte_carlo(imu_pos: np.ndarray,
                    steps: int,
                    trials: int,
                    meas_std_deg: float,
                    seed_base: int,
                    gamma: int,
                    e3_vec: np.ndarray) -> Dict:
    """
    Run `trials` independent EKF trials (each with a unique random m_true and
    noise seed) and return aggregated statistics.
    """
    rmse_finals, rmse_means = [], []
    rmse_trajs = []

    for t in range(trials):
        rng    = np.random.RandomState(seed_base + t)
        m_true = rng.uniform(-2.0, 2.0, 5)

        meas_seq = generate_meas_seq(m_true, imu_pos, e3_vec, gamma,
                                     steps, meas_std_deg, rng)
        result   = run_ekf(m_true, meas_seq, imu_pos, e3_vec,
                           gamma, meas_std_deg)

        rmse_finals.append(result["rmse_final"])
        rmse_means.append(result["rmse_mean"])
        rmse_trajs.append(result["rmse_t"])

    trajs = np.array(rmse_trajs)            # (trials, steps)
    return {
        "rmse_final_mean": float(np.mean(rmse_finals)),
        "rmse_final_std":  float(np.std(rmse_finals)),
        "rmse_mean_mean":  float(np.mean(rmse_means)),
        "rmse_traj_mean":  trajs.mean(axis=0),
        "rmse_traj_std":   trajs.std(axis=0),
        "imu_pos":         imu_pos,
    }


# ========================== PLOTTING ==========================

# Consistent colors per sensor count
_COUNT_COLOR = {1: "#4878d0", 2: "#6acc65", 3: "#d65f5f",
                4: "#b47cc7", 5: "#c4ad66"}
_COUNT_LABEL = {1: "1 sensor", 2: "2 sensors", 3: "3 sensors",
                4: "4 sensors", 5: "5 sensors"}


def _pick_convergence_subset(stats_dict: Dict[str, Dict]) -> List[str]:
    """
    Pick one representative config per sensor count, plus the best 2-sensor
    placement, for the convergence plot.
    """
    by_count: Dict[int, List[Tuple[str, float]]] = {}
    for name, stat in stats_dict.items():
        n = len(stat["imu_pos"])
        by_count.setdefault(n, []).append((name, stat["rmse_final_mean"]))

    selected = []
    for n in sorted(by_count.keys()):
        # Pick the best (lowest RMSE) config for each sensor count
        best = min(by_count[n], key=lambda x: x[1])[0]
        selected.append(best)
    return selected


def plot_convergence(stats_dict: Dict[str, Dict], steps: int) -> plt.Figure:
    """RMSE convergence curves (mean ± 1 std) for one best config per count."""
    subset = _pick_convergence_subset(stats_dict)
    t = np.arange(1, steps + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in subset:
        stat  = stats_dict[name]
        n     = len(stat["imu_pos"])
        color = _COUNT_COLOR.get(n, "gray")
        mean  = stat["rmse_traj_mean"]
        std   = stat["rmse_traj_std"]
        ax.plot(t, mean, color=color, lw=2.0, label=f"{name}")
        ax.fill_between(t, mean - std, mean + std, alpha=0.18, color=color)

    ax.set_yscale("log")
    ax.set_xlabel("EKF step")
    ax.set_ylabel("RMSE  (modal coefficients)")
    ax.set_title("Convergence: best config per sensor count  (mean ± 1 std)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_all_convergence(stats_dict: Dict[str, Dict], steps: int) -> plt.Figure:
    """Full convergence curves panel, one subplot per sensor count."""
    by_count: Dict[int, List[str]] = {}
    for name, stat in stats_dict.items():
        n = len(stat["imu_pos"])
        by_count.setdefault(n, []).append(name)

    counts = sorted(by_count.keys())
    ncols  = min(3, len(counts))
    nrows  = (len(counts) + ncols - 1) // ncols
    t      = np.arange(1, steps + 1)

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
            ax.plot(t, mean, color=color, lw=1.6, label=name)
            ax.fill_between(t, mean - std, mean + std, alpha=0.12, color=color)
        ax.set_yscale("log")
        ax.set_title(f"{n} sensor(s)", fontsize=10)
        ax.set_xlabel("EKF step")
        ax.set_ylabel("RMSE")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3, which="both")

    # Hide unused subplots
    for idx in range(len(counts), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("RMSE convergence by sensor configuration  (mean ± 1 std)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_bar_chart(stats_dict: Dict[str, Dict]) -> plt.Figure:
    """
    Horizontal bar chart of final RMSE for all configurations,
    grouped and colored by sensor count.
    """
    names  = list(stats_dict.keys())
    means  = [stats_dict[n]["rmse_final_mean"] for n in names]
    stds   = [stats_dict[n]["rmse_final_std"]  for n in names]
    counts = [len(stats_dict[n]["imu_pos"])    for n in names]
    colors = [_COUNT_COLOR.get(c, "gray")      for c in counts]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, max(5, len(names) * 0.38)))
    ax.barh(y, means, xerr=stds, color=colors, alpha=0.85,
            edgecolor="white", lw=0.5, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Final RMSE  (mean ± std over trials)")
    ax.set_title("Sensor configuration comparison")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=_COUNT_COLOR[n], label=_COUNT_LABEL[n])
                      for n in sorted(_COUNT_COLOR)]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def plot_boxplot_by_count(stats_dict: Dict[str, Dict]) -> plt.Figure:
    """
    Box plot: distribution of per-trial final RMSE grouped by sensor count.
    Requires re-running MC to collect per-trial values; here we approximate
    from mean/std using the stored trajectory std as a proxy.
    Shows the mean + std across configurations of the same count.
    """
    by_count: Dict[int, Tuple[List, List]] = {}
    for name, stat in stats_dict.items():
        n = len(stat["imu_pos"])
        by_count.setdefault(n, ([], []))
        by_count[n][0].append(stat["rmse_final_mean"])
        by_count[n][1].append(stat["rmse_final_std"])

    counts = sorted(by_count.keys())
    fig, ax = plt.subplots(figsize=(7, 4))

    x_pos = np.arange(len(counts))
    for xi, n in zip(x_pos, counts):
        means = np.array(by_count[n][0])
        stds  = np.array(by_count[n][1])
        # Show all configs as scatter + mean bar
        ax.scatter(np.full(len(means), xi) + np.random.uniform(-0.12, 0.12, len(means)),
                   means, color=_COUNT_COLOR[n], zorder=3, s=40, alpha=0.8)
        ax.errorbar(xi, means.mean(), yerr=means.std(),
                    fmt="D", color="black", capsize=5, zorder=4,
                    ms=6, label="mean over configs" if xi == 0 else "")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{n}\nsensor(s)" for n in counts])
    ax.set_ylabel("Final RMSE  (mean over trials)")
    ax.set_title("Final RMSE vs sensor count\n(each dot = one configuration)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_2sensor_heatmap(trials: int, steps: int, meas_std_deg: float,
                         seed_base: int, gamma: int,
                         e3_vec: np.ndarray,
                         grid_n: int = 8) -> plt.Figure:
    """
    Sweep all ordered pairs from a uniform grid and plot mean final RMSE
    as a heatmap.  Uses a lighter MC (fewer trials and steps) to keep
    runtime reasonable.
    """
    grid   = np.linspace(0.10, 1.00, grid_n)
    hmap   = np.full((grid_n, grid_n), np.nan)
    n_pair = grid_n * (grid_n - 1) // 2

    print(f"\n  [2-sensor heatmap]  grid={grid_n}×{grid_n} → {n_pair} unique pairs"
          f"  ×{trials} trials each  (this may take a moment…)")

    done = 0
    for i, s1 in enumerate(grid):
        for j, s2 in enumerate(grid):
            if j <= i:
                continue
            stat = run_monte_carlo(
                np.array([s1, s2]), steps, trials,
                meas_std_deg, seed_base, gamma, e3_vec
            )
            val           = stat["rmse_final_mean"]
            hmap[i, j]    = val
            hmap[j, i]    = val         # symmetric
            done += 1
            if done % 5 == 0 or done == n_pair:
                print(f"    {done}/{n_pair} pairs done …", flush=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ext = [grid[0], grid[-1], grid[0], grid[-1]]
    im  = ax.imshow(hmap, origin="lower", aspect="auto", extent=ext,
                    cmap="RdYlGn_r", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Mean final RMSE")

    ax.set_xlabel("Sensor 2 position  s₂")
    ax.set_ylabel("Sensor 1 position  s₁")
    ax.set_title("2-Sensor placement sweep\n(lower = better; diagonal excluded)")
    fig.tight_layout()
    return fig


# ========================== SUMMARY TABLE ==========================

def print_summary(stats_dict: Dict[str, Dict]) -> None:
    hdr = f"{'Config':32s}  {'n':>2}  {'RMSE_final mean':>16}  {'std':>10}  {'RMSE_mean mean':>16}"
    print("\n" + "=" * len(hdr))
    print("SENSOR PLACEMENT SUMMARY")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    prev_n = -1
    for name, stat in stats_dict.items():
        n = len(stat["imu_pos"])
        if n != prev_n:
            if prev_n != -1:
                print()
            prev_n = n
        print(f"{name:32s}  {n:>2}  "
              f"{stat['rmse_final_mean']:>16.4e}  "
              f"{stat['rmse_final_std']:>10.2e}  "
              f"{stat['rmse_mean_mean']:>16.4e}")
    print("=" * len(hdr))


# ========================== MAIN ==========================

def main():
    parser = argparse.ArgumentParser(
        description="EKF sensor placement study for continuum-robot shape estimation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--trials",     type=int,   default=10,
                        help="Monte Carlo trials per configuration  (default: 10)")
    parser.add_argument("--steps",      type=int,   default=100,
                        help="EKF steps per trial  (default: 100)")
    parser.add_argument("--gamma",      type=int,   default=GAMMA,
                        help="Magnus PoE subdivisions  (default: 10)")
    parser.add_argument("--meas-std",   type=float, default=0.5,
                        help="Measurement noise std dev [deg]  (default: 0.5)")
    parser.add_argument("--seed-base",  type=int,   default=100,
                        help="Base random seed  (default: 100)")
    parser.add_argument("--no-heatmap", action="store_true",
                        help="Skip 2-sensor position heatmap")
    parser.add_argument("--fast",       action="store_true",
                        help="Quick run: 5 trials, 50 steps, 6×6 heatmap grid")
    parser.add_argument("--smoke",      action="store_true",
                        help="Very fast sanity check: 1 trial, 10 steps, no heatmap")
    parser.add_argument("--all-curves", action="store_true",
                        help="Show full convergence panel (all configurations)")
    args = parser.parse_args()

    # Apply fast-mode overrides
    if args.fast:
        args.trials = min(args.trials, 5)
        args.steps  = min(args.steps,  50)

    # Smoke-test mode is intentionally aggressive for quick iteration.
    if args.smoke:
        args.trials = min(args.trials, 1)
        args.steps  = min(args.steps, 10)
        args.no_heatmap = True

    e3_vec      = np.array([0.0, 0.0, L_PHYS])
    heatmap_cfg = dict(
        trials      = max(3, args.trials // 2),
        steps       = max(30, args.steps  // 2),
        grid_n      = 4 if args.smoke else (6 if args.fast else 9),
        meas_std_deg = args.meas_std,
        seed_base   = args.seed_base,
        gamma       = args.gamma,
        e3_vec      = e3_vec,
    )

    print(f"\n{'='*62}")
    print("EKF SENSOR PLACEMENT STUDY")
    print(f"{'='*62}")
    print(f"  Trials per config : {args.trials}")
    print(f"  Steps per trial   : {args.steps}")
    print(f"  Magnus gamma      : {args.gamma}")
    print(f"  Meas noise        : {args.meas_std} deg")
    print(f"  Seed base         : {args.seed_base}")
    print(f"{'='*62}\n")

    configs     = build_configs()
    stats_dict  = {}

    for name, imu_pos in configs.items():
        n = len(imu_pos)
        print(f"  {name:32s}  n={n}  …", end=" ", flush=True)
        stat = run_monte_carlo(
            imu_pos, args.steps, args.trials, args.meas_std,
            args.seed_base, args.gamma, e3_vec
        )
        stats_dict[name] = stat
        print(f"RMSE = {stat['rmse_final_mean']:.4e} ± {stat['rmse_final_std']:.2e}")

    print_summary(stats_dict)

    # --- Figures ---
    plot_convergence(stats_dict, args.steps)
    plot_bar_chart(stats_dict)
    plot_boxplot_by_count(stats_dict)

    if args.all_curves:
        plot_all_convergence(stats_dict, args.steps)

    if not args.no_heatmap:
        plot_2sensor_heatmap(**heatmap_cfg)

    plt.show()


if __name__ == "__main__":
    main()
