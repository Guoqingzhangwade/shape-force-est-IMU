#!/usr/bin/env python3
"""
Constrained Kirchhoff-rod Ground-Truth Datasets.

Generates three separate GT datasets for the constrained tip-wrench study:

  Case 1 — force_only        l_ext = 0,  f_ext ~ U[-F, F]^3        (world frame)
  Case 2 — transverse_force  l_ext = 0,  f_ext ~ U[-F,F]^2 × {0}   (world XY only)
  Case 3 — moment_only       f_ext = 0,  l_ext ~ U[-M, M]^3         (world frame)

All three cases use the same rod geometry, tendon layout, and cable-tension
sampling as the existing general Kirchhoff study (Step 1).  They are saved to
a separate folder so the original study is completely untouched.

Output folder:  gt_data_constrained/
  kirchhoff_gt_force_only.npz / .json
  kirchhoff_gt_transverse_force.npz / .json
  kirchhoff_gt_moment_only.npz / .json
  kirchhoff_gt_*_overview.png   (optional)

NPZ arrays (identical layout to kirchhoff_gt_dataset.npz)
----------------------------------------------------------
  case_id      : (N,)             int32
  tau          : (N, 4)           cable tensions [N]
  f_ext        : (N, 3)           GT tip force   [N]   — world frame
  l_ext        : (N, 3)           GT tip moment  [N·m] — world frame
  positions    : (N, num_pts, 3)  backbone centreline p(s) [m]
  orientations : (N, num_pts, 9)  R(s) row-major flattened
  T_tip        : (N, 4, 4)        tip homogeneous transform
  meta         : scalar           JSON string

Frame notes
-----------
f_ext and l_ext are expressed in the *world* (global) frame.
  Case 1  : l_ext = 0 identically; f_ext is fully general 3-D world force.
  Case 2  : l_ext = 0 identically; f_ext[2] = 0 (zero world-z component).
             In the estimator the constraint is posed in the body (tip) frame
             [S_trans = diag(0,0,0,1,1,0)], which is approximately equal to
             zero world-z for mildly deflected rods.
  Case 3  : f_ext = 0 identically; l_ext is fully general 3-D world moment.

Usage
-----
  cd src/shape_force_est_imu/crt
  python generate_kirchhoff_gt_dataset_constrained.py
  python generate_kirchhoff_gt_dataset_constrained.py --num-samples 100 --seed 42
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers (mirrored from generate_kirchhoff_gt_dataset.py)
# ---------------------------------------------------------------------------

def _git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__) or ".",
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def sample_tensions(n_tendon: int = 4, tau_max: float = 4.0) -> np.ndarray:
    """One or two contiguous tendons active; tension in [1.0, tau_max] N."""
    tau = np.zeros(n_tendon)
    k   = np.random.choice([1, 2])
    start = np.random.randint(0, n_tendon)
    idx   = [start] if k == 1 else [start, (start + 1) % n_tendon]
    tau[idx] = np.random.uniform(1.0, tau_max, size=len(idx))
    return tau


def sample_tip_wrench_constrained(
    case: str,
    force_range: float = 0.4,
    moment_range: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample GT tip wrench according to constrained case.

    Parameters
    ----------
    case : one of "force_only", "transverse_force", "moment_only"
    force_range  : uniform half-range for force components [N]
    moment_range : uniform half-range for moment components [N·m]

    Returns (f_ext, l_ext) both in the world frame.
    """
    if case == "force_only":
        f_ext = np.random.uniform(-force_range, force_range, 3)
        l_ext = np.zeros(3)

    elif case == "transverse_force":
        # Zero world-z force and zero moment.
        # In the estimator the S matrix is posed in body (tip) frame, which
        # approximates the world-XY constraint for small deflections.
        f_ext = np.zeros(3)
        f_ext[:2] = np.random.uniform(-force_range, force_range, 2)
        l_ext = np.zeros(3)

    elif case == "moment_only":
        f_ext = np.zeros(3)
        l_ext = np.random.uniform(-moment_range, moment_range, 3)

    else:
        raise ValueError(f"Unknown constrained case: {case!r}")

    return f_ext, l_ext


def _tip_lateral_deflection(T_tip: np.ndarray) -> float:
    p = T_tip[:3, 3]
    return float(np.sqrt(p[0]**2 + p[1]**2))


def _extract_states(traj_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return traj_y[0:3, :].T.copy(), traj_y[3:12, :].T.copy()


def _build_T_tip(traj_y: np.ndarray) -> np.ndarray:
    yL = traj_y[:, -1]
    T  = np.eye(4)
    T[:3, :3] = yL[3:12].reshape(3, 3)
    T[:3,  3] = yL[0:3]
    return T


def _resample_trajectory(traj, s_uniform: np.ndarray) -> np.ndarray:
    """
    Resample a solve_ivp result from its adaptive arc-length grid to a
    uniform grid s_uniform, using linear interpolation per state row.

    Using s_eval=None during forward_kinematics lets the ODE solver use its
    natural adaptive step (more stable shooting convergence); this function
    then maps the result to the fixed-size output required by the GT arrays.
    """
    fn = interp1d(traj.t, traj.y, kind="linear", axis=1,
                  bounds_error=False, fill_value="extrapolate")
    return fn(s_uniform)   # (18, len(s_uniform))


# ---------------------------------------------------------------------------
# Overview figure (minimal)
# ---------------------------------------------------------------------------

def _plot_overview(positions_list, T_tip_list, rod_length, title, save_path):
    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection="3d")
    for k, p in enumerate(positions_list):
        ax.plot(p[:, 0], p[:, 1], p[:, 2], lw=0.8, alpha=0.45,
                color=plt.cm.tab20(k / max(len(positions_list) - 1, 1)))
    ax.plot([0, 0], [0, 0], [0, rod_length], "k-", lw=1.5, label="undeformed")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(title, fontsize=9)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Figure -> {save_path}")


# ---------------------------------------------------------------------------
# Per-case dataset generation
# ---------------------------------------------------------------------------

# Minimal s_eval used during the FAST CHECK phase.
# Passing only the endpoint means the ODE solver's shooting phase does not
# need to hit 100 intermediate points on every least_squares iteration —
# it can use its natural adaptive step size (max_step = L/N).  This makes
# the rejection loop ~10–30× faster with no loss of solution accuracy.
_S_EVAL_CHECK = None   # None → solver uses its own adaptive output grid


def generate_one_case(
    case: str,
    rod,
    s_eval: np.ndarray,
    num_samples: int,
    max_attempts: int,
    tau_max: float,
    force_range: float,
    moment_range: float,
    min_deflection: float,
) -> dict:
    """
    Generate a single constrained dataset.

    Two-step strategy (performance fix)
    ------------------------------------
    1. FAST CHECK  — call forward_kinematics with s_eval=None so the ODE
       solver uses its natural adaptive grid (no forced intermediate points).
       Limits max_nfev=200 (usually converges in <50 LM iterations).
       Used only to accept/reject the sample; no trajectory stored.

    2. FULL TRAJECTORY — for accepted samples only, re-solve with the full
       100-point s_eval to get the fine-grained backbone for storage.
       This second solve is fast because the shooting converges in very few
       iterations (the BVP is the same; only the output grid differs).

    Returns a dict of lists: case_ids, tau_list, f_list, l_list,
    pos_list, ori_list, T_tip_list.
    """
    case_ids, tau_list, f_list, l_list = [], [], [], []
    pos_list, ori_list, T_tip_list     = [], [], []

    attempts = n_rejected = n_solver_fail = 0
    target   = num_samples
    t_case_start = time.perf_counter()

    # Solver options: limit LM iterations to 200 — converges quickly or not at all.
    _fast_opts = {"max_nfev": 200}
    _full_opts = {"max_nfev": 400}

    case_label = case.replace("_", " ")
    print(f"\n  [{case_label}]  generating {target} samples ...")

    while len(case_ids) < target:
        attempts += 1
        if attempts > max_attempts:
            print(
                f"  WARNING: reached max_attempts={max_attempts}. "
                f"Collected {len(case_ids)}/{target}."
            )
            break

        tau        = sample_tensions(n_tendon=rod.n_tendon, tau_max=tau_max)
        f_ext, l_ext = sample_tip_wrench_constrained(
            case, force_range=force_range, moment_range=moment_range
        )

        # ---- STEP 1: fast check (no forced intermediate eval points) --------
        try:
            T_tip_check = rod.forward_kinematics(
                tau=tau, f_ext=f_ext, l_ext=l_ext,
                return_states=False,        # only need T_tip for the check
                s_eval=_S_EVAL_CHECK,       # None → solver uses adaptive grid
                solver_opts=_fast_opts,
            )
        except RuntimeError:
            n_solver_fail += 1
            continue

        if _tip_lateral_deflection(T_tip_check) < min_deflection:
            n_rejected += 1
            continue

        # ---- STEP 2: full trajectory (accepted sample only) ----------------
        # Re-integrate with s_eval=None so the solver's natural adaptive grid
        # is used (same improved convergence as the fast check).  The result
        # is then resampled to the uniform num_pts grid via linear interpolation.
        try:
            _, traj = rod.forward_kinematics(
                tau=tau, f_ext=f_ext, l_ext=l_ext,
                return_states=True,
                s_eval=None,           # adaptive grid → resampled below
                solver_opts=_full_opts,
            )
        except RuntimeError:
            n_solver_fail += 1
            continue

        # Resample adaptive solution to the uniform 100-point arc-length grid
        traj_y_resampled = _resample_trajectory(traj, s_eval)
        T_tip      = _build_T_tip(traj_y_resampled)
        positions, orientations = _extract_states(traj_y_resampled)
        k = len(case_ids)
        case_ids.append(k)
        tau_list.append(tau)
        f_list.append(f_ext)
        l_list.append(l_ext)
        pos_list.append(positions)
        ori_list.append(orientations)
        T_tip_list.append(T_tip)

        if (k + 1) % 5 == 0 or (k + 1) == target:
            elapsed = time.perf_counter() - t_case_start
            rate    = (k + 1) / elapsed
            eta_s   = (target - k - 1) / rate if rate > 0 else float("inf")
            print(f"    [{k+1:3d}/{target}]  attempts={attempts:4d}  "
                  f"fails={n_solver_fail}  rejected={n_rejected}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta_s:.0f}s")

    n_col = len(case_ids)
    print(f"  Collected {n_col}/{target}  "
          f"(solver fails={n_solver_fail}, rejected={n_rejected})")

    return {
        "case_ids": case_ids,
        "tau_list": tau_list,
        "f_list":   f_list,
        "l_list":   l_list,
        "pos_list": pos_list,
        "ori_list": ori_list,
        "T_tip_list": T_tip_list,
    }


def pack_and_save(
    data: dict,
    case: str,
    args,
    save_dir: Path,
    stem: str,
    rod_length: float,
    num_points: int,
) -> None:
    """Pack lists into arrays and save NPZ + JSON."""
    case_id_arr  = np.array(data["case_ids"],   dtype=np.int32)
    tau_arr      = np.vstack(data["tau_list"])
    f_arr        = np.vstack(data["f_list"])
    l_arr        = np.vstack(data["l_list"])
    pos_arr      = np.stack(data["pos_list"],     axis=0)
    ori_arr      = np.stack(data["ori_list"],     axis=0)
    T_tip_arr    = np.stack(data["T_tip_list"],   axis=0)

    meta = {
        "description"       : f"Constrained Kirchhoff-rod GT dataset — {case}",
        "constrained_case"  : case,
        "n_samples"         : len(data["case_ids"]),
        "num_points"        : num_points,
        "length_m"          : rod_length,
        "backbone_radius"   : args.backbone_radius,
        "youngs_modulus"    : args.youngs_modulus,
        "tendon_offset"     : args.tendon_offset,
        "num_disks"         : args.num_disks,
        "tau_max"           : args.tau_max,
        "force_range"       : args.force_range,
        "moment_range"      : args.moment_range,
        "min_deflection"    : args.min_deflection,
        "rng_seed"          : args.seed,
        "date"              : datetime.datetime.now().isoformat(timespec="seconds"),
        "git_hash"          : _git_hash(),
        "wrench_frame"      : "world (global reference)",
        "constrained_case_notes": {
            "force_only"       : "l_ext = 0 identically; f_ext random 3D world force",
            "transverse_force" : "l_ext = 0; f_ext[2] = 0 (zero world-z force component)",
            "moment_only"      : "f_ext = 0 identically; l_ext random 3D world moment",
        },
        "array_layout": {
            "case_id"      : "(N,) int32",
            "tau"          : "(N, 4)  cable tensions [N]",
            "f_ext"        : "(N, 3)  tip forces [N] — world frame",
            "l_ext"        : "(N, 3)  tip moments [N·m] — world frame",
            "positions"    : "(N, num_pts, 3)  backbone centreline p(s) [m]",
            "orientations" : "(N, num_pts, 9)  R(s) flattened row-major",
            "T_tip"        : "(N, 4, 4)  tip homogeneous transform",
        },
    }

    npz_path  = save_dir / f"{stem}.npz"
    json_path = save_dir / f"{stem}.json"

    np.savez_compressed(
        npz_path,
        case_id      = case_id_arr,
        tau          = tau_arr,
        f_ext        = f_arr,
        l_ext        = l_arr,
        positions    = pos_arr,
        orientations = ori_arr,
        T_tip        = T_tip_arr,
        meta         = json.dumps(meta),
    )
    json_path.write_text(json.dumps(meta, indent=2))

    print(f"  Saved: {npz_path}")
    print(f"         {json_path}")
    print(f"  Shapes:  tau={tau_arr.shape}  positions={pos_arr.shape}"
          f"  orientations={ori_arr.shape}")

    if args.plot:
        fig_path = save_dir / f"{stem}_overview.png"
        _plot_overview(
            data["pos_list"], data["T_tip_list"],
            rod_length,
            title=f"Constrained GT — {case}  ({len(data['case_ids'])} configs)",
            save_path=fig_path,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_CASE_STEMS = {
    "force_only"       : "kirchhoff_gt_force_only",
    "transverse_force" : "kirchhoff_gt_transverse_force",
    "moment_only"      : "kirchhoff_gt_moment_only",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate constrained Kirchhoff-rod GT datasets (force-only, "
                    "transverse-force, moment-only).\n\n"
                    "Runtime note: the Kirchhoff shooting solver (Python BVP) takes\n"
                    "~3-7 s per sample on a typical laptop.  Expect 5-20 min per case\n"
                    "depending on solver success rate.  Use --quick for a fast test run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-samples",    type=int,   default=50,
                        help="target valid samples per case")
    parser.add_argument("--max-attempts",   type=int,   default=800,
                        help="max solver attempts before giving up for a case")
    parser.add_argument("--num-points",     type=int,   default=100,
                        help="arc-length evaluation points per shape in the saved dataset")
    parser.add_argument("--seed",           type=int,   default=456,
                        help="RNG seed (different from main study seed=123)")
    parser.add_argument("--tau-max",        type=float, default=4.0)
    parser.add_argument("--force-range",    type=float, default=0.4,
                        help="uniform half-range for tip force [N]")
    parser.add_argument("--moment-range",   type=float, default=0.04,
                        help="uniform half-range for tip moment [N·m]")
    parser.add_argument("--min-deflection", type=float, default=5e-4,
                        help="min lateral tip deflection to accept [m] "
                             "(lowered from 1e-3 to help moment-only case)")
    # Rod geometry — must match main study
    parser.add_argument("--length",           type=float, default=0.10)
    parser.add_argument("--backbone-radius",  type=float, default=5e-4)
    parser.add_argument("--youngs-modulus",   type=float, default=60e9)
    parser.add_argument("--tendon-offset",    type=float, default=0.008)
    parser.add_argument("--num-disks",        type=int,   default=40)
    # Output
    parser.add_argument("--save-dir",   default="gt_data_constrained",
                        help="output folder (separate from gt_data/ to preserve original)")
    parser.add_argument("--cases", nargs="+",
                        choices=["force_only", "transverse_force", "moment_only"],
                        default=["force_only", "transverse_force", "moment_only"],
                        help="which constrained cases to generate")
    parser.add_argument("--quick", action="store_true",
                        help="fast test run: 10 samples, 100 max attempts per case")
    parser.add_argument("--plot",    action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    args = parser.parse_args()

    if args.quick:
        args.num_samples  = 10
        args.max_attempts = 150
        print("--quick mode: 10 samples, 150 max attempts per case")

    np.random.seed(args.seed)

    try:
        from cosserat_rod_model import CosseratRodModel
    except ModuleNotFoundError:
        sys.exit(
            "ERROR: cannot import 'cosserat_rod_model'.\n"
            "Run from inside the crt/ directory:\n"
            "  cd src/shape_force_est_imu/crt\n"
            "  python generate_kirchhoff_gt_dataset_constrained.py"
        )

    angles = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    tendon_routing = [
        np.array([args.tendon_offset * np.cos(a),
                  args.tendon_offset * np.sin(a), 0.0])
        for a in angles
    ]
    rod = CosseratRodModel(
        length          = args.length,
        backbone_radius = args.backbone_radius,
        youngs_modulus  = args.youngs_modulus,
        tendon_routing  = tendon_routing,
        num_disks       = args.num_disks,
        inextensible    = True,
    )

    s_eval   = np.linspace(0.0, rod.L, args.num_points)
    save_dir = Path(__file__).resolve().parent / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Constrained GT dataset generation")
    print(f"  Save dir  : {save_dir}")
    print(f"  Cases     : {args.cases}")
    print(f"  Samples   : {args.num_samples} per case")
    print(f"  Seed      : {args.seed}")
    print(f"  Note: the Cosserat BVP solver takes ~3-8 s/sample.")
    print(f"        {args.num_samples} samples => {args.num_samples * 8 // 60}-"
          f"{args.num_samples * 20 // 60} min per case (depends on success rate).")

    for case in args.cases:
        data = generate_one_case(
            case         = case,
            rod          = rod,
            s_eval       = s_eval,
            num_samples  = args.num_samples,
            max_attempts = args.max_attempts,
            tau_max      = args.tau_max,
            force_range  = args.force_range,
            moment_range = args.moment_range,
            min_deflection = args.min_deflection,
        )
        pack_and_save(
            data      = data,
            case      = case,
            args      = args,
            save_dir  = save_dir,
            stem      = _CASE_STEMS[case],
            rod_length = rod.L,
            num_points = args.num_points,
        )

    print("\nAll constrained datasets generated.")


if __name__ == "__main__":
    main()
