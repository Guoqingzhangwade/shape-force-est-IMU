#!/usr/bin/env python3
"""
Cross-Model Robustness Study — Step 1: Kirchhoff-Rod Ground-Truth Dataset.

Generates a constrained random dataset of static rod configurations using
the inextensible Cosserat (Kirchhoff) rod model and saves them to an NPZ
file for later use in Steps 2 (IMU synthesis) and 3 (EKF robustness eval).

Sampling design
---------------
* 4 tendons; each sample activates either one tendon or two contiguous tendons.
* Active tension drawn from U[1.0, tau_max] N.
* Random tip wrench: f_ext ~ U[-0.4, 0.4]^3 N, l_ext ~ U[-0.04, 0.04]^3 N·m.
* If the shooting solver fails, the sample is discarded and re-drawn.
* Light filtering: reject near-trivial cases where lateral tip deflection < 1 mm.

Output files (in --save-dir)
----------------------------
  kirchhoff_gt_dataset.npz   — compressed arrays (see below)
  kirchhoff_gt_dataset.json  — metadata / rod parameters
  kirchhoff_gt_overview.png  — 3-D overview figure

NPZ arrays
----------
  case_id     : (N,)            integer index 0 … N-1
  tau         : (N, 4)          cable tensions [N]
  f_ext       : (N, 3)          tip forces [N]
  l_ext       : (N, 3)          tip moments [N·m]
  positions   : (N, num_pts, 3) backbone centreline p(s) [m]
  orientations: (N, num_pts, 9) rotation matrix R(s) flattened row-major
  T_tip       : (N, 4, 4)       tip homogeneous transform
  meta        : scalar          JSON string (rod params, date, seed, …)

Usage
-----
  python generate_kirchhoff_gt_dataset.py
  python generate_kirchhoff_gt_dataset.py --num-samples 50 --plot
  python generate_kirchhoff_gt_dataset.py --num-samples 100 --plot-all-tip-frames
  python generate_kirchhoff_gt_dataset.py --num-samples 200 --seed 0 --save-dir results/gt
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive; switched to TkAgg if --show is set
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__) or ".",
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def sample_tensions(n_tendon: int = 4, tau_max: float = 4.0) -> np.ndarray:
    """One or two contiguous tendons active; tension in [1.0, tau_max] N."""
    tau = np.zeros(n_tendon)
    k = np.random.choice([1, 2])
    start = np.random.randint(0, n_tendon)
    idx = [start] if k == 1 else [start, (start + 1) % n_tendon]
    tau[idx] = np.random.uniform(1.0, tau_max, size=len(idx))
    return tau


def sample_tip_wrench(
    force_range: float = 0.4, moment_range: float = 0.04
) -> tuple[np.ndarray, np.ndarray]:
    f_ext = np.random.uniform(-force_range, force_range, 3)
    l_ext = np.random.uniform(-moment_range, moment_range, 3)
    return f_ext, l_ext


def _tip_lateral_deflection(T_tip: np.ndarray) -> float:
    """Lateral distance of tip from the undeformed straight-rod tip (0, 0, L)."""
    p_tip = T_tip[:3, 3]
    return float(np.sqrt(p_tip[0] ** 2 + p_tip[1] ** 2))


def _extract_states(traj_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract positions and rotation matrices from ODE solution array.

    traj_y rows:  0:3  -> p(s),   3:12 -> R(s) flattened,   12:18 -> n/u (ignored)
    Returns
    -------
    positions    : (n_pts, 3)
    orientations : (n_pts, 9)  row-major flattened R(s)
    """
    positions    = traj_y[0:3, :].T.copy()           # (n_pts, 3)
    orientations = traj_y[3:12, :].T.copy()          # (n_pts, 9)
    return positions, orientations


def _build_T_tip(traj_y: np.ndarray) -> np.ndarray:
    yL = traj_y[:, -1]
    T  = np.eye(4)
    T[:3, :3] = yL[3:12].reshape(3, 3)
    T[:3,  3] = yL[0:3]
    return T


# ---------------------------------------------------------------------------
# 3-D visualisation
# ---------------------------------------------------------------------------

_AXIS_COLORS     = ("r", "g", "b")
_REF_FRAME_SCALE = 0.010   # reference backbone triad arm length [m]
_REF_FRAME_S     = (0.0, 0.25, 0.50, 0.75, 1.00)   # normalised arc-lengths


def _draw_frame(
    ax,
    origin: np.ndarray,
    R: np.ndarray,
    scale: float = _REF_FRAME_SCALE,
    lw: float = 1.2,
) -> None:
    """Draw an RGB triad at `origin` with rotation matrix `R`."""
    for col, col_idx in zip(_AXIS_COLORS, range(3)):
        ax.quiver(
            *origin, *(scale * R[:, col_idx]),
            color=col, linewidth=lw, arrow_length_ratio=0.25,
        )


def plot_overview(
    positions_list: list[np.ndarray],
    T_tip_list: list[np.ndarray],
    rod_length: float = 0.10,
    show_reference_frames: bool = True,
    hide_sample_tip_frames: bool = True,
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    """
    3-D overview: sampled shapes (light) + straight reference backbone + frames.

    Reference backbone
    ------------------
    A dark straight line from [0,0,0] to [0,0,L] represents the nominal
    undeformed rod.  Identity-rotation frame triads are drawn at normalised
    arc-lengths ``_REF_FRAME_S`` along it.

    Sampled shapes
    --------------
    All collected Kirchhoff-rod shapes are plotted with low alpha (0.25) so
    the reference backbone remains visually dominant.  Sample tip frames are
    hidden by default (``hide_sample_tip_frames=True``).
    """
    fig = plt.figure(figsize=(9, 8))
    ax  = fig.add_subplot(111, projection="3d")

    n    = len(positions_list)
    cmap = plt.cm.tab20

    # --- sampled shapes: light visual weight, no tip frames by default ---
    for k, p in enumerate(positions_list):
        color = cmap(k / max(n - 1, 1))
        ax.plot(p[:, 0], p[:, 1], p[:, 2], lw=1.0, alpha=0.55, color=color)
        if not hide_sample_tip_frames:
            _draw_frame(ax, T_tip_list[k][:3, 3], T_tip_list[k][:3, :3],
                        scale=0.006, lw=0.6)

    # --- reference straight backbone (undeformed, z-axis) ---
    ax.plot(
        [0.0, 0.0], [0.0, 0.0], [0.0, rod_length],
        color="k", lw=1.8, zorder=5, label="reference (undeformed)",
    )

    # --- reference frames at fixed arc-length stations ---
    if show_reference_frames:
        R_id = np.eye(3)   # identity rotation: straight rod aligned with z
        for t in _REF_FRAME_S:
            _draw_frame(
                ax,
                origin=np.array([0.0, 0.0, t * rod_length]),
                R=R_id,
                scale=_REF_FRAME_SCALE,
                lw=1.8,
            )

    ax.set_xlabel("x  [m]", fontsize=10)
    ax.set_ylabel("y  [m]", fontsize=10)
    ax.set_zlabel("z  [m]", fontsize=10)
    ax.set_title(
        f"Kirchhoff-rod ground-truth dataset  ({n} configs)\n"
        "black = reference backbone,  colour = deformed shapes",
        fontsize=10,
    )
    ax.set_box_aspect([1, 1, 1])

    # Axis limits: x/y centred on data cloud; z anchored at 0 so the base
    # frame sits at the visual floor of the plot.
    ref_pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, rod_length]])
    all_pts = np.vstack([np.vstack(positions_list), ref_pts])
    lo, hi  = all_pts.min(0), all_pts.max(0)
    span    = hi - lo
    pad     = span * 0.07
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"  Figure saved → {save_path}")
    if show:
        matplotlib.use("TkAgg")
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: generate Kirchhoff-rod ground-truth dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-samples",    type=int,   default=50,
                        help="number of valid configurations to collect")
    parser.add_argument("--max-attempts",   type=int,   default=500,
                        help="max solver attempts before aborting")
    parser.add_argument("--num-points",     type=int,   default=100,
                        help="arc-length evaluation points per shape")
    parser.add_argument("--seed",           type=int,   default=123,
                        help="NumPy RNG seed")
    parser.add_argument("--tau-max",        type=float, default=4.0,
                        help="upper bound on active cable tension [N]")
    parser.add_argument("--force-range",    type=float, default=0.4,
                        help="uniform half-range for tip force sampling [N]")
    parser.add_argument("--moment-range",   type=float, default=0.04,
                        help="uniform half-range for tip moment sampling [N·m]")
    parser.add_argument("--min-deflection", type=float, default=1e-3,
                        help="minimum lateral tip deflection to accept [m]")
    # Rod geometry
    parser.add_argument("--length",            type=float, default=0.10)
    parser.add_argument("--backbone-radius",   type=float, default=5e-4)
    parser.add_argument("--youngs-modulus",    type=float, default=60e9)
    parser.add_argument("--tendon-offset",     type=float, default=0.008)
    parser.add_argument("--num-disks",         type=int,   default=40,
                        help="ODE discretisation disks (affects solver accuracy)")
    parser.add_argument("--inextensible",      action="store_true", default=True,
                        help="use Kirchhoff (inextensible) rod model")
    # Output
    parser.add_argument("--save-dir",          type=str,   default="gt_data",
                        help="directory to write output files")
    parser.add_argument("--outfile",           type=str,   default="kirchhoff_gt_dataset.npz",
                        help="filename for the NPZ dataset (inside --save-dir)")
    # Visualisation
    parser.add_argument("--plot",               action="store_true", default=True,
                        help="save overview 3-D figure (default on)")
    parser.add_argument("--no-plot",            dest="plot", action="store_false",
                        help="disable figure saving")
    parser.add_argument("--show-reference-frames", action="store_true", default=True,
                        help="draw RGB triads along the reference straight backbone")
    parser.add_argument("--no-show-reference-frames", dest="show_reference_frames",
                        action="store_false",
                        help="omit reference frame triads from the overview figure")
    parser.add_argument("--hide-sample-tip-frames", action="store_true", default=True,
                        help="hide tip frames on sampled shapes (default; cleaner plot)")
    parser.add_argument("--show-sample-tip-frames", dest="hide_sample_tip_frames",
                        action="store_false",
                        help="draw tip frames for every sampled shape")
    parser.add_argument("--show",               action="store_true",
                        help="display figure interactively (requires a display)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Build rod model
    # ------------------------------------------------------------------
    # Local import so the script works from inside the crt/ directory
    try:
        from cosserat_rod_model import CosseratRodModel
    except ModuleNotFoundError:
        sys.exit(
            "ERROR: cannot import 'cosserat_rod_model'.\n"
            "Run this script from inside the crt/ directory:\n"
            "  cd src/shape_force_est_imu/crt\n"
            "  python generate_kirchhoff_gt_dataset.py"
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
        inextensible    = args.inextensible,
    )

    s_eval = np.linspace(0.0, rod.L, args.num_points)

    # ------------------------------------------------------------------
    # Sampling loop
    # ------------------------------------------------------------------
    case_ids: list[int]       = []
    tau_list: list[np.ndarray] = []
    f_list:   list[np.ndarray] = []
    l_list:   list[np.ndarray] = []
    pos_list: list[np.ndarray] = []
    ori_list: list[np.ndarray] = []
    T_tip_list: list[np.ndarray] = []

    attempts     = 0
    n_rejected   = 0
    n_solver_fail = 0
    target       = args.num_samples
    max_attempts = args.max_attempts

    print(f"Generating {target} valid Kirchhoff-rod configurations …")

    while len(case_ids) < target:
        attempts += 1
        if attempts > max_attempts:
            print(
                f"\nWARNING: reached max_attempts={max_attempts}. "
                f"Collected {len(case_ids)}/{target} samples "
                f"({n_solver_fail} solver failures, {n_rejected} rejected)."
            )
            break

        tau   = sample_tensions(n_tendon=rod.n_tendon, tau_max=args.tau_max)
        f_ext, l_ext = sample_tip_wrench(
            force_range=args.force_range, moment_range=args.moment_range
        )

        try:
            _, traj = rod.forward_kinematics(
                tau          = tau,
                f_ext        = f_ext,
                l_ext        = l_ext,
                return_states= True,
                s_eval       = s_eval,
            )
        except RuntimeError:
            n_solver_fail += 1
            continue

        T_tip = _build_T_tip(traj.y)

        # Light validity filter: reject near-trivial undeformed shapes
        if _tip_lateral_deflection(T_tip) < args.min_deflection:
            n_rejected += 1
            continue

        positions, orientations = _extract_states(traj.y)

        k = len(case_ids)
        case_ids.append(k)
        tau_list.append(tau)
        f_list.append(f_ext)
        l_list.append(l_ext)
        pos_list.append(positions)
        ori_list.append(orientations)
        T_tip_list.append(T_tip)

        if (k + 1) % 10 == 0 or (k + 1) == target:
            print(f"  [{k+1:4d}/{target}]  attempts={attempts}  "
                  f"failures={n_solver_fail}  rejected={n_rejected}")

    n_collected = len(case_ids)
    print(f"\nCollection complete: {n_collected} valid samples "
          f"in {attempts} attempts  "
          f"(solver failures: {n_solver_fail}, "
          f"trivial rejected: {n_rejected})")

    if n_collected == 0:
        sys.exit("No valid samples collected. Adjust --tau-max or --max-attempts.")

    # ------------------------------------------------------------------
    # Pack arrays
    # ------------------------------------------------------------------
    case_id_arr  = np.array(case_ids,      dtype=np.int32)    # (N,)
    tau_arr      = np.vstack(tau_list)                         # (N, 4)
    f_arr        = np.vstack(f_list)                           # (N, 3)
    l_arr        = np.vstack(l_list)                           # (N, 3)
    pos_arr      = np.stack(pos_list,  axis=0)                 # (N, num_pts, 3)
    ori_arr      = np.stack(ori_list,  axis=0)                 # (N, num_pts, 9)
    T_tip_arr    = np.stack(T_tip_list, axis=0)                # (N, 4, 4)

    meta = {
        "description"     : "Kirchhoff-rod GT dataset for cross-model robustness study",
        "n_samples"       : n_collected,
        "num_points"      : args.num_points,
        "length_m"        : args.length,
        "backbone_radius" : args.backbone_radius,
        "youngs_modulus"  : args.youngs_modulus,
        "tendon_offset"   : args.tendon_offset,
        "num_disks"       : args.num_disks,
        "inextensible"    : args.inextensible,
        "tau_max"         : args.tau_max,
        "force_range"     : args.force_range,
        "moment_range"    : args.moment_range,
        "min_deflection"  : args.min_deflection,
        "rng_seed"        : args.seed,
        "date"            : datetime.datetime.now().isoformat(timespec="seconds"),
        "git_hash"        : _git_hash(),
        "array_layout"    : {
            "case_id"     : "(N,) int32",
            "tau"         : "(N, 4)  cable tensions [N]",
            "f_ext"       : "(N, 3)  tip forces [N]",
            "l_ext"       : "(N, 3)  tip moments [N·m]",
            "positions"   : "(N, num_pts, 3)  backbone centreline p(s) [m]",
            "orientations": "(N, num_pts, 9)  R(s) flattened row-major",
            "T_tip"       : "(N, 4, 4)  tip homogeneous transform",
        },
    }

    # ------------------------------------------------------------------
    # Save dataset
    # ------------------------------------------------------------------
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    npz_path  = save_dir / args.outfile
    json_path = save_dir / Path(args.outfile).with_suffix(".json").name

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

    print(f"\nDataset saved:")
    print(f"  {npz_path}   (case_id, tau, f_ext, l_ext, positions, orientations, T_tip, meta)")
    print(f"  {json_path}  (metadata)")
    print(f"\nArray shapes:")
    print(f"  tau          {tau_arr.shape}")
    print(f"  positions    {pos_arr.shape}")
    print(f"  orientations {ori_arr.shape}")
    print(f"  T_tip        {T_tip_arr.shape}")

    # ------------------------------------------------------------------
    # Overview plot
    # ------------------------------------------------------------------
    if args.plot or args.show:
        fig_path = save_dir / Path(args.outfile).stem
        fig_path = fig_path.parent / (fig_path.name + "_overview.png")
        plot_overview(
            positions_list         = pos_list,
            T_tip_list             = T_tip_list,
            rod_length             = rod.L,
            show_reference_frames  = args.show_reference_frames,
            hide_sample_tip_frames = args.hide_sample_tip_frames,
            save_path              = fig_path if args.plot else None,
            show                   = args.show,
        )


if __name__ == "__main__":
    main()
