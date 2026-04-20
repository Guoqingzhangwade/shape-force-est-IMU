#!/usr/bin/env python3
"""
Cross-Model Robustness Study — Step 2: Synthetic IMU Measurements.

Reads the saved Kirchhoff-rod ground-truth dataset (Step 1) and generates
noisy sparse IMU orientation measurements for the sensor layouts used in the
shape-estimation study.  No rod solver is run here.

Scientific role
---------------
This script corresponds to the paragraph:
    \\paragraph{Synthetic IMU Measurements.}
Step 3 will load these files and evaluate the EKF under cross-model mismatch.

Noise model
-----------
For each noise realisation a small random rotation is left-multiplied onto
each ground-truth orientation frame:

    R_meas = R_noise @ R_true

where  R_noise = Rot.from_rotvec(xi),   xi ~ N(0, sigma^2 * I_3)
and    sigma   = meas_std_deg × pi/180

This matches the noise convention in sensor_placement_study_final.py (Part A)
exactly, ensuring apples-to-apples comparison with the matched-model baseline.

IMU-position mapping
--------------------
Sensor locations are specified as normalised arc-lengths s in [0, 1].
The Step-1 dataset stores orientations at ``num_pts`` uniformly spaced points
(indices 0 … num_pts-1).  Each requested position is mapped by:

    idx = round(s * (num_pts - 1))      ← nearest-neighbour

This is stable, reversible, and documented here so Step 3 can reproduce the
same mapping without reloading this script.

Output files (one per layout, in --save-dir)
--------------------------------------------
  kirchhoff_imu_2imu.npz
  kirchhoff_imu_3imu.npz

Each NPZ contains:
  case_id          (N,)                     GT case indices
  imu_positions    (n_imu,)                 normalised arc-length s values
  imu_arc_indices  (n_imu,)                 corresponding stored array indices
  R_true           (N, n_imu, 3, 3)         noise-free orientation matrices
  R_meas           (N, n_noise, n_imu, 3, 3) noisy orientation matrices
  meta             scalar                   JSON string (params, seed, date)

Usage
-----
  python generate_kirchhoff_imu_measurements.py
  python generate_kirchhoff_imu_measurements.py --show-summary
  python generate_kirchhoff_imu_measurements.py --plot-sanity
  python generate_kirchhoff_imu_measurements.py \\
      --layouts "0.50,1.00;0.25,0.50,1.00" --num-noise-realizations 10
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load the Step-1 ground-truth dataset.

    Returns
    -------
    orientations  : (N, num_pts, 9)  flattened R(s) row-major
    positions     : (N, num_pts, 3)  backbone centreline p(s)
    case_ids      : (N,)
    meta          : dict
    """
    data = np.load(npz_path, allow_pickle=True)
    orientations = data["orientations"]   # (N, num_pts, 9)
    positions    = data["positions"]      # (N, num_pts, 3)
    case_ids     = data["case_id"]        # (N,)

    meta: dict = {}
    if "meta" in data:
        try:
            meta = json.loads(str(data["meta"].item()))
        except Exception:
            pass

    json_path = npz_path.with_suffix(".json")
    if json_path.exists():
        try:
            meta.update(json.loads(json_path.read_text()))
        except Exception:
            pass

    return orientations, positions, case_ids, meta


# ---------------------------------------------------------------------------
# IMU-position → arc-length index mapping
# ---------------------------------------------------------------------------

def positions_to_indices(imu_positions: np.ndarray, num_pts: int) -> np.ndarray:
    """
    Map normalised arc-length positions s in [0, 1] to the nearest stored
    index using nearest-neighbour rounding:

        idx = clip( round(s * (num_pts - 1)), 0, num_pts - 1 )
    """
    raw = np.round(np.asarray(imu_positions, dtype=float) * (num_pts - 1))
    return np.clip(raw.astype(int), 0, num_pts - 1)


# ---------------------------------------------------------------------------
# Measurement generation
# ---------------------------------------------------------------------------

def generate_measurements(
    orientations: np.ndarray,
    imu_indices: np.ndarray,
    num_noise_realizations: int,
    sigma_rad: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate noise-free and noisy IMU orientation measurements.

    Noise model (left-multiply convention, matches sensor_placement_study_final.py):
        R_meas = R_noise @ R_true
        R_noise = Rot.from_rotvec(xi),  xi ~ N(0, sigma^2 * I_3)

    Parameters
    ----------
    orientations           : (N, num_pts, 9)  row-major flattened R(s)
    imu_indices            : (n_imu,)          arc-length indices
    num_noise_realizations : int
    sigma_rad              : float             noise std [rad]
    rng                    : np.random.Generator

    Returns
    -------
    R_true : (N, n_imu, 3, 3)
    R_meas : (N, num_noise_realizations, n_imu, 3, 3)
    """
    N        = orientations.shape[0]
    n_imu    = len(imu_indices)

    # Noise-free: slice selected indices and reshape to (N, n_imu, 3, 3)
    R_true = orientations[:, imu_indices, :].reshape(N, n_imu, 3, 3)

    # Noisy: draw one rotvec per (realization, IMU sensor), apply left-multiply
    R_meas = np.empty((N, num_noise_realizations, n_imu, 3, 3))
    for r in range(num_noise_realizations):
        # sample n_imu independent noise rotations (same across all cases,
        # as in the matched-model study — one noise draw per step per sensor)
        xi      = rng.standard_normal((n_imu, 3)) * sigma_rad   # (n_imu, 3)
        R_noise = Rot.from_rotvec(xi).as_matrix()               # (n_imu, 3, 3)
        # broadcast: R_meas[case, r, sensor] = R_noise[sensor] @ R_true[case, sensor]
        R_meas[:, r, :, :, :] = np.einsum(
            "jkl, ijlm -> ijkm", R_noise, R_true
        )

    return R_true, R_meas


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_layouts(layout_str: str) -> List[np.ndarray]:
    """
    Parse a semicolon-separated layout string.

    "0.50,1.00;0.25,0.50,1.00"  →  [array([0.50, 1.00]), array([0.25, 0.50, 1.00])]
    """
    layouts = []
    for group in layout_str.strip().split(";"):
        group = group.strip()
        if group:
            layouts.append(np.array([float(x) for x in group.split(",")]))
    return layouts


def layout_outpath(imu_positions: np.ndarray, save_dir: Path) -> Path:
    return save_dir / f"kirchhoff_imu_{len(imu_positions)}imu.npz"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(layout_results: list) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("  Step-2 IMU Measurement Summary")
    print(sep)
    for res in layout_results:
        pos_str = "{" + ", ".join(f"{s:.2f}" for s in res["imu_positions"]) + "}"
        act_str = "[" + ", ".join(f"{v:.4f}" for v in res["imu_actual_s"]) + "]"
        print(f"\n  Layout         : {pos_str}")
        print(f"  Arc indices    : {res['imu_indices'].tolist()}")
        print(f"  Actual s       : {act_str}")
        print(f"  R_true  shape  : {res['R_true'].shape}")
        print(f"  R_meas  shape  : {res['R_meas'].shape}")
        print(f"  Saved to       : {res['out_path']}")
    print(sep)


# ---------------------------------------------------------------------------
# Optional sanity-check visualisation
# ---------------------------------------------------------------------------

def plot_sanity(
    positions: np.ndarray,
    layout_results: list,
    case_idx: int = 0,
) -> None:
    """
    Quick sanity-check figure.

    For one representative case, plots the backbone with true IMU frames
    (solid RGB arrows) and the first noisy realisation (transparent arrows).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    COLORS     = ("r", "g", "b")
    FRAME_SCALE = 0.008

    def draw_frame(ax, origin, R, alpha=1.0, lw=1.2):
        for ci, col in enumerate(COLORS):
            ax.quiver(*origin, *(FRAME_SCALE * R[:, ci]),
                      color=col, linewidth=lw, alpha=alpha,
                      arrow_length_ratio=0.3)

    n_layouts = len(layout_results)
    fig = plt.figure(figsize=(5 * n_layouts, 5))

    p = positions[case_idx]   # (num_pts, 3)

    for k, res in enumerate(layout_results):
        ax = fig.add_subplot(1, n_layouts, k + 1, projection="3d")
        ax.plot(p[:, 0], p[:, 1], p[:, 2], "k-", lw=1.0, alpha=0.6)

        R_true  = res["R_true"][case_idx]      # (n_imu, 3, 3)
        R_noisy = res["R_meas"][case_idx, 0]   # first realization (n_imu, 3, 3)

        for j, idx in enumerate(res["imu_indices"]):
            origin = p[idx]
            draw_frame(ax, origin, R_true[j],  alpha=1.0,  lw=1.4)
            draw_frame(ax, origin, R_noisy[j], alpha=0.35, lw=0.8)

        pos_str = "{" + ", ".join(f"{s:.2f}" for s in res["imu_positions"]) + "}"
        ax.set_title(f"Layout {pos_str}\nsolid=true  transparent=noisy",
                     fontsize=8)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.set_zlabel("z [m]", fontsize=8)
        ax.set_box_aspect([1, 1, 1])

    plt.suptitle(f"Sanity check — case {case_idx}", fontsize=10)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: generate synthetic IMU measurements from Kirchhoff-rod GT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, default="gt_data/kirchhoff_gt_dataset.npz",
        help="path to Step-1 ground-truth NPZ file",
    )
    parser.add_argument(
        "--save-dir", type=str, default="gt_data",
        help="directory to write measurement NPZ files",
    )
    parser.add_argument(
        "--layouts", type=str, default="0.50,1.00;0.25,0.50,1.00",
        help="semicolon-separated layouts; each layout is comma-separated "
             "normalised arc-length positions in [0,1]  "
             "e.g. \"0.50,1.00;0.25,0.50,1.00\"",
    )
    parser.add_argument(
        "--num-noise-realizations", type=int, default=5,
        help="independent IMU-noise realisations per ground-truth case",
    )
    parser.add_argument(
        "--meas-std-deg", type=float, default=0.5,
        help="IMU orientation noise std [deg]",
    )
    parser.add_argument(
        "--seed", type=int, default=1234,
        help="NumPy RNG seed for reproducibility",
    )
    parser.add_argument(
        "--show-summary", action="store_true", default=True,
        help="print array-shape summary after generation (default on)",
    )
    parser.add_argument(
        "--no-show-summary", dest="show_summary", action="store_false",
    )
    parser.add_argument(
        "--plot-sanity", action="store_true",
        help="show quick sanity-check figure for case 0",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    npz_path = Path(args.input)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Step-1 dataset not found: {npz_path}\n"
            "Run generate_kirchhoff_gt_dataset.py first."
        )

    print(f"Loading dataset : {npz_path}")
    orientations, positions, case_ids, gt_meta = load_dataset(npz_path)
    N, num_pts, _ = orientations.shape
    print(f"  {N} cases  ×  {num_pts} arc-length points")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    sigma_rad = args.meas_std_deg * np.pi / 180.0
    rng       = np.random.default_rng(args.seed)
    layouts   = parse_layouts(args.layouts)
    save_dir  = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nNoise model    : R_meas = R_noise @ R_true  (left-multiply)")
    print(f"Noise sigma    : {args.meas_std_deg} deg  =  {sigma_rad:.5f} rad")
    print(f"Realisations   : {args.num_noise_realizations} per case")
    print(f"RNG seed       : {args.seed}")

    # ------------------------------------------------------------------
    # Generate and save one file per layout
    # ------------------------------------------------------------------
    layout_results = []

    for imu_positions in layouts:
        imu_indices = positions_to_indices(imu_positions, num_pts)
        imu_actual_s = imu_indices / (num_pts - 1)

        print(f"\nLayout  s = "
              "{" + ", ".join(f"{v:.2f}" for v in imu_positions) + "}")
        print(f"  → indices   : {imu_indices.tolist()}")
        print(f"  → actual s  : {[round(float(v), 4) for v in imu_actual_s]}")

        R_true, R_meas = generate_measurements(
            orientations           = orientations,
            imu_indices            = imu_indices,
            num_noise_realizations = args.num_noise_realizations,
            sigma_rad              = sigma_rad,
            rng                    = rng,
        )

        out_path = layout_outpath(imu_positions, save_dir)

        layout_meta = {
            "description"           : "Kirchhoff-rod synthetic IMU measurements (Step 2)",
            "source_dataset"        : str(npz_path),
            "n_cases"               : int(N),
            "n_imu"                 : int(len(imu_positions)),
            "imu_positions_norm"    : imu_positions.tolist(),
            "imu_arc_indices"       : imu_indices.tolist(),
            "imu_actual_s"          : imu_actual_s.tolist(),
            "num_pts_per_case"      : int(num_pts),
            "num_noise_realizations": int(args.num_noise_realizations),
            "meas_std_deg"          : float(args.meas_std_deg),
            "meas_std_rad"          : float(sigma_rad),
            "noise_convention"      : "R_meas = R_noise @ R_true  (left-multiply); "
                                      "R_noise = Rot.from_rotvec(xi), xi ~ N(0, sigma^2 I_3)",
            "imu_mapping"           : "nearest-neighbour: idx = round(s * (num_pts-1))",
            "rng_seed"              : int(args.seed),
            "date"                  : datetime.datetime.now().isoformat(timespec="seconds"),
            "array_layout"          : {
                "case_id"        : "(N,)",
                "imu_positions"  : "(n_imu,)  normalised arc-length s",
                "imu_arc_indices": "(n_imu,)  stored array indices",
                "R_true"         : "(N, n_imu, 3, 3)  noise-free",
                "R_meas"         : "(N, n_noise, n_imu, 3, 3)  noisy",
            },
        }

        np.savez_compressed(
            out_path,
            case_id         = case_ids,
            imu_positions   = imu_positions,
            imu_arc_indices = imu_indices,
            R_true          = R_true,
            R_meas          = R_meas,
            meta            = json.dumps(layout_meta),
        )
        print(f"  Saved → {out_path}")

        layout_results.append({
            "imu_positions": imu_positions,
            "imu_indices"  : imu_indices,
            "imu_actual_s" : imu_actual_s,
            "R_true"       : R_true,
            "R_meas"       : R_meas,
            "out_path"     : out_path,
        })

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if args.show_summary:
        print_summary(layout_results)

    # ------------------------------------------------------------------
    # Sanity-check plot
    # ------------------------------------------------------------------
    if args.plot_sanity:
        plot_sanity(positions, layout_results, case_idx=0)


if __name__ == "__main__":
    main()
