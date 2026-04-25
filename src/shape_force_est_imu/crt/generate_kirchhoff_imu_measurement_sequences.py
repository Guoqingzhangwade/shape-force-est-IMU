#!/usr/bin/env python3
"""
Cross-model robustness study - Step 2b: sequence-based synthetic IMU measurements.

This script intentionally leaves the original static-frame Step-2 workflow
untouched:

  - Original Step 2 (`generate_kirchhoff_imu_measurements.py`)
    saves one noisy IMU frame per realization:
        R_meas.shape == (N, n_noise, n_imu, 3, 3)
    The Step-3 evaluator then reuses that same frame at every EKF step.

  - New Step 2b (this script)
    keeps the Kirchhoff-rod ground-truth shape fixed within each case, but
    generates a fresh noisy IMU frame at every EKF step:
        R_meas_seq.shape == (N, n_noise, n_steps, n_imu, 3, 3)

Scientific intent
-----------------
For each static Kirchhoff-rod ground-truth case, the true shape does not
change, but each EKF step receives a newly perturbed sparse IMU observation.
This makes the cross-model setup conceptually aligned with the matched-model
studies while preserving the same SO(3) noise convention and the same
nearest-neighbour arc-length mapping used by the existing workflow.

Noise model
-----------
For every (case, realization, step, IMU):

    R_meas = R_noise @ R_true

where

    R_noise = Rot.from_rotvec(xi),   xi ~ N(0, sigma^2 I_3)

with sigma = meas_std_deg * pi / 180.

Output files (one per layout, in --save-dir)
--------------------------------------------
  kirchhoff_imu_seq_2imu.npz
  kirchhoff_imu_seq_3imu.npz

Each NPZ contains at least:
  case_id          (N,)
  imu_positions    (n_imu,)
  imu_arc_indices  (n_imu,)
  R_true           (N, n_imu, 3, 3)
  R_meas_seq       (N, n_noise, n_steps, n_imu, 3, 3)
  meta             JSON string
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from generate_kirchhoff_imu_measurements import (
    load_dataset,
    parse_layouts,
    positions_to_indices,
)


def resolve_cli_path(path_str: str, script_dir: Path, must_exist: bool) -> Path:
    """
    Resolve a CLI path robustly for both:
      1. running from this directory, and
      2. running from the repository root via `python path/to/script.py`.
    """
    raw = Path(path_str)
    if raw.is_absolute():
        path = raw
    elif raw.exists():
        path = raw.resolve()
    else:
        path = (script_dir / raw).resolve()

    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def build_noise_seed_bank(
    num_cases: int,
    num_noise_realizations: int,
    seed_base: int,
) -> np.ndarray:
    """
    Deterministic seed table with one seed per (case, realization).

    This mirrors the matched-model seed-bank pattern so that every run is
    reproducible and the same case/realization pair can be regenerated
    identically later.
    """
    seeds = np.arange(num_cases * num_noise_realizations, dtype=np.int64)
    return seeds.reshape(num_cases, num_noise_realizations) + int(seed_base)


def generate_measurement_sequences(
    orientations: np.ndarray,
    imu_indices: np.ndarray,
    noise_seed_bank: np.ndarray,
    steps: int,
    sigma_rad: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate noise-free IMU frames and noisy sequence measurements.

    Parameters
    ----------
    orientations     : (N, num_pts, 9) flattened row-major R(s)
    imu_indices      : (n_imu,)
    noise_seed_bank  : (N, n_noise) deterministic integer seeds
    steps            : int
    sigma_rad        : float

    Returns
    -------
    R_true     : (N, n_imu, 3, 3)
    R_meas_seq : (N, n_noise, n_steps, n_imu, 3, 3)
    """
    num_cases, _, _ = orientations.shape
    num_noise_realizations = noise_seed_bank.shape[1]
    n_imu = len(imu_indices)

    R_true = orientations[:, imu_indices, :].reshape(num_cases, n_imu, 3, 3)
    R_meas_seq = np.empty(
        (num_cases, num_noise_realizations, steps, n_imu, 3, 3),
        dtype=float,
    )

    for case_idx in range(num_cases):
        R_true_case = R_true[case_idx]
        for noise_idx in range(num_noise_realizations):
            rng = np.random.default_rng(int(noise_seed_bank[case_idx, noise_idx]))
            xi = rng.standard_normal((steps, n_imu, 3)) * sigma_rad
            R_noise = Rot.from_rotvec(xi.reshape(-1, 3)).as_matrix()
            R_noise = R_noise.reshape(steps, n_imu, 3, 3)
            R_meas_seq[case_idx, noise_idx] = (
                R_noise @ R_true_case[np.newaxis, :, :, :]
            )

    return R_true, R_meas_seq


def layout_outpath(imu_positions: np.ndarray, save_dir: Path) -> Path:
    return save_dir / f"kirchhoff_imu_seq_{len(imu_positions)}imu.npz"


def print_summary(layout_results: List[Dict]) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  Step-2b IMU Measurement Sequence Summary")
    print(sep)
    for res in layout_results:
        pos_str = "{" + ", ".join(f"{s:.2f}" for s in res["imu_positions"]) + "}"
        act_str = "[" + ", ".join(f"{v:.4f}" for v in res["imu_actual_s"]) + "]"
        print(f"\n  Layout            : {pos_str}")
        print(f"  Arc indices       : {res['imu_indices'].tolist()}")
        print(f"  Actual s          : {act_str}")
        print(f"  R_true shape      : {res['R_true'].shape}")
        print(f"  R_meas_seq shape  : {res['R_meas_seq'].shape}")
        print(f"  Saved to          : {res['out_path']}")
    print(sep)


def plot_sanity_sequence(
    positions: np.ndarray,
    layout_results: List[Dict],
    case_idx: int = 0,
    realization_idx: int = 0,
    num_steps_to_show: int = 3,
) -> None:
    """
    Plot one fixed ground-truth case with a few time-varying noisy IMU frames.

    The backbone stays static; only the noisy IMU frames change across steps.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    colors = ("r", "g", "b")
    frame_scale = 0.008
    step_alphas = np.linspace(0.25, 0.65, max(num_steps_to_show, 1))

    def draw_frame(ax, origin, R, alpha=1.0, lw=1.0):
        for axis_idx, color in enumerate(colors):
            ax.quiver(
                *origin,
                *(frame_scale * R[:, axis_idx]),
                color=color,
                linewidth=lw,
                alpha=alpha,
                arrow_length_ratio=0.3,
            )

    fig = plt.figure(figsize=(5 * len(layout_results), 5))
    p_case = positions[case_idx]

    for plot_idx, res in enumerate(layout_results):
        ax = fig.add_subplot(1, len(layout_results), plot_idx + 1, projection="3d")
        ax.plot(p_case[:, 0], p_case[:, 1], p_case[:, 2], "k-", lw=1.0, alpha=0.6)

        R_true = res["R_true"][case_idx]
        R_seq = res["R_meas_seq"][case_idx, realization_idx]
        steps_to_show = min(num_steps_to_show, R_seq.shape[0])

        for sensor_idx, arc_idx in enumerate(res["imu_indices"]):
            origin = p_case[arc_idx]
            draw_frame(ax, origin, R_true[sensor_idx], alpha=1.0, lw=1.4)
            for step_idx in range(steps_to_show):
                draw_frame(
                    ax,
                    origin,
                    R_seq[step_idx, sensor_idx],
                    alpha=float(step_alphas[step_idx]),
                    lw=0.8,
                )

        pos_str = "{" + ", ".join(f"{s:.2f}" for s in res["imu_positions"]) + "}"
        ax.set_title(
            f"Layout {pos_str}\nsolid=true  faint=noisy sequence",
            fontsize=8,
        )
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.set_zlabel("z [m]", fontsize=8)
        ax.set_box_aspect([1, 1, 1])

    plt.suptitle(
        f"Sequence sanity check - case {case_idx}, realization {realization_idx}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2b: generate sequence-based IMU measurements from Kirchhoff GT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="gt_data/kirchhoff_gt_dataset.npz",
        help="path to Step-1 ground-truth NPZ file",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="gt_data_seq",
        help="directory to write sequence measurement NPZ files",
    )
    parser.add_argument(
        "--layouts",
        type=str,
        default="0.50,1.00;0.25,0.50,1.00",
        help=(
            "semicolon-separated layouts; each layout is comma-separated "
            "normalized arc-length positions in [0,1]"
        ),
    )
    parser.add_argument(
        "--num-noise-realizations",
        type=int,
        default=5,
        help="independent measurement sequences per ground-truth case",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="number of EKF measurement steps stored per realization",
    )
    parser.add_argument(
        "--meas-std-deg",
        type=float,
        default=0.5,
        help="IMU orientation noise std [deg]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="base RNG seed used to build the deterministic seed bank",
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        default=True,
        help="print array-shape summary after generation (default on)",
    )
    parser.add_argument(
        "--no-show-summary",
        dest="show_summary",
        action="store_false",
    )
    parser.add_argument(
        "--plot-sanity",
        action="store_true",
        help="show a quick sanity plot for case 0",
    )
    parser.add_argument(
        "--sanity-case",
        type=int,
        default=0,
        help="case index for the optional sanity plot",
    )
    parser.add_argument(
        "--sanity-realization",
        type=int,
        default=0,
        help="noise realization index for the optional sanity plot",
    )
    parser.add_argument(
        "--sanity-steps",
        type=int,
        default=3,
        help="number of noisy time steps to overlay in the sanity plot",
    )
    args = parser.parse_args()

    if args.num_noise_realizations <= 0:
        raise ValueError("--num-noise-realizations must be positive.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.meas_std_deg <= 0.0:
        raise ValueError("--meas-std-deg must be positive.")

    script_dir = Path(__file__).resolve().parent
    npz_path = resolve_cli_path(args.input, script_dir, must_exist=True)
    save_dir = resolve_cli_path(args.save_dir, script_dir, must_exist=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset : {npz_path}")
    orientations, positions, case_ids, gt_meta = load_dataset(npz_path)
    num_cases, num_pts, _ = orientations.shape
    sigma_rad = np.deg2rad(args.meas_std_deg)
    layouts = parse_layouts(args.layouts)
    noise_seed_bank = build_noise_seed_bank(
        num_cases=num_cases,
        num_noise_realizations=args.num_noise_realizations,
        seed_base=args.seed,
    )

    print(f"  {num_cases} cases x {num_pts} arc-length points")
    print("\nSequence workflow : static true Kirchhoff shape, fresh IMU noise each EKF step")
    print("Noise convention  : R_meas = R_noise @ R_true")
    print(f"Noise sigma       : {args.meas_std_deg:.4f} deg = {sigma_rad:.6f} rad")
    print(f"Realizations      : {args.num_noise_realizations} per case")
    print(f"EKF steps saved   : {args.steps}")
    print(f"Seed base         : {args.seed}")

    layout_results: List[Dict] = []

    for imu_positions in layouts:
        imu_indices = positions_to_indices(imu_positions, num_pts)
        imu_actual_s = imu_indices / (num_pts - 1)

        print(
            "\nLayout s = {"
            + ", ".join(f"{value:.2f}" for value in imu_positions)
            + "}"
        )
        print(f"  -> indices  : {imu_indices.tolist()}")
        print(f"  -> actual s : {[round(float(v), 4) for v in imu_actual_s]}")

        R_true, R_meas_seq = generate_measurement_sequences(
            orientations=orientations,
            imu_indices=imu_indices,
            noise_seed_bank=noise_seed_bank,
            steps=args.steps,
            sigma_rad=sigma_rad,
        )

        out_path = layout_outpath(imu_positions, save_dir)
        layout_meta = {
            "description": "Kirchhoff-rod sequence-based IMU measurements (Step 2b)",
            "method_note": (
                "Original cross-model Step 2 stored one static noisy IMU frame per "
                "realization. This sequence-based Step 2b keeps the ground-truth "
                "shape fixed but resamples fresh SO(3) IMU noise at every EKF step."
            ),
            "source_dataset": str(npz_path),
            "source_dataset_meta": gt_meta,
            "n_cases": int(num_cases),
            "n_imu": int(len(imu_positions)),
            "imu_positions_norm": imu_positions.tolist(),
            "imu_arc_indices": imu_indices.tolist(),
            "imu_actual_s": imu_actual_s.tolist(),
            "num_pts_per_case": int(num_pts),
            "num_noise_realizations": int(args.num_noise_realizations),
            "num_steps": int(args.steps),
            "meas_std_deg": float(args.meas_std_deg),
            "meas_std_rad": float(sigma_rad),
            "rng_seed_base": int(args.seed),
            "seed_rule": "noise_seed_bank[case, realization] = seed_base + linear index",
            "nearest_neighbour_mapping_rule": "idx = round(s * (num_pts - 1))",
            "noise_convention": (
                "R_meas = R_noise @ R_true; "
                "R_noise = Rot.from_rotvec(xi), xi ~ N(0, sigma^2 I_3)"
            ),
            "array_layout": {
                "case_id": "(N,)",
                "imu_positions": "(n_imu,) normalized arc-length s",
                "imu_arc_indices": "(n_imu,) stored array indices",
                "noise_seed_bank": "(N, n_noise) deterministic integer seeds",
                "R_true": "(N, n_imu, 3, 3) static noise-free frames",
                "R_meas_seq": "(N, n_noise, n_steps, n_imu, 3, 3) noisy measurement sequences",
            },
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        }

        np.savez_compressed(
            out_path,
            case_id=case_ids,
            imu_positions=imu_positions,
            imu_arc_indices=imu_indices,
            noise_seed_bank=noise_seed_bank,
            R_true=R_true,
            R_meas_seq=R_meas_seq,
            meta=json.dumps(layout_meta),
        )
        print(f"  Saved -> {out_path}")

        layout_results.append(
            {
                "imu_positions": imu_positions,
                "imu_indices": imu_indices,
                "imu_actual_s": imu_actual_s,
                "R_true": R_true,
                "R_meas_seq": R_meas_seq,
                "out_path": out_path,
            }
        )

    if args.show_summary:
        print_summary(layout_results)

    if args.plot_sanity:
        plot_sanity_sequence(
            positions=positions,
            layout_results=layout_results,
            case_idx=args.sanity_case,
            realization_idx=args.sanity_realization,
            num_steps_to_show=args.sanity_steps,
        )

    print("\nWorkflow distinction")
    print("  Original static-frame workflow : one noisy IMU frame per realization.")
    print("  New sequence workflow          : one noisy IMU sequence per realization.")


if __name__ == "__main__":
    main()
