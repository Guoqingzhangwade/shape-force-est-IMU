#!/usr/bin/env python3
"""
Generate a Monte-Carlo ground-truth data-set for a single-segment TDCR.

Outputs
-------
Compressed NPZ containing
    tau      : (N, 4)          cable tensions  [N]
    f_ext    : (N, 3)          tip forces      [N]
    l_ext    : (N, 3)          tip moments     [N·m]
    T_all    : (N, n_disks, 4, 4) homogeneous transforms of every disk
    meta     : dict            misc meta-data (JSON-serialisable)

Author: Guoqing Zhang, 2025-05-20
"""
from __future__ import annotations
import argparse, json, os, subprocess, datetime
import numpy as np
from tqdm import tqdm                           # progress-bar
from cosserat_rod_model import CosseratRodModel


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def sample_tensions(n_tendon: int = 4, τ_max: float = 4.0) -> np.ndarray:
    τ = np.zeros(n_tendon)
    k_active = np.random.choice([1, 2])
    start    = np.random.randint(0, n_tendon)
    idx      = [start] if k_active == 1 else [start, (start + 1) % n_tendon]
    τ[idx]   = np.random.uniform(1.0, τ_max, size=len(idx))
    return τ


def sample_tip_wrench() -> tuple[np.ndarray, np.ndarray]:
    f_ext = np.random.uniform(-0.4, 0.4, 3)
    l_ext = np.random.uniform(-0.04, 0.04, 3)
    return f_ext, l_ext


def states_to_frames(traj_y: np.ndarray) -> np.ndarray:
    """
    Convert the state vector returned by CosseratRodModel into
    4×4 frames along the backbone.

    Expected row layout in `traj_y`  (y.shape = (state_dim , n_pts)):
        0:3  -> position  p(s)
        3:6  -> body-frame tangent basis e1, e2, e3  (or however your
                 model stores orientation).
    Modify if your model packs orientation differently.
    """
    n_pts = traj_y.shape[1]
    T = np.eye(4)[None, ...].repeat(n_pts, axis=0)          # (n_pts, 4, 4)
    T[:, :3, 3] = traj_y[0:3, :].T

    # ---- simple example: orientation as 3×3 R(s) in rows 6:15 ----
    # (comment out if your model already gives 4×4 frames)
    if traj_y.shape[0] >= 15:
        R_flat = traj_y[6:15, :].T.reshape(n_pts, 3, 3)
        T[:, :3, :3] = R_flat

    return T


def git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int,   default=100,
                        help="number of valid samples to generate")
    parser.add_argument("--outfile",    type=str,   default="tdcr_gt_samples.npz")
    parser.add_argument("--seed",       type=int,   default=None,
                        help="set RNG seed for repeatability")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    rod = CosseratRodModel(length=0.10, backbone_radius=5e-4, num_disks=40)
    s_eval = np.linspace(0.0, rod.L, rod.N)

    tau_list, f_list, l_list, T_list = [], [], [], []

    pbar = tqdm(total=args.n, desc="Generating samples", unit="sample")
    while len(tau_list) < args.n:
        τ      = sample_tensions(n_tendon=rod.n_tendon, τ_max=4.0)
        f_ext, l_ext = sample_tip_wrench()

        try:
            _, traj = rod.forward_kinematics(
                tau=τ,
                f_ext=f_ext,
                l_ext=l_ext,
                return_states=True,           # we need raw states
                s_eval=s_eval
                )
            T_all = states_to_frames(traj.y)  # (n_disks,4,4)

            tau_list.append(τ)
            f_list.append(f_ext)
            l_list.append(l_ext)
            T_list.append(T_all)

            pbar.update()

        except RuntimeError:
            # non-converged shooting -> discard and resample
            continue
    pbar.close()

    tau_arr = np.vstack(tau_list)                      # (N,4)
    f_arr   = np.vstack(f_list)                        # (N,3)
    l_arr   = np.vstack(l_list)                        # (N,3)
    T_arr   = np.stack(T_list, axis=0)                 # (N,n_disks,4,4)

    meta = {
        "n_samples"   : int(args.n),
        "n_disks"     : int(rod.N),
        "length_m"    : float(rod.L),
        "date"        : datetime.datetime.now().isoformat(timespec="seconds"),
        "git_hash"    : git_hash(),
        "rng_seed"    : args.seed,
    }

    np.savez_compressed(args.outfile,
                        tau=tau_arr,
                        f_ext=f_arr,
                        l_ext=l_arr,
                        T=T_arr,
                        meta=json.dumps(meta))

    print(f"Saved ground-truth data-set to “{args.outfile}”.\n"
          f"  ↑ tau     shape = {tau_arr.shape}\n"
          f"  ↑ T poses shape = {T_arr.shape}")


if __name__ == "__main__":
    main()
