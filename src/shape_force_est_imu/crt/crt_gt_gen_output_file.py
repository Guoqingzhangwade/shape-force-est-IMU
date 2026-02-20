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


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def sample_tensions(n_tendon: int = 4, τ_max: float = 4.0,
                    active_idx: list[int] | None = None) -> np.ndarray:
    τ = np.zeros(n_tendon)
    if active_idx is None:
        k_active = np.random.choice([1, 2])
        start    = np.random.randint(0, n_tendon)
        idx      = [start] if k_active == 1 else [start, (start + 1) % n_tendon]
    else:
        idx = active_idx
    τ[idx]   = np.random.uniform(1.0, τ_max, size=len(idx))
    return τ


def sample_tip_wrench(mode: str = "full", plane: str = "xz",
                      force_range: float = 0.4, moment_range: float = 0.04
                      ) -> tuple[np.ndarray, np.ndarray]:
    f_ext = np.random.uniform(-force_range, force_range, 3)
    l_ext = np.random.uniform(-moment_range, moment_range, 3)
    if mode == "force-only":
        l_ext = np.zeros(3)
        if plane == "xz":
            f_ext[1] = 0.0
        elif plane == "yz":
            f_ext[0] = 0.0
        elif plane == "xy":
            f_ext[2] = 0.0
    elif mode == "moment-only":
        f_ext = np.zeros(3)
    elif mode == "zero":
        f_ext = np.zeros(3)
        l_ext = np.zeros(3)
    return f_ext, l_ext


def parse_vec3(arg: str) -> np.ndarray:
    parts = [p.strip() for p in arg.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {arg}")
    return np.array([float(p) for p in parts], dtype=float)


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

    # ---- simple example: orientation as 3×3 R(s) in rows 3:12 ----
    # (comment out if your model already gives 4×4 frames)
    R_flat = traj_y[3:12, :].T.reshape(n_pts, 3, 3)
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
    parser.add_argument("--n",          type=int,   default=10,
                        help="number of valid samples to generate")
    parser.add_argument("--outfile",    type=str,   default="tdcr_gt_samples_10.npz")
    parser.add_argument("--seed",       type=int,   default=22,
                        help="set RNG seed for repeatability")
    parser.add_argument("--wrench-mode", type=str, default="full",
                        choices=["full", "force-only", "moment-only", "zero"],
                        help="sample wrench type")
    parser.add_argument("--wrench-plane", type=str, default="xz",
                        choices=["xz", "yz", "xy"],
                        help="plane for in-plane force (force-only mode)")
    parser.add_argument("--force-range", type=float, default=0.4,
                        help="uniform range for random forces (N)")
    parser.add_argument("--moment-range", type=float, default=0.04,
                        help="uniform range for random moments (Nm)")
    parser.add_argument("--use-scaled-wrench", action="store_true",
                        help="set force/moment ranges from c_F,c_M and EI/L scales")
    parser.add_argument("--c-f", type=float, default=1.0,
                        help="dimensionless force scale factor")
    parser.add_argument("--c-m", type=float, default=1.0,
                        help="dimensionless moment scale factor")
    parser.add_argument("--active-cables", type=str, default=None,
                        help="comma-separated tendon indices to activate (e.g., 0,2)")
    parser.add_argument("--inplane", action="store_true",
                        help="force-only in-plane wrench and fixed active cables")
    parser.add_argument("--inextensible", action="store_true", default=True,
                        help="enforce v(s)=e3 (no shear/stretch) in rod model")
    parser.add_argument("--length", type=float, default=0.10,
                        help="backbone length (m)")
    parser.add_argument("--youngs-modulus", type=float, default=60e9,
                        help="Young's modulus (Pa)")
    parser.add_argument("--backbone-radius", type=float, default=5e-4,
                        help="backbone radius (m)")
    parser.add_argument("--tendon-offset", type=float, default=0.008,
                        help="tendon radial offset from backbone center (m)")
    parser.add_argument("--tendon-radius", type=float, default=None,
                        help="deprecated alias for --tendon-offset")
    parser.add_argument("--num-disks", type=int, default=40,
                        help="number of disks along backbone")
    parser.add_argument("--tau-max", type=float, default=4.0,
                        help="max cable tension for sampling (N)")
    parser.add_argument("--max-tries", type=int, default=1000,
                        help="max attempts before giving up")
    parser.add_argument("--solver-max-nfev", type=int, default=500,
                        help="max function evals for shooting solver")
    parser.add_argument("--solver-xtol", type=float, default=1e-10,
                        help="xtol for shooting solver")
    parser.add_argument("--solver-ftol", type=float, default=1e-10,
                        help="ftol for shooting solver")
    parser.add_argument("--solver-gtol", type=float, default=1e-10,
                        help="gtol for shooting solver")
    parser.add_argument("--force", type=str, default=None,
                        help="fixed tip force as 'fx,fy,fz' in world frame (N)")
    parser.add_argument("--moment", type=str, default=None,
                        help="fixed tip moment as 'mx,my,mz' in world frame (Nm)")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    from cosserat_rod_model import CosseratRodModel
    tendon_offset = args.tendon_offset
    if args.tendon_radius is not None:
        tendon_offset = args.tendon_radius
    angles = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    tendon_routing = [
        np.array([tendon_offset * np.cos(a),
                  tendon_offset * np.sin(a), 0.0])
        for a in angles
    ]
    rod = CosseratRodModel(length=args.length,
                           backbone_radius=args.backbone_radius,
                           youngs_modulus=args.youngs_modulus,
                           tendon_routing=tendon_routing,
                           num_disks=args.num_disks,
                           inextensible=args.inextensible)

    force_range = args.force_range
    moment_range = args.moment_range
    if args.use_scaled_wrench:
        I = np.pi * (args.backbone_radius ** 4) / 4.0
        EI = args.youngs_modulus * I
        F0 = EI / (args.length ** 2)
        M0 = EI / args.length
        force_range = args.c_f * F0
        moment_range = args.c_m * M0
    s_eval = np.linspace(0.0, rod.L, rod.N)

    tau_list, f_list, l_list, T_list = [], [], [], []

    pbar = tqdm(total=args.n, desc="Generating samples", unit="sample")
    tries = 0
    while len(tau_list) < args.n:
        tries += 1
        if args.max_tries > 0 and tries > args.max_tries:
            raise RuntimeError(f"Exceeded max tries ({args.max_tries}). Generated {len(tau_list)} samples.")
        if args.force is not None or args.moment is not None:
            f_ext = parse_vec3(args.force) if args.force is not None else np.zeros(3)
            l_ext = parse_vec3(args.moment) if args.moment is not None else np.zeros(3)
            active_idx = None
            if args.active_cables is not None:
                active_idx = [int(x) for x in args.active_cables.split(",")]
            τ = sample_tensions(n_tendon=rod.n_tendon, τ_max=args.tau_max,
                                active_idx=active_idx)
        elif args.inplane:
            if args.active_cables is None:
                raise ValueError("--inplane requires --active-cables")
            active_idx = [int(x) for x in args.active_cables.split(",")]
            τ = sample_tensions(n_tendon=rod.n_tendon, τ_max=args.tau_max,
                                active_idx=active_idx)
            f_ext, l_ext = sample_tip_wrench(mode="force-only",
                                             plane=args.wrench_plane,
                                             force_range=force_range,
                                             moment_range=moment_range)
        else:
            τ = sample_tensions(n_tendon=rod.n_tendon, τ_max=args.tau_max)
            f_ext, l_ext = sample_tip_wrench(mode=args.wrench_mode,
                                             plane=args.wrench_plane,
                                             force_range=force_range,
                                             moment_range=moment_range)

        try:
            solver_opts = {
                "max_nfev": args.solver_max_nfev,
                "xtol": args.solver_xtol,
                "ftol": args.solver_ftol,
                "gtol": args.solver_gtol,
            }
            _, traj = rod.forward_kinematics(
                tau=τ,
                f_ext=f_ext,
                l_ext=l_ext,
                return_states=True,           # we need raw states
                s_eval=s_eval,
                solver_opts=solver_opts
                )
            T_all = states_to_frames(traj.y)  # (n_disks,4,4)

            tau_list.append(τ)
            f_list.append(f_ext)
            l_list.append(l_ext)
            T_list.append(T_all)

            pbar.update()

        except RuntimeError as exc:
            if tries % 50 == 0:
                print(f"Attempt {tries}: shooting failed ({exc})")
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
        "youngs_modulus": float(args.youngs_modulus),
        "backbone_radius": float(args.backbone_radius),
        "tendon_offset": float(tendon_offset),
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
