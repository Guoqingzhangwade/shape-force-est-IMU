#!/usr/bin/env python3
"""
Batch-sampling demo for a single-segment TDCR (Cosserat rod statics).

* 4 tendons, tensions τ_i ∈ [0, 4] N
* Only one tendon or two contiguous tendons are active per sample
  (pairs: 0-1, 1-2, 2-3, 3-0 in zero-based indexing).
* Random tip wrench:
      f_ext  ∈ [-0.4 N , 0.4 N]  (each component)
      l_ext  ∈ [-0.04 Nm , 0.04 Nm]
* 50 valid samples are collected; if the static solver fails
  (rare), the sample is discarded and re-drawn.
* All shapes are plotted in a single 3-D figure.

Author: Guoqing Zhang, 2025-05-19
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (activates 3-D)
from cosserat_rod_model import CosseratRodModel

# ---------------------------------------------------------------------------
# Helper-functions -----------------------------------------------------------
# ---------------------------------------------------------------------------

def sample_tensions(n_tendon: int = 4,
                    τ_max: float = 5.0) -> np.ndarray:
    """
    Return a tension vector τ ∈ ℝ⁴ obeying:
      • Either one tendon OR two contiguous tendons are active.
      • Each active τᵢ is uniform in [0, τ_max] (N).
    Tendon indices are 0 … 3 around the cross-section.
    """
    τ = np.zeros(n_tendon)
    k_active = np.random.choice([1, 2])         # choose 1 or 2 tendons
    start    = np.random.randint(0, n_tendon)   # random starting index
    idx = [start] if k_active == 1 else [start, (start + 1) % n_tendon]
    τ[idx] = np.random.uniform(1.0, τ_max, size=len(idx))
    return τ


def sample_tip_wrench() -> tuple[np.ndarray, np.ndarray]:
    """
    Tip force  f_ext  ~ U([-0.4, +0.4] N)^3
    Tip moment l_ext  ~ U([-0.04, +0.04] Nm)^3
    """
    f_ext = np.random.uniform(-0.4, 0.4, 3)
    l_ext = np.random.uniform(-0.04, 0.04, 3)
    return f_ext, l_ext


# ---------------------------------------------------------------------------
# Main routine ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    # --- build a rod model (adjust parameters if needed) -------------------
    rod = CosseratRodModel(
        length=0.10,          # [m]
        backbone_radius=5e-4, # [m]
        num_disks=40          # finer discretisation for smoother plots
    )

    shapes_xyz: list[np.ndarray] = []
    tip_frames: list[np.ndarray] = []
    n_target = 5

    while len(shapes_xyz) < n_target:
        τ = sample_tensions(n_tendon=rod.n_tendon, τ_max=4.0)
        f_ext, l_ext = sample_tip_wrench()

        try:
            _, traj = rod.forward_kinematics(
                tau=τ,
                f_ext=f_ext,
                l_ext=l_ext,
                return_states=True
            )
            # Centre-line positions p(s)
            p = traj.y[0:3, :].T   # shape (N_pts, 3)
            shapes_xyz.append(p)

            # Extract tip frame T_tip
            yL = traj.y[:, -1]
            T_tip = np.eye(4)
            T_tip[0:3, 0:3] = yL[3:12].reshape(3, 3)
            T_tip[0:3, 3] = yL[0:3]
            tip_frames.append(T_tip)

        except RuntimeError:
            continue

    print(f"Generated {len(shapes_xyz)} valid configurations.")

    # --- plot all shapes and tip frames in one figure ----------------------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for p, T in zip(shapes_xyz, tip_frames):
        ax.plot(p[:, 0], p[:, 1], p[:, 2], lw=0.8, alpha=0.75)

        # Plot tip frame as coordinate axes
        origin = T[0:3, 3]
        x_axis = T[0:3, 0]
        y_axis = T[0:3, 1]
        z_axis = T[0:3, 2]
        scale = 0.01  # Length of axis lines

        ax.quiver(*origin, *(scale * x_axis), color='r', linewidth=0.8)
        ax.quiver(*origin, *(scale * y_axis), color='g', linewidth=0.8)
        ax.quiver(*origin, *(scale * z_axis), color='b', linewidth=0.8)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("50 random static configurations of a 4-tendon TDCR with tip frames")
    ax.set_box_aspect([1, 1, 1])  # equal aspect ratio
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()