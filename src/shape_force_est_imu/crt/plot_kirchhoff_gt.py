#!/usr/bin/env python3
"""
Plot-only companion to generate_kirchhoff_gt_dataset.py.

Reads a saved kirchhoff_gt_dataset.npz (and optional .json metadata) and
regenerates the 3-D overview figures without re-running the solver.

Two figures are always produced:
  <stem>_overview.png          — detailed version with reference frames
  <stem>_overview_clean.png/.pdf — manuscript version (no reference frames)

Usage
-----
  python plot_kirchhoff_gt.py
  python plot_kirchhoff_gt.py --dataset gt_data/kirchhoff_gt_dataset.npz
  python plot_kirchhoff_gt.py --no-show-reference-frames
  python plot_kirchhoff_gt.py --show-sample-tip-frames
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Visualisation (mirrors generate_kirchhoff_gt_dataset.py)
# ---------------------------------------------------------------------------

_AXIS_COLORS     = ("r", "g", "b")
_REF_FRAME_SCALE = 0.010
_REF_FRAME_S     = (0.0, 0.25, 0.50, 0.75, 1.00)


def _draw_frame(ax, origin: np.ndarray, R: np.ndarray,
                scale: float = _REF_FRAME_SCALE, lw: float = 1.8) -> None:
    for col, col_idx in zip(_AXIS_COLORS, range(3)):
        ax.quiver(
            *origin, *(scale * R[:, col_idx]),
            color=col, linewidth=lw, arrow_length_ratio=0.25,
        )


def _tight_axis_limits(ax, positions_list: list[np.ndarray],
                       rod_length: float, pad_frac: float = 0.07) -> None:
    """
    Set per-axis tight bounds with a small padding margin.

    Each axis is independently sized to its data range (plus pad_frac on each
    side).  set_box_aspect([1,1,1]) still renders the visual box as a cube, so
    the figure looks properly proportioned.  Unlike the old max-span approach,
    this avoids the large empty space that appears when rod_length >> lateral
    deflections.
    """
    # include the reference backbone endpoints in the bounds
    ref_pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, rod_length]])
    all_pts = np.vstack([np.vstack(positions_list), ref_pts])
    lo, hi  = all_pts.min(0), all_pts.max(0)
    span    = hi - lo
    pad     = span * pad_frac
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])


def _save_and_show(fig, save_path: Path | None) -> None:
    plt.tight_layout()
    plt.show()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved → {save_path}")
    plt.close(fig)


def plot_overview(
    positions_list: list[np.ndarray],
    T_tip_list: list[np.ndarray],
    rod_length: float = 0.10,
    show_reference_frames: bool = True,
    hide_sample_tip_frames: bool = True,
    save_path: Path | None = None,
) -> None:
    """Detailed overview: reference backbone + optional arc-length frames."""
    fig = plt.figure(figsize=(9, 8))
    ax  = fig.add_subplot(111, projection="3d")

    n    = len(positions_list)
    cmap = plt.cm.tab20

    for k, p in enumerate(positions_list):
        color = cmap(k / max(n - 1, 1))
        lbl = "deformed shapes" if k == 0 else None
        ax.plot(p[:, 0], p[:, 1], p[:, 2], lw=1.0, alpha=0.55,
                color=color, label=lbl)
        if not hide_sample_tip_frames:
            _draw_frame(ax, T_tip_list[k][:3, 3], T_tip_list[k][:3, :3],
                        scale=0.006, lw=0.6)

    ax.plot(
        [0.0, 0.0], [0.0, 0.0], [0.0, rod_length],
        color="k", lw=1.8, zorder=5, label="reference (undeformed)",
    )

    if show_reference_frames:
        R_id = np.eye(3)
        for t in _REF_FRAME_S:
            _draw_frame(ax, np.array([0.0, 0.0, t * rod_length]), R_id,
                        scale=_REF_FRAME_SCALE, lw=1.8)

    ax.set_xlabel("x  [m]", fontsize=10)
    ax.set_ylabel("y  [m]", fontsize=10)
    ax.set_zlabel("z  [m]", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_box_aspect([1, 1, 1])
    _tight_axis_limits(ax, positions_list, rod_length)
    _save_and_show(fig, save_path)


def plot_overview_manuscript(
    positions_list: list[np.ndarray],
    rod_length: float = 0.10,
    save_stem: Path | None = None,
) -> None:
    """
    Manuscript-clean version: reference backbone only, no arc-length frames,
    no tip frames, shorter title.  Saves both PNG and PDF.
    """
    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")

    n    = len(positions_list)
    cmap = plt.cm.tab20

    for k, p in enumerate(positions_list):
        color = cmap(k / max(n - 1, 1))
        lbl = "deformed shapes" if k == 0 else None
        ax.plot(p[:, 0], p[:, 1], p[:, 2], lw=1.0, alpha=0.55,
                color=color, label=lbl)

    ax.plot(
        [0.0, 0.0], [0.0, 0.0], [0.0, rod_length],
        color="k", lw=1.8, zorder=5, label="reference (undeformed)",
    )

    ax.set_xlabel("x  [m]", fontsize=10)
    ax.set_ylabel("y  [m]", fontsize=10)
    ax.set_zlabel("z  [m]", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_box_aspect([1, 1, 1])
    _tight_axis_limits(ax, positions_list, rod_length)

    plt.tight_layout()
    plt.show()
    if save_stem is not None:
        for ext in ("png", "pdf"):
            p = save_stem.parent / (save_stem.name + f"_clean.{ext}")
            fig.savefig(p, dpi=150)
            print(f"Figure saved → {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot saved Kirchhoff-rod GT dataset (no re-simulation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str,
                        default="gt_data/kirchhoff_gt_dataset.npz",
                        help="path to the .npz dataset file")
    parser.add_argument("--out", type=str, default=None,
                        help="output PNG path (default: same dir as dataset, "
                             "same stem + _overview.png)")
    parser.add_argument("--show-reference-frames", action="store_true", default=True)
    parser.add_argument("--no-show-reference-frames", dest="show_reference_frames",
                        action="store_false")
    parser.add_argument("--hide-sample-tip-frames", action="store_true", default=True)
    parser.add_argument("--show-sample-tip-frames", dest="hide_sample_tip_frames",
                        action="store_false")
    args = parser.parse_args()

    npz_path = Path(args.dataset)
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    positions_arr = data["positions"]     # (N, num_pts, 3)
    T_tip_arr     = data["T_tip"]         # (N, 4, 4)
    orientations  = data["orientations"]  # loaded but not needed for plotting

    positions_list = [positions_arr[k] for k in range(len(positions_arr))]
    T_tip_list     = [T_tip_arr[k]     for k in range(len(T_tip_arr))]

    # Read rod_length from metadata if available, otherwise fall back to
    # the z-coordinate of the last point of the first shape.
    rod_length = 0.10
    if "meta" in data:
        try:
            meta = json.loads(str(data["meta"].item()))
            rod_length = float(meta.get("length_m", rod_length))
        except Exception:
            pass

    json_path = npz_path.with_suffix(".json")
    if json_path.exists():
        try:
            meta = json.loads(json_path.read_text())
            rod_length = float(meta.get("length_m", rod_length))
        except Exception:
            pass

    out_path   = Path(args.out) if args.out else npz_path.with_name(
        npz_path.stem + "_overview.png"
    )
    # Manuscript files: <stem>_overview_clean.png / .pdf (built inside the fn)
    clean_stem = npz_path.parent / (npz_path.stem + "_overview")

    print(f"Loaded {len(positions_list)} shapes from {npz_path}")
    print(f"Rod length: {rod_length} m")

    print("\n--- detailed overview (with reference frames) ---")
    plot_overview(
        positions_list         = positions_list,
        T_tip_list             = T_tip_list,
        rod_length             = rod_length,
        show_reference_frames  = args.show_reference_frames,
        hide_sample_tip_frames = args.hide_sample_tip_frames,
        save_path              = out_path,
    )

    print("\n--- manuscript version (clean, no reference frames) ---")
    plot_overview_manuscript(
        positions_list = positions_list,
        rod_length     = rod_length,
        save_stem      = clean_stem,
    )


if __name__ == "__main__":
    main()
